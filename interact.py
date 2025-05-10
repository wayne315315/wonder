from threading import Thread, Event
from queue import Queue
from pathlib import Path
from uuid import uuid4
from collections import defaultdict

from flask import Flask, send_from_directory, jsonify, request, make_response

from game import Game, CIVS, CARDS
from web import WebHumanPlayer, WebRandomPlayer


app = Flask("Seven Wonders")
app.tasks = defaultdict(Queue) # tasks[uid] = Queue([Task])
app.done = defaultdict(list) # done[uid] = [Task] i.e. the tasks which the player have finished in the current game; used for reloading; only include notice tasks
app.events = defaultdict(list) # events[uid][0] = e
app.res = {} # res[uid] = {"pick": "Palace", "action": "WONDER", "coins": [1, 2]}
app.url = "http://127.0.0.1:5000"
app.active = {} # active[uid] = gid
app.history = defaultdict(dict) # history[gid][uid] 

web_dir = Path(Path(__file__).parent, "app")

@app.route('/')
def index():
    resp = make_response(send_from_directory(web_dir, "index.html"))
    if request.cookies.get("uid") is None:
        uid = str(uuid4())
        resp.set_cookie("uid", uid)
    return resp

@app.route("/<path:filepath>")
def getFile(filepath):
    return send_from_directory(web_dir, filepath)

@app.route('/card.json')
def card():
    return send_from_directory(".", "card.json")

@app.route('/enqueue', methods=['POST'])
def enqueue():
    uid = request.cookies.get("uid")
    task = request.get_json()
    app.tasks[uid].put(task)
    return make_response("Success", 200)

@app.route("/dequeue")
def dequeue():
    uid = request.cookies.get("uid")
    if uid is None:
        return 
    # game not exist
    if uid not in app.active:
        task = {
            "type": "CREATE",
        }
    # game exists
    else:
        task = app.tasks[uid].get()
        # any notice should be added to done for game replay or reloading
        if task["type"] in {"SETTING", "AGE", "UPDATE", "CLEAR", "BATTLE", "SCORE"}:
            app.done[uid].append(task)
        # game finished
        if task["type"] == "SCORE":
            gid = app.active.pop(uid)
            app.history[gid][uid] = app.done.pop(uid) # register task list for game replay
            app.tasks.pop(uid) # remove task queue just in case
    return task

@app.route("/submit", methods=['POST'])
def submit():
    # submit the result of the task
    uid = request.cookies.get("uid")
    res = request.get_json()
    app.res[uid] = res
    # set the event so player can continue
    if app.events[uid]:
        e = app.events[uid][0]
        e.set()
        app.events[uid].clear()
    return make_response("Success", 200)

@app.route("/execute")
def execute():
    uid = request.cookies.get("uid")
    res = app.res.pop(uid)
    return res

@app.route("/game", methods=['POST'])
def game():
    data = request.get_json()
    p2p = {"H": WebHumanPlayer, "R": WebRandomPlayer}
    players = data["players"]
    uids = data["uids"]
    random_face = data["random_face"]
    players = [p2p[p](uid, app.url, app.events[uid]) for uid, p in zip(uids, players)]
    # loop other non human players except itself
    for player in players[1:]:
        t = Thread(target=player.loop)
        t.start()
    # create game and register players
    n = len(players)
    game = Game(n, random_face=random_face)
    for i in range(n):
        game.register(i, players[i])
    t = Thread(target=game.run)
    t.start()
    # register gid
    gid = str(uuid4())
    for uid in uids:
        app.active[uid] = gid
    return make_response("Success", 200)

if __name__ == '__main__':
    app.run(debug=True)
