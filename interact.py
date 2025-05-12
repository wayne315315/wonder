from threading import Thread, Condition
from queue import Queue
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import random

from flask import Flask, send_from_directory, jsonify, request, make_response
from flask_socketio import SocketIO, emit

from game import Game, CIVS, CARDS
from web import WebHumanPlayer, WebRandomPlayer
from utils import KeyedRemovableQueue as KRQ


app = Flask("Seven Wonders")
app.tasks = defaultdict(Queue) # tasks[uid] = Queue([Task])
app.done = defaultdict(list) # done[uid] = [Task] i.e. the tasks which the player have finished in the current game; used for reloading; only include notice tasks
app.events = defaultdict(list) # events[uid][0] = e
app.res = {} # res[uid] = {"pick": "Palace", "action": "WONDER", "coins": [1, 2]}
app.url = "http://127.0.0.1:5000"
app.active = {} # active[uid] = gid
app.history = defaultdict(dict) # history[gid][uid] 
app.game = KRQ() # KRQ([{uid:{"players": ["H", "R", "R"], "random_face": False}}])
app.join = KRQ() # KRQ[{uid:uid}]) for waiting players able to cancel by uid
socketio = SocketIO(app)

web_dir = Path(Path(__file__).parent, "app")
c = Condition()
u2s = {}

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
            "type": "GAME",
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

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.event
def create(payload):
    uid = payload.pop("uid")
    with c:
        app.game.enqueue(uid, payload)
        c.notify()
    sid = request.sid
    u2s[uid] = sid
    socketio.emit("create", to=sid)

@socketio.event
def join(uid):
    with c:
        app.join.enqueue(uid, uid)
        c.notify()
    sid = request.sid
    u2s[uid] = sid
    socketio.emit("join", to=sid)

@socketio.event
def cancel_create(uid):
    app.game.remove_by_key(uid)
    sid = u2s[uid]
    socketio.emit("cancel_create", to=sid)

@socketio.event
def cancel_join(uid):
    app.join.remove_by_key(uid)
    sid = u2s[uid]
    socketio.emit("cancel_join", to=sid)

@socketio.event
def game():
    p2p = {"H": WebHumanPlayer, "R": WebRandomPlayer}
    # use thrading.condition c to avoid excess looping
    while True:
        with c:
            while app.game.is_empty():
                c.wait()
            uid_host, payload = app.game.peek()
            players = payload["players"]
            random_face = payload["random_face"]
            num_others = players[1:].count("H") # number of other human players
            if len(app.join) < num_others:
                c.wait()
            else:
                # pop the setting from game queue
                app.game.dequeue()
                # pop the players from join queue and shuffle
                uids_others = [app.join.dequeue() for _ in range(num_others)]
                random.shuffle(uids_others)
                # collect all human players for use later
                uids_h = [uid_host] + uids_others
                # Create players
                uids = [uid_host] + [uids_others.pop() if p == "H" else str(uuid4()) for p in players[1:]]
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
                # emit event to all human players
                for uid in uids_h:
                    sid = u2s[uid]
                    socketio.emit("game", {"gid": gid}, to=sid)
                    print(f"Game {gid} started for {uid}")

if __name__ == '__main__':
    t = Thread(target=game)
    t.start()
    socketio.run(app, debug=True)
