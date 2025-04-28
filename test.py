from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from uuid import uuid4
import logging

import numpy as np
import tensorflow as tf

from player import RandomPlayer
from rl import AIPlayer2
from game import Game
from model import ActorCritic
from serving import launch_server, kill_server
from serving import export_archive, clean_archive
from serving import probe


def test_single(game):
    uid = str(uuid4())
    game.logger = logging.getLogger(uid)
    game.logger.propagate = False
    with StringIO() as f:
        handler = logging.StreamHandler(stream=f)
        game.logger.addHandler(handler)
        history = game.run()
        message = f.getvalue()
    return history, message

def test(modelpath, num_play=10, num_game=10):
    # loading model
    model = tf.keras.models.load_model(modelpath)
    # export base model version 0
    serve_name = "ac"
    t1 = time.time()
    export_archive(serve_name, model, 0)
    t2 = time.time()
    print("Export model took %.2f second" % (t2 - t1))
    # launch server
    t1 = time.time()
    container_id = launch_server()
    # probe server until it is ready
    while not probe(serve_name, serve_version=None, verbose=False):
        pass
    t2 = time.time()
    print("Server took %.2f second to start" % (t2 - t1))
    # Create games and register players
    games = [Game(n, random_face=False) for n in range(3, 8) for _ in range(num_game)]
    games = [deepcopy(g) for g in games for _ in range(num_play)]
    for game in games:
        players = [RandomPlayer() for _ in range(game.n)]
        players[0] = AIPlayer2(serve_name)
        for i in range(game.n):
            game.register(i, players[i])
    # run games in parallel
    totals = defaultdict(list)
    ranks = defaultdict(list)
    max_workers = num_play * num_game * 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2n = {executor.submit(test_single, game): game.n for game in games}
        for future in as_completed(f2n):
            history, message = future.result()
            print(message)
            # process history
            _, [scores, *_], _, _ = history[0].pop()
            total = scores["total"]
            n = len(total)
            rank = [0] * n
            for i in range(n):
                for j in range(n):
                    if total[i] > total[j]:
                        rank[i] += 1
                    elif total[i] < total[j]:
                        rank[i] -= 1
            totals[n].append(total)
            ranks[n].append(rank)
    # kill server
    kill_server(container_id)

    # clean archive
    clean_archive(serve_name)

    # calculate metrices
    for items in [totals, ranks]:
        for n, item in items.items():
            item = np.asarray(item)
            item = np.mean(item, axis=0)
            items[n] = item
    
    for n in range(3, 8):
        total = totals[n]
        rank = ranks[n]
        print(total, rank)


if __name__ == "__main__":
    from pathlib import Path
    import time

    model_dir = Path("model")
    model_path = Path(model_dir, "ac.keras")

    t1 = time.time()
    test(model_path, num_play=10, num_game=10)
    t2 = time.time()
    print("Elapsed time : %.2f second" % (t2 - t1))
