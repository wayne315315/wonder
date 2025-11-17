from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from uuid import uuid4
import logging
import random

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from player import RandomPlayer
from rl import AIPlayer
from game import Game
from model import ActorCritic


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

def test(num_game, fn_model=None, fn_others=[None], w_others=None):
    """ Print the win loss ratio and average rank of the model against other models
    Args:
        num_game: The number of games for each number of the total players
        fn_model: The concrete function of the model for the AI player 0; None for RandomPlayer
        fn_others: The concrete functions of the models for the other AI players; None for RandomPlayer
        w_others: The weights for the other AI players; should sum to 1.0; len(w_others) should be equal to len(fn_others); 
            if None, the w_others will be set equal weight accordingly
    """
    if w_others is None:
        w_others = [1.0 / len(fn_others)] * len(fn_others)
    assert len(fn_others) == len(w_others)
    # Create games and register players
    games = [Game(n, random_face=False) for n in range(3, 8) for _ in range(num_game)]
    for game in games:
        fns = [fn_model] + random.choices(fn_others, weights=w_others, k=game.n-1)
        players = [AIPlayer(fn) if fn else RandomPlayer() for fn in fns]
        for i in range(game.n):
            game.register(i, players[i])
    # run games in parallel
    totals = defaultdict(list)
    ranks = defaultdict(list)
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2n = {executor.submit(test_single, game): game.n for game in games}
        for future in tqdm(as_completed(f2n), total=len(f2n)):
            history, message = future.result()
            #print(message)
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

    # Faster with CPU rather than GPU
    tf.config.set_visible_devices([], 'GPU')

    # loading models and concrete functions
    p_model = "model/base.keras"
    p_exploiter = "model/exploiter.keras"
    model = tf.keras.models.load_model(p_model)
    exploiter = tf.keras.models.load_model(p_exploiter)
    fn_model = tf.function(lambda state, hand: model([state, hand])[:2]).get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    fn_exploiter = tf.function(lambda state, hand: exploiter([state, hand])[:2]).get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    num_game = 10
    t1 = time.time()
    test(num_game, fn_model=fn_model, fn_others=[None], w_others=None)
    #test(num_game, fn_model=fn_exploiter, fn_others=[fn_model], w_others=None)
    t2 = time.time()
    print("Elapsed time : %.2f second" % (t2 - t1))
