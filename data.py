import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from player import RandomPlayer
from rl import AIPlayer
from game import Game
from helper import Adaptor
from example import create_example


def translate(episode, gamma=0.9, penalty=-1.0):
    rec_valid, rec_invalid = episode
    # Task 1 : translate rewards to expected return with discounted factor gamma
    n = len(rec_valid)
    for i, rec in enumerate(rec_valid):
        rec[-1] *= gamma ** (n - 1 - i) # rec[-1] : reward -> expected return (discounted w/ gamma)
    # Task 2 : remove all record which api call is not 'move'
    rec_valid = [rec for rec in rec_valid if rec[0] == "move"]
    # Task 3 : add penalty to rec_invalid & sample n examples from rec_invalid
    if rec_invalid:
        rec_invalid_ = [[rec[0], rec[1], rec[2], penalty] for rec in rec_invalid]
        random.shuffle(rec_invalid_)
        m = len(rec_invalid)
        n = len(rec_valid)
        rec_invalid = []
        while n:
            if n >= m:
                rec_invalid += rec_invalid_
                n -= m
            else:
                rec_invalid += rec_invalid_[:n]
                n = 0
    # Task 4 : Collect all records (valid : invalid = 1 : 1)
    recs = rec_valid + rec_invalid
    # Task 5 : convert state, record, hand to v, h; pick, action to y
    adaptor = Adaptor()
    vs = defaultdict(list)
    hs = defaultdict(list)
    ys = defaultdict(list)
    rs = defaultdict(list)
    for rec in recs:
        api, (state, record, hand), (pick, action), r = rec
        v = adaptor.s2v(state, record)[0]
        h = adaptor.h2v(hand)[0]
        y = adaptor.pair2idx[(pick, action)]
        key = (v.shape[0], h.shape[0])
        vs[key].append(v)
        hs[key].append(h)
        ys[key].append(y)
        rs[key].append(r)
    # vectorize into numpy arrays
    for key in vs:
        for item in [vs, hs, ys]:
            item[key] = np.array(item[key], dtype=np.int32)
        rs[key] = np.array(rs[key], dtype=np.float32)
    return vs, hs, ys, rs


def get_reward(scores):
    totals, coins = scores["total"], scores["coin"]
    x = list(zip(totals, coins)) # eg. [(42, 4), (61,2), (42, 5)]
    n = len(x)
    y = [0] * n
    for i in range(n):
        for j in range(n):
            if x[i] > x[j]:
                y[i] += 1
            elif x[i] < x[j]:
                y[i] -= 1
    y = np.asarray(y)
    y = np.sign(y) * np.power(np.abs(y), 2)
    reward = y[0] + sum([totals[0] - totals[i] for i in range(n)])
    reward /= n
    return reward


def extract(history):
    rec_valid = []
    rec_invalid = []
    _, [scores, *_], _, _ = history[0].pop() # (api, args, res, is_valid) : (str, list, list, bool)
    reward = get_reward(scores)
    for rec in history[0]:
        api, args, res, is_valid = rec
        if is_valid == True:
            rec = [api, args, res, reward]
            rec_valid.append(rec) # rec: (api, args, res, reward)
        else:
            rec = [api, args, res]
            rec_invalid.append(rec) # rec: (api, args, res)
    episode = (rec_valid, rec_invalid)
    return episode


def epi_gen(game, gamma=0.9, penalty=-1.0):
    history = game.run(verbose=50)
    episode = extract(history)
    vs, hs, ys, rs = translate(episode, gamma=gamma, penalty=penalty)
    return vs, hs, ys, rs


def data_gen(num_game, fn_model=None, fn_others=[None], w_others=None, gamma=0.9, penalty=-1.0, max_workers=4):
    """ Generate data for training/evaluation
    Args:
        num_game: The number of games for each number of the total players
        fn_model: The concrete function of the model for the AI player 0; None for RandomPlayer
        fn_others: The concrete functions of the models for the other AI players; None for RandomPlayer
        w_others: The weights for the other AI players; should sum to 1.0; len(w_others) should be equal to len(fn_others); 
            if None, the w_others will be set equal weight accordingly
    Yields:
        episode: A tuple of (rec_valid, rec_invalid) for player 0;
            rec_valid: A list of valid records; Each record is a tuple of (api, args, res, reward)
            rec_invalid: A list of invalid records; Each record is a tuple of (api, args, res)
    """
    if w_others is None:
        w_others = [1.0 / len(fn_others)] * len(fn_others)
    assert len(fn_others) == len(w_others)

    # Create all games
    games = [Game(n, random_face=False) for n in range(3, 8) for _ in range(num_game)]
    for game in games:
        fns = [fn_model] + random.choices(fn_others, weights=w_others, k=game.n-1)
        players = [AIPlayer(fn) if fn else RandomPlayer() for fn in fns]
        for i in range(game.n):
            game.register(i, players[i])

    # Start generating episodes
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2n = {executor.submit(lambda g:epi_gen(g, gamma=gamma, penalty=penalty), game): game.n for game in games}
        for f in as_completed(f2n):
            vs, hs, ys, rs = f.result()
            yield (vs, hs, ys, rs)

def write_data(p_data, num_game, fn_model, fn_others=[None], w_others=None, gamma=0.9, penalty=-1.0, max_workers=4, batch_size=4096):
    data_iterator = data_gen(num_game, fn_model=fn_model, fn_others=fn_others, w_others=w_others, gamma=gamma, penalty=penalty, max_workers=max_workers)
    vs = defaultdict(list)
    hs = defaultdict(list)
    ys = defaultdict(list)
    rs = defaultdict(list)
    ls = defaultdict(list) # old logits
    for vs_, hs_, ys_, rs_ in tqdm(data_iterator, total=num_game * 5):
        for key in vs_:
            vs[key].append(vs_[key])
            hs[key].append(hs_[key])
            ys[key].append(ys_[key])
            rs[key].append(rs_[key])
    ###
    writer = tf.io.TFRecordWriter(str(p_data))
    for key in vs:
        for item in [vs, hs, ys, rs]:
            item[key] = tf.concat(item[key], axis=0)
        ls[key] = tf.zeros_like(rs[key], dtype=tf.float32)
        v, h, y, r, l = vs[key], hs[key], ys[key], rs[key], ls[key]
        example = create_example(v, h, y, r, l)
        writer.write(example.SerializeToString())
    writer.close()
    ###
    """
    writer = tf.io.TFRecordWriter(str(p_data))
    for key in vs:
        for item in [vs, hs, ys, rs]:
            item[key] = tf.concat(item[key], axis=0)
        inputs = tf.data.Dataset.from_tensor_slices((vs[key], hs[key])).batch(batch_size)
        ls[key] = tf.concat([fn_model(v, h)[0] for v, h in inputs], axis=0)
        v, h, y, r, l = vs[key], hs[key], ys[key], rs[key], ls[key]
        example = create_example(v, h, y, r, l)
        writer.write(example.SerializeToString())
    writer.close()
    """


if __name__ == "__main__":
    import time
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from model import ActorCritic

    # Faster with CPU rather than GPU
    tf.config.set_visible_devices([], 'GPU')

    # Set mixed precision policy
    mixed_precision.set_global_policy('mixed_float16')

    # path
    p_data = "data/exploiter.tfrecord"
    p_model = "model/exploiter.keras"
    p_other = "model/base.keras"

    # load model and its concrete function
    model = tf.keras.models.load_model(p_model)
    fn_model = tf.function(lambda state, hand: model([state, hand])[:2]).get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    other = tf.keras.models.load_model(p_other)
    fn_other = tf.function(lambda state, hand: other([state, hand])[:2]).get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    # write TFRecord
    num_game = 10
    t1 = time.time()
    write_data(p_data, num_game, fn_model, fn_others=[fn_other])
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
