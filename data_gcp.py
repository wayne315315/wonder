"""Generate data sample on GCP with CPU"""
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import tensorflow as tf
from model import ActorCritic
from game import Game
from data import epi_gen
from rl import AIPlayer3
from train import translate
from example import create_example

#tf.config.set_visible_devices([], 'GPU')


def generate(num_game, fn_model, fn_exploiter):
    games = [Game(n, random_face=False) for _ in range(num_game) for n in range(3, 8)]
    for game in games:
        players = [AIPlayer3(fn_model) for _ in range(game.n)]
        players[0] = AIPlayer3(fn_exploiter)
        for i in range(game.n):
            game.register(i, players[i])
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2n = {executor.submit(epi_gen, game): game.n for game in games}
        for f in as_completed(f2n):
            episodes = f.result()
            for episode in episodes:
                yield episode

def main(num_game, batch_size=512):
    p_data = "gs://wayne315315/wonder/data/exploiter.tfrecord"
    p_model = "gs://wayne315315/wonder/model/base.keras"
    p_exploiter = "gs://wayne315315/wonder/model/exploiter.keras"
    """
    p_data = "data/exploiter.tfrecord"
    p_model = "model/base.keras"
    p_exploiter = "model/exploiter.keras"
    """
    model = tf.keras.models.load_model(p_model)
    exploiter = tf.keras.models.load_model(p_exploiter)
    fn_model = model.predict_move.get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    fn_exploiter = exploiter.predict_move.get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    total = num_game * 5
    data_iterator = generate(num_game, fn_model, fn_exploiter)
    vs = defaultdict(list)
    hs = defaultdict(list)
    ys = defaultdict(list)
    rs = defaultdict(list)
    ls = defaultdict(list) # old logits
    for episode in tqdm(data_iterator, total=total):
        vs_, hs_, ys_, rs_ = translate(episode)
        for key in vs_:
            vs[key].append(vs_[key])
            hs[key].append(hs_[key])
            ys[key].append(ys_[key])
            rs[key].append(rs_[key])
    writer = tf.io.TFRecordWriter(p_data)
    for key in vs:
        for item in [vs, hs, ys, rs]:
            item[key] = tf.concat(item[key], axis=0)
        logits, _ = exploiter.predict([vs[key], hs[key]], batch_size=batch_size)
        ls[key] = logits
        v, h, y, r, l = vs[key], hs[key], ys[key], rs[key], ls[key]
        example = create_example(v, h, y, r, l)
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_game', type=int, default=100, help='number of games per player count')
    args = parser.parse_args()

    t1 = time.time()
    main(args.num_game)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
