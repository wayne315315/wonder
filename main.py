import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import mixed_precision
from model import ActorCritic
from model_fn import create_ac
from data import write_data
from train import train


# Faster with CPU rather than GPU
tf.config.set_visible_devices([], 'GPU')

#write_data(p_data, num_game, fn_model, fn_others=[fn_other])
def gen_data(p_data, p_model, p_other="", num_game=10):
    model = tf.keras.models.load_model(p_model)
    fn_model = tf.function(lambda state, hand: model([state, hand])[:2]).get_concrete_function(
            tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32)
        )
    if p_other:
        other = tf.keras.models.load_model(p_other)
        fn_other = tf.function(lambda state, hand: other([state, hand])[:2]).get_concrete_function(
                tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
                tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            )
    else:
        fn_other = None
    # write TFRecord
    t1 = time.time()
    write_data(p_data, num_game, fn_model, fn_others=[fn_other])
    t2 = time.time()
    print(f"Gen data Time taken: {t2 - t1} seconds")


def main(p_model, p_other="", p_data="data/meow.tfrecord", policy="mixed_float16", round=100):
    mixed_precision.set_global_policy(policy)
    # path
    for r in range(round):
        gen_data(p_data, p_model, p_other=p_other)
        t1 = time.time()
        train(p_data, p_model, epoch=10, learning_rate=1e-4, batch_size=4096)
        t2 = time.time()
        print(f"Train Time taken: {t2 - t1} seconds")
        print(f"Round {r} completed", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_float16/meow.keras")
    parser.add_argument("--other", type=str, default="")
    parser.add_argument("--data", type=str, default="data/meow.tfrecord")
    parser.add_argument("--policy", type=str, default="mixed_float16")
    parser.add_argument("--round", type=int, default=100)
    args = parser.parse_args()
    main(args.model, p_other=args.other, p_data=args.data, policy=args.policy, round=args.round)
