import time
import ray
from ftp import RayFTPClient

import tensorflow as tf
from tensorflow.keras import mixed_precision
from model import ActorCritic
from model_fn import create_ac
from data import write_data
from train import train


# master node generate data
def master(p_data, p_model, p_other="", num_game=10, policy="mixed_float16"):
    t1 = time.time()
    # download model from FTP server
    with RayFTPClient() as ftp:
        # download the latest model
        ftp.download(p_model)

    # set policy
    mixed_precision.set_global_policy(policy)

    # load model and concrete function
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
    write_data(p_data, num_game, fn_model, fn_others=[fn_other])
    # upload data to FTP server
    with RayFTPClient() as ftp:
        ftp.upload(p_data)
    t2 = time.time()
    print(f"Master took: {t2 - t1} seconds")


@ray.remote(num_gpus=1, resources={"arch_x86": 0.999, "gpu_type_cuda": 1})
def worker(p_data, p_model, policy="mixed_float16", round=100, batch_size=128):
    t1 = time.time()
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from model import ActorCritic
    from model_fn import create_ac
    from train import train
    # download data and model from FTP server
    with RayFTPClient() as ftp:
        ftp.download(p_data)
        ftp.download(p_model)

    # set policy
    mixed_precision.set_global_policy(policy)
    # train
    train(p_data, p_model, epoch=10, learning_rate=1e-4, batch_size=batch_size)

    # upload model to FTP server
    with RayFTPClient() as ftp:
        ftp.upload(p_model)

    t2 = time.time()
    print(f"Master took: {t2 - t1} seconds")


def main(p_model, p_other="", p_data="data/woof.tfrecord", policy="mixed_float16", round=100):
    mixed_precision.set_global_policy(policy)
    # path
    for r in range(round):
        master(p_data, p_model, p_other=p_other)
        future = worker.remote(p_data, p_model, policy="mixed_float16", round=100, batch_size=128)
        ray.get(future)
        print(f"Round {r} completed", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_float16/woof.keras")
    parser.add_argument("--other", type=str, default="")
    parser.add_argument("--data", type=str, default="data/woof.tfrecord")
    parser.add_argument("--policy", type=str, default="mixed_float16")
    parser.add_argument("--round", type=int, default=100)
    args = parser.parse_args()
    main(args.model, p_other=args.other, p_data=args.data, policy=args.policy, round=args.round)
