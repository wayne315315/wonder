import time
import ray
from pathlib import Path
from ftp import RayFTPClient


# master node generate data
def master(p_data, p_model, p_other="", num_game=10, policy="mixed_float16", start_turn=0):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from model import ActorCritic
    from model_fn import create_ac
    from data import write_data

    t1 = time.time()
    # set policy
    tf.config.set_visible_devices([], 'GPU')
    mixed_precision.set_global_policy(policy)

    # download model from FTP server
    t3 = time.time()
    try:
        with RayFTPClient() as ftp:
            # download the latest model
            ftp.download(p_model)
            #ftp.download(p_other) ###
    except Exception as e:
        print(f"Model {p_model} not found on FTP server: {str(e)}")
        # create a new model if not found
        model = create_ac()
        model.save(p_model)
        # upload model to FTP server
        with RayFTPClient() as ftp:
            ftp.upload(p_model)
    t4 = time.time()
    print(f"Master download took: {t4 - t3} seconds")

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
    write_data(p_data, num_game, fn_model, fn_others=[fn_other, None], start_turn=start_turn)
    # upload data to FTP server
    t3 = time.time()
    with RayFTPClient() as ftp:
        ftp.upload(p_data)
    t4 = time.time()
    print(f"Master upload took: {t4 - t3} seconds")
    t2 = time.time()
    print(f"Master took: {t2 - t1} seconds")


@ray.remote(num_gpus=1, resources={"arch_x86": 0.999, "gpu_type_cuda": 1})
def worker(p_data, p_model, p_optimizer, policy="mixed_float16", epoch=10, batch_size=128, learning_rate=1e-4):
    t1 = time.time()
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from model import ActorCritic
    from model_fn import create_ac
    from train import train
    from ftp import RayFTPClient

    # download data and model from FTP server
    t3 = time.time()
    with RayFTPClient() as ftp:
        ftp.download(p_data)
        ftp.download(p_model)
        #ftp.download(p_optimizer)

    t4 = time.time()
    print(f"Worker download took: {t4 - t3} seconds")

    # set policy
    mixed_precision.set_global_policy(policy)
    # train
    train(p_data, p_model, p_optimizer, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size)

    # upload model to FTP server
    t3 = time.time()
    with RayFTPClient() as ftp:
        ftp.upload(p_model)
        #ftp.upload(p_optimizer)
    t4 = time.time()
    print(f"Worker upload took: {t4 - t3} seconds")

    t2 = time.time()
    print(f"Worker took: {t2 - t1} seconds")


def main(p_model, p_other="model_float16/base.keras", policy="mixed_float16", num_game=10, epoch=10, round=100, batch_size=128, learning_rate=1e-4, start_turn=0):
    # initialize ray
    ray.init(
        address='auto', 
        runtime_env={
            "working_dir": ".",
            "excludes": ["app/", "data/", "model_bfloat16/", "model_float16/", "model_float32/"]
        }
    )
    # path
    p_data = Path("data", Path(p_model).stem).with_suffix(".tfrecord")

    for r in range(round):
        while True:
            try:
                master(p_data, p_model, p_other=p_other, policy=policy, num_game=num_game, start_turn=start_turn)
            except Exception as e:
                print(f"Error in master: {e}, retrying...")
            else:
                break
        while True:
            p_optimizer = Path("/tmp/optimizer", Path(p_model).stem)
            try:
                future = worker.remote(p_data, p_model, p_optimizer, policy=policy, epoch=epoch, batch_size=batch_size, learning_rate=learning_rate)
                ray.get(future)
            except Exception as e:
                print(f"Error in worker: {e}, retrying...")
            else:
                break
        print(f"Round {r} completed", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_float16/woof.keras")
    parser.add_argument("--other", type=str, default="")
    parser.add_argument("--policy", type=str, default="mixed_float16")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--start_turn", type=int, default=0)
    args = parser.parse_args()
    main(args.model, p_other=args.other, policy=args.policy, num_game=args.num_game, epoch=args.epoch, round=args.round, batch_size=args.batch_size, learning_rate=args.learning_rate, start_turn=args.start_turn)