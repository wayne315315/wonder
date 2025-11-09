if __name__ == "__main__":
    import argparse
    import time

    import tensorflow as tf
    from model import ActorCritic
    from data import write_data

    # Faster with CPU rather than GPU
    #tf.config.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_game', type=int, default=100, help='number of games per player count')
    args = parser.parse_args()

    # path
    p_data = "gs://wayne315315/wonder/data/exploiter.tfrecord"
    p_model = "gs://wayne315315/wonder/model/exploiter.keras" # model to be trained
    p_other = "gs://wayne315315/wonder/model/base.keras" # models to be played against

    # load model
    other = tf.keras.models.load_model(p_other)
    fn_other = other.predict_move.get_concrete_function(
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
    t1 = time.time()
    write_data(p_data, p_model, args.num_game, fn_others=[fn_other])
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
