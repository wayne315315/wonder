if __name__ == "__main__":
    import time
    import tensorflow as tf
    from train import train
    
    # path
    p_data = "gs://wayne315315/wonder/data/exploiter.tfrecord"
    p_model = "gs://wayne315315/wonder/model/exploiter.keras"

    # Faster with CPU rather than GPU
    #tf.config.set_visible_devices([], 'GPU')
    # Start training
    t1 = time.time()
    train(p_data, p_model, epoch=10, learning_rate=1e-4, batch_size=2048)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
