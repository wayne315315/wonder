import tensorflow as tf
from tqdm import tqdm

from const import CARDS
from exploit import compute_loss_ppo
from example import parse_example


#tf.config.set_visible_devices([], 'GPU')


input_signature = (
    tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32), # vs
    tf.TensorSpec(shape=[None, None], dtype=tf.int32), # hs
    tf.TensorSpec(shape=[None], dtype=tf.int32), # ys
    tf.TensorSpec(shape=[None], dtype=tf.float32), # rs
    tf.TensorSpec(shape=[None, 3 * len(CARDS)], dtype=tf.float32) # ls
)

def train(p_data, p_model, epoch=10, learning_rate=1e-4, batch_size=512):
    # model
    model = tf.keras.models.load_model(p_model)
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # metrices
    metrices_acc = [tf.Variable(0.0, trainable=False) for _ in range(5)]
    # grads
    grads_acc = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32), trainable=False) for tv in model.trainable_variables]

    # custom train step
    @tf.function(input_signature=input_signature)
    def train_step(v, h, y, r, l):
        # reset metrices_acc
        for metric_acc in metrices_acc:
            metric_acc.assign(0.0)
        with tf.GradientTape() as tape:
            logits, values = model([v, h], training=True)
            metrices = compute_loss_ppo(logits, values, y, r, l)
        loss = metrices[0]
        # compute grads
        grads = tape.gradient(loss, model.trainable_variables)
        # accumulate grads
        for i in range(len(model.trainable_variables)):
            if grads[i] is not None:
                grads_acc[i].assign_add(grads[i])
        # accumulate metrices
        for metric, metric_acc in zip(metrices, metrices_acc):
            if metric is not None:
                metric_acc.assign_add(metric)
        return tuple(metrices_acc)
    
    # load the data
    raw = tf.data.TFRecordDataset(p_data).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    for i, item in enumerate(raw):
        d = tf.data.Dataset.from_tensor_slices(item).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.concatenate(d) if i > 0 else d
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # run ppo
    for e in tqdm(range(epoch)):
        # reset grads_acc
        for grad_acc in grads_acc:
            grad_acc.assign(tf.zeros_like(grad_acc))
        # replay
        losses = []
        losses_actor = []
        losses_critic = []
        losses_entropy = []
        expected_returns = []
        
        # compute gradients & metrices
        total = 0
        for v, h, y, r, l in dataset:
            loss, loss_actor, loss_critic, loss_entropy, expected_return  = train_step(v,h,y,r,l)
            losses.append(loss.numpy())
            losses_actor.append(loss_actor.numpy())
            losses_critic.append(loss_critic.numpy())
            losses_entropy.append(loss_entropy.numpy())
            expected_returns.append(expected_return.numpy())
            n = v.shape[0]
            total += n
            tf.print("batch size:", n)
            tf.print("loss:", loss/n)
            tf.print("loss actor:", loss_actor/n)
            tf.print("loss critic:", loss_critic/n)
            tf.print("loss entropy:", loss_entropy/n)
            tf.print("expected return:", expected_return/n)

        # compute the mean for all matrices in this epoch
        loss_avg = sum(losses)/total
        loss_actor_avg = sum(losses_actor)/total
        loss_critic_avg = sum(losses_critic)/total
        loss_entropy_avg = sum(losses_entropy)/total
        expected_return_avg = sum(expected_returns)/total
        print("epoch:", e)
        print("loss: %.2E" % loss_avg)
        print("loss actor: %.2E" % loss_actor_avg)
        print("loss critic: %.2E" % loss_critic_avg)
        print("loss entropy: %.2E" % loss_entropy_avg)
        print("expected return: %.2E" % expected_return_avg)

        # apply grads for each epoch
        optimizer.apply_gradients(zip(grads_acc, model.trainable_variables))
        # save model
        model.save(p_model)
        print("model saved", flush=True)

if __name__ == "__main__":
    import time
    p_data = "gs://wayne315315/wonder/data/exploiter.tfrecord"
    p_model = "gs://wayne315315/wonder/model/exploiter.keras"
    """
    p_data = "data/exploiter.tfrecord"
    p_model = "model/exploiter.keras"
    """
    t1 = time.time()
    train(p_data, p_model, epoch=10, learning_rate=1e-4, batch_size=2048)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
