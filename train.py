import tensorflow as tf
from tqdm import tqdm

from const import CARDS
from example import parse_example


input_signature = (
    tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32), # vs
    tf.TensorSpec(shape=[None, None], dtype=tf.int32), # hs
    tf.TensorSpec(shape=[None], dtype=tf.int32), # ys
    tf.TensorSpec(shape=[None], dtype=tf.float32), # rs
    tf.TensorSpec(shape=[None, 3 * len(CARDS)], dtype=tf.float32) # ls
)

def compute_loss_ppo(logits, values, actions, rewards, logits_old, epsilon=0.2, entropy_coeff=0.01):
    # inputs : v, h, y_true -> outputs : p(a|s), v(s)
    # inputs : rewards -> outputs: g(a,s)
    # vanilla loss_actor = -log(p(a|s)) * (g(a,s) - v(s))
    # ratio = p(a|s) / p_old(a|s)
    # loss_actor_ppo = -min(ratio * advantage_norm, clip(ratio, 1-epsilon, 1+epsilon) * advantage_norm)
    # loss_critic = huber_loss(g(a,s), v(s))
    # loss = loss_actor_ppo + loss_critic
    # TensorShape([None, 231]) TensorShape([None, 1]) TensorShape([None]) TensorShape([None]) TensorShape([None, 231])
    
    # loss critic
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    values = values[:,0] # v(s) TensorShape([None])
    loss_critic = huber_loss(rewards, values) # TensorShape([])
    advantages = rewards - tf.stop_gradient(values) # (g(a,s) - v(s))  TensorShape([None])
    # normalize advantages
    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-4)

    # loss entropy
    logsoftmax = tf.nn.log_softmax(logits, axis=1) # log(p(ai|s), ...) TensorShape([None, 231])
    probs = tf.exp(logsoftmax) # p(ai|s) TensorShape([None, 231])
    loss_entropy = tf.reduce_sum(probs * logsoftmax) # we wanna maximize entropy

    # loss actor ppo
    logsoftmax = tf.gather(logsoftmax, actions, batch_dims=1) # log(p(a|s)) TensorShape([None])
    logsoftmax_old = tf.nn.log_softmax(logits_old, axis=1) # log(p_old(ai|s), ...) TensorShape([None, 231])
    logsoftmax_old = tf.gather(logsoftmax_old, actions, batch_dims=1) # log(p_old(a|s)) TensorShape([None])
    log_ratio = logsoftmax - logsoftmax_old
    # kl divergence for monitoring
    kld = tf.reduce_sum(-log_ratio)
    log_ratio = tf.clip_by_value(log_ratio, -3.0, 3.0) # clip to avoid NaN
    ratio = tf.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    loss_actor_ppo = -tf.reduce_sum(tf.minimum(surr1, surr2))

    # overall loss
    loss = loss_actor_ppo + loss_critic + entropy_coeff * loss_entropy

    # metrics
    probs = tf.gather(probs, actions, batch_dims=1) # p(a|s) TensorShape([None])
    expected_return = tf.reduce_sum(probs * rewards) # E[R|a,s] TensorShape([])
    return loss, loss_actor_ppo, loss_critic, loss_entropy, expected_return, kld



def train(p_data, p_model, p_optimizer, epoch=10, learning_rate=1e-4, batch_size=512):
    # model
    model = tf.keras.models.load_model(p_model)
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    ### NEW: Setup Independent Optimizer Checkpoint ###
    # We create a Checkpoint object tracking ONLY the optimizer.
    ckpt_opt = tf.train.Checkpoint(optimizer=optimizer)

    # We use a Manager to handle file versioning (keep last 3).
    # It saves to a separate folder so it doesn't mess with your model file.
    ckpt_manager = tf.train.CheckpointManager(ckpt_opt, p_optimizer, max_to_keep=1)

    # ### NEW: Restore Optimizer State (Lazy Loading) ###
    # If a checkpoint exists, this schedules the values to be loaded.
    # They will physically load the moment 'apply_gradients' is called for the first time.
    if ckpt_manager.latest_checkpoint:
        status = ckpt_opt.restore(ckpt_manager.latest_checkpoint)
        status.expect_partial()
        print(f"Restored optimizer state from {ckpt_manager.latest_checkpoint}")
    else:
        print("No optimizer checkpoint found. Starting fresh.")
    ###

    # metrices
    metrices_acc = [tf.Variable(0.0, trainable=False) for _ in range(6)] # loss, loss_actor, loss_critic, loss_entropy, expected_return, kl_divergence
    # grads
    grads_acc = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32), trainable=False) for tv in model.trainable_variables]

    # custom train step
    @tf.function(input_signature=input_signature)
    def train_step(v, h, y, r, l):
        # reset metrices_acc
        for metric_acc in metrices_acc:
            metric_acc.assign(0.0)
        with tf.GradientTape() as tape:
            logits, _, values = model([v, h], training=True)
            metrices = compute_loss_ppo(logits, values, y, r, l)
        loss = metrices[0]
        # compute grads
        grads = tape.gradient(loss, model.trainable_variables)
        # ensure loss and grads are finite number, not +inf, -inf, or NaN
        is_finite = tf.reduce_all(tf.math.is_finite(loss))
        is_finite &= tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None])

        # accumulate grads
        if is_finite:
            for i in range(len(model.trainable_variables)):
                if grads[i] is not None:
                    grads_acc[i].assign_add(grads[i])
        else:
            tf.print("Non-finite loss or gradients detected. Skipping gradient update for this batch.")
        # accumulate metrices
        for metric, metric_acc in zip(metrices, metrices_acc):
            if metric is not None:
                metric_acc.assign_add(metric)
        return tuple(metrices_acc)
    
    # load the data
    raw = tf.data.TFRecordDataset(p_data).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    for i, item in enumerate(raw):
        d = tf.data.Dataset.from_tensor_slices(item).map(lambda v,h,y,r,l: (v,h,y,r), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.concatenate(d) if i > 0 else d
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # precompute old logits
    for i, (v, h, _, _) in enumerate(dataset):
        l = tf.data.Dataset.from_tensors(model([v, h])[0])
        ls = ls.concatenate(l) if i > 0 else l
    dataset = tf.data.Dataset.zip((dataset, ls)).map(lambda item, l: (*item,l), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

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
        klds = []
        
        # compute gradients & metrices
        total = 0
        for v, h, y, r, l in dataset:
            loss, loss_actor, loss_critic, loss_entropy, expected_return, kld = train_step(v,h,y,r,l)
            losses.append(loss.numpy())
            losses_actor.append(loss_actor.numpy())
            losses_critic.append(loss_critic.numpy())
            losses_entropy.append(loss_entropy.numpy())
            expected_returns.append(expected_return.numpy())
            klds.append(kld.numpy())
            n = v.shape[0]
            total += n
            """
            tf.print("batch size:", n)
            tf.print("loss:", loss/n)
            tf.print("loss actor:", loss_actor/n)
            tf.print("loss critic:", loss_critic/n)
            tf.print("loss entropy:", loss_entropy/n)
            tf.print("kl divergence:", kld/n)
            tf.print("expected return:", expected_return/n)
            """

        # compute the mean for all matrices in this epoch
        loss_avg = sum(losses)/total
        loss_actor_avg = sum(losses_actor)/total
        loss_critic_avg = sum(losses_critic)/total
        loss_entropy_avg = sum(losses_entropy)/total
        expected_return_avg = sum(expected_returns)/total
        kld_avg = sum(klds)/total
        print("epoch:", e)
        print("loss: %.2E" % loss_avg)
        print("loss actor: %.2E" % loss_actor_avg)
        print("loss critic: %.2E" % loss_critic_avg)
        print("loss entropy: %.2E" % loss_entropy_avg)
        print("kl divergence: %.2E" % kld_avg)
        print("expected return: %.2E" % expected_return_avg)
        # apply grads for each epoch
        optimizer.apply_gradients(zip(grads_acc, model.trainable_variables))
        # early stopping
        if kld_avg > 0.02:
            print("Early stopping at epoch", e, "due to large KL divergence.", flush=True)
            break
    # save model
    model.save(p_model)
    print("model saved", flush=True)

    ### NEW: Save Optimizer Separately ###
    ckpt_manager.save()
    print("optimizer saved", flush=True)

if __name__ == "__main__":
    import time
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    from model import ActorCritic
    # path
    p_data = "data/exploiter.tfrecord"
    p_model = "model/exploiter.keras"
    p_optimizer = "optimizer/exploiter"
    # Faster with CPU rather than GPU
    tf.config.set_visible_devices([], 'GPU')
    # Start training
    t1 = time.time()
    train(p_data, p_model, p_optimizer, epoch=10, learning_rate=1e-4, batch_size=4096)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
