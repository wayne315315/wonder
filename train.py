from pathlib import Path
import random
from collections import defaultdict

from tqdm import tqdm
import tensorflow as tf

from helper import Adaptor
from data import data_gen
from model import ActorCritic
from serving import launch_server, kill_server
from serving import export_archive, clean_archive
from serving import probe


def create_model():
    from game import CARDS
    model = ActorCritic(len(CARDS))
    return model

def translate(episode, gamma=0.9, penalty=-1.0):
    rec_valid, rec_invalid = episode
    # Task 1 : translate rewards to expected return with discounted factor gamma
    n = len(rec_valid)
    for i, rec in enumerate(rec_valid):
        rec[-1] *= gamma ** (n - i) # rec[-1] : reward -> expected return (discounted w/ gamma)
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
    # stack all vectors
    for key in vs:
        for item in [vs, hs, ys]:
            item[key] = tf.constant(item[key], dtype=tf.int32)
        rs[key] = tf.constant(rs[key], dtype=tf.float32)
    return vs, hs, ys, rs

def compute_loss(logits, values, actions, rewards):
    # inputs : v, h, y_true -> outputs : p(a|s), v(s)
    # inputs : rewards -> outputs: g(a,s)
    # loss_actor = -log(p(a|s)) * (g(a,s) - v(s))
    # loss_critic = huber_loss(g(a,s), v(s))
    # loss = loss_actor + loss_critic
    # tf.gather(x, y, batch_dims=1)
    # TensorShape([None, 231]) TensorShape([None, 1]) TensorShape([None]) TensorShape([None])
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    values = values[:,0] # v(s) TensorShape([None])
    loss_critic = huber_loss(rewards, values) # TensorShape([])
    advantages = rewards - values # (g(a,s) - v(s))  TensorShape([None])
    logsoftmax = tf.nn.log_softmax(logits, axis=1) # log(p(ai|s), ...) TensorShape([None, 231])
    logsoftmax = tf.gather(logsoftmax, actions, batch_dims=1) # log(p(a|s)) TensorShape([None])
    loss_actor = -tf.reduce_sum(logsoftmax * advantages)
    #loss = loss_critic + loss_actor
    loss = -tf.reduce_sum(logsoftmax * rewards)
    # metrics
    probs = tf.nn.softmax(logits, axis=1) # p(a|s) TensorShape([None, 231])
    probs = tf.gather(probs, actions, batch_dims=1) # p(a|s) TensorShape([None])
    prob = tf.reduce_sum(probs)
    expected_return = tf.reduce_sum(probs * rewards) # E[R|a,s] TensorShape([])
    return loss, loss_actor, loss_critic, prob, expected_return


def train(model_path, serve_name, epoch, num_play, num_game, gamma=0.99, penalty=-1.0, run_episode=True, batch_size=512):
    # model
    model = tf.keras.models.load_model(model_path) if model_path.exists() else create_model()
    # dry run model in case model hasn't been built
    if not model.built:
        model.build(1)
    # export base model with serve_name
    export_archive(serve_name, model, 0)
    # launch server
    launch_server()
    # probe server until it is ready
    while not probe(serve_name, serve_version=None, verbose=False):
        pass
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # train_step
    input_signature = (
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32), # vs
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # hs
        tf.TensorSpec(shape=[None], dtype=tf.int32), # ys
        tf.TensorSpec(shape=[None], dtype=tf.float32) # rs
    )
    grads_acc = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32), trainable=False) for tv in model.trainable_variables]
    metrices_acc = [tf.Variable(0.0, trainable=False) for _ in range(5)]
    @tf.function(input_signature=input_signature)
    def train_step(vs, hs, ys, rs):
        # reset metrices_acc
        for metric_acc in metrices_acc:
            metric_acc.assign(0.0)
        # split large tensor into batches
        dataset = tf.data.Dataset.from_tensor_slices((vs, hs, ys, rs)).batch(batch_size)
        # iterate over batches
        for vs_, hs_, ys_, rs_ in dataset:
            # compute metrices
            with tf.GradientTape() as tape:
                logits, values = model(vs_, hs_, training=True)
                metrices = compute_loss(logits, values, ys_, rs_)
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
            tf.print("metrices_acc:", metrices_acc)
        return tuple(metrices_acc)
    # training loop
    for e in range(epoch):
        # reset grads_acc
        for grad_acc in grads_acc:
            grad_acc.assign(tf.zeros_like(grad_acc))
        # run episodes
        losses = []
        losses_actor = []
        losses_critic = []
        probs = []
        expected_returns = []

        data_iterator = data_gen(num_play, num_game, serve_name=serve_name) if run_episode else data_gen(num_play, num_game)
        #data_iterator = data_gen(num_play, num_game, model=model) if run_episode else data_gen(num_play, num_game)
        
        vs = defaultdict(list)
        hs = defaultdict(list)
        ys = defaultdict(list)
        rs = defaultdict(list)
        total = 15 * num_play * num_game # data only from player 0, the best & the worst player
        for episode in tqdm(data_iterator, total=total):
            vs_, hs_, ys_, rs_ = translate(episode, gamma=gamma, penalty=penalty)
            for key in vs_:
                vs[key].append(vs_[key])
                hs[key].append(hs_[key])
                ys[key].append(ys_[key])
                rs[key].append(rs_[key])
        # compute gradients & metrices
        for key in vs:
            v, h, y, r = [tf.concat(x, axis=0) for x in [vs[key], hs[key], ys[key], rs[key]]]
            tf.print("")
            tf.print("key:", key)
            tf.print("v.shape:", v.shape)
            tf.print("h.shape:", h.shape)
            tf.print("y.shape:", y.shape)
            tf.print("r.shape:", r.shape)
            loss, loss_actor, loss_critic, prob, expected_return  = train_step(v,h,y,r)
            losses.append(loss.numpy())
            losses_actor.append(loss_actor.numpy())
            losses_critic.append(loss_critic.numpy())
            probs.append(prob.numpy())
            expected_returns.append(expected_return.numpy())
        # compute the mean for all matrices in this epoch
        loss_avg = sum(losses)/total
        loss_actor_avg = sum(losses_actor)/total
        loss_critic_avg = sum(losses_critic)/total
        prob_avg = sum(probs)/total
        expected_return_avg = sum(expected_returns)/total
        print("epoch:", e)
        print("loss: %.2E" % loss_avg)
        print("loss actor: %.2E" % loss_actor_avg)
        print("loss critic: %.2E" % loss_critic_avg)
        print("prob: %.2E" % prob_avg)
        print("expected return: %.2E" % expected_return_avg)
        # apply grads for each epoch
        optimizer.apply_gradients(zip(grads_acc, model.trainable_variables))
        # save model
        model.save(model_path)
        print("model saved")
        # export the updated model with version e
        export_archive(serve_name, model, e)

if __name__ == "__main__":
    epoch = 1000
    num_play = 10 # The number of rehearsals for each game
    num_game = 10 # The number of games for each number of the total players
    # model
    model_dir = Path("model")
    if not model_dir.exists():
        model_dir.mkdir()
    model_path = Path(model_dir, "ac.keras")
    serve_name = "ac"
    try:
        train(model_path, serve_name, epoch, num_play, num_game, run_episode=True)
    except Exception as e:
        raise e
    finally:
        # clean archive
        clean_archive(serve_name)
        # kill server
        kill_server()