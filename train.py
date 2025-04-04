from pathlib import Path
import random
import itertools

from tqdm import tqdm
import tensorflow as tf

from helper import Adaptor, CARDS, Action
from data import data_gen


def create_model():
    from model import ActorCritic
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
    # Task 3 : generate illegal examples from rec_valid (pick card not in hand)
    """
    num_per_state = 1 if rec_invalid else 2
    rec_illegal = []
    for _ in range(num_per_state):
        for api, args, _, _ in rec_valid:
            _, _, hand = args
            others = set(CARDS) - set(hand)
            pick, action = random.choice(list(itertools.product(others, Action)))
            rec_illegal.append([api, args, [pick, action], penalty]) # illegal move"
    """
    # Task 4 : add penalty to rec_invalid & sample n examples from rec_invalid
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
    # Task 5 : Collect all records (valid : invalid : illegal = 1 : 1 : 1)
    #recs = rec_valid + rec_invalid + rec_illegal
    recs = rec_valid + rec_invalid
    # Task 6 : convert state, record, hand to v, h; pick, action to y
    adaptor = Adaptor()
    vs = []
    hs = []
    ys = []
    rs = []
    for rec in recs:
        api, (state, record, hand), (pick, action), reward = rec
        v = adaptor.s2v(state, record)[0]
        h = adaptor.h2v(hand)[0]
        y = adaptor.pair2idx[(pick, action)]
        vs.append(v)
        hs.append(h)
        ys.append(y)
        rs.append(reward)
    # stack all vectors : vs, hs, ys
    vs = tf.constant(vs, dtype=tf.int32)
    hs = tf.ragged.constant(hs, dtype=tf.int32)
    hs = hs.to_tensor()
    ys = tf.constant(ys, dtype=tf.int32)
    rs = tf.constant(rs, dtype=tf.float32)
    return vs, hs, ys, rs
        
def compute_loss(logits, values, actions, rewards):
    # inputs : v, h, y_true -> outputs : p(a|s), v(s)
    # inputs : rewards -> outputs: g(a,s)
    # loss_actor = -log(p(a|s)) * (g(a,s) - v(s))
    # loss_critic = huber_loss(g(a,s), v(s))
    # loss = loss_actor + loss_critic
    # tf.gather(x, y, batch_dims=1)
    # TensorShape([None, 231]) TensorShape([None, 1]) TensorShape([None]) TensorShape([None])
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    values = values[:,0] # v(s) TensorShape([None])
    loss_critic = huber_loss(rewards, values) # TensorShape([])
    advantages = rewards - values # (g(a,s) - v(s))  TensorShape([None])
    logsoftmax = tf.nn.log_softmax(logits, axis=1) # log(p(ai|s), ...) TensorShape([None, 231])
    logsoftmax = tf.gather(logsoftmax, actions, batch_dims=1) # log(p(a|s)) TensorShape([None])
    loss_actor = -tf.reduce_mean(logsoftmax * advantages)
    loss = loss_critic + loss_actor
    return loss, loss_actor, loss_critic

def train(model_path, epoch, num_play, num_game, gamma=0.99, penalty=-10.0, run_episode=True):
    # model
    model = tf.keras.models.load_model(model_path) if model_path.exists() else create_model()
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # train_step
    input_signature = (
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32), # vs
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # hs
        tf.TensorSpec(shape=[None], dtype=tf.int32), # ys
        tf.TensorSpec(shape=[None], dtype=tf.float32) # rs
    )
    @tf.function(input_signature=input_signature)
    def train_step(vs, hs, ys, rs):
        with tf.GradientTape() as tape:
            logits, values = model(vs, hs)
            loss, loss_actor, loss_critic = compute_loss(logits, values, ys, rs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_actor, loss_critic
    # training loop
    for e in range(epoch):
        losses = []
        losses_actor = []
        losses_critic = []
        data_iterator = data_gen(num_play, num_game, model=model) if run_episode else data_gen(num_play, num_game)
        for episode in tqdm(data_iterator, total=25*num_play*num_game):
            vs, hs, ys, rs = translate(episode, gamma=gamma, penalty=penalty)
            loss, loss_actor, loss_critic = train_step(vs, hs, ys, rs)
            losses.append(loss.numpy())
            losses_actor.append(loss_actor.numpy())
            losses_critic.append(loss_critic.numpy())
        # save model
        loss_avg = sum(losses)/len(losses)
        loss_actor_avg = sum(losses_actor)/len(losses_actor)
        loss_critic_avg = sum(losses_critic)/len(losses_critic)
        print("epoch:", e)
        print("loss: %.2E" % loss_avg)
        print("loss actor: %.2E" % loss_actor_avg)
        print("loss critic: %.2E" % loss_critic_avg)
        model.save(model_path)
        print("model saved")

if __name__ == "__main__":
    epoch = 1000
    num_play = 8 # The number of rehearsals for each game
    num_game = 8 # The number of games for each number of the total players
    # model
    model_dir = Path("model")
    if not model_dir.exists():
        model_dir.mkdir()
    model_path = Path(model_dir, "ac.keras")
    train(model_path, epoch, num_play, num_game)
