from pathlib import Path
import random
import itertools

import tensorflow as tf

from helper import Adaptor, CARDS, Action
from data import data_gen


def create_model(num_feature=100):
    from model import ActorCritic
    from game import CARDS
    model = ActorCritic(len(CARDS), num_feature)
    return model

def translate(episode, gamma=0.9, penalty=-1.0):
    rec_valid, rec_invalid = episode
    # Task 1 : translate rewards to expected return with discounted factor gamma
    n = len(rec_valid)
    for i, rec in enumerate(rec_valid):
        rec[-1] *= gamma ** (n - i)# rec[-1] : reward -> expected return (discounted w/ gamma)
        api, args, res, reward = rec
    # Task 2 : remove all record which api call is not 'move'
    rec_valid = [rec for rec in rec_valid if rec[0] == "move"]
    # Task 3 : generate illegal examples from rec_valid (pick card not in hand)
    num_per_state = 1 if rec_invalid else 2
    rec_illegal = []
    for _ in range(num_per_state):
        for api, args, _, _ in rec_valid:
            _, _, hand = args
            others = set(CARDS) - set(hand)
            pick, action = random.choice(list(itertools.product(others, Action)))
            rec_illegal.append([api, args, [pick, action], penalty]) # illegal move
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
    recs = rec_valid + rec_invalid + rec_illegal
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
    rs = tf.constant(rs, dtype=tf.float64)
    return vs, hs, ys, rs
        
def compute_loss(logits, values, ys, rs):
    # inputs : v, h, y_true -> outputs : p(a|s), v(s)
    # inputs : rewards -> outputs: g(a,s)
    # loss_actor = -log(p(a|s)) * (g(a,s) - v(s))
    # loss_critic = huber_loss(g(a,s), v(s))
    # loss = loss_actor + loss_critic
    pass

def train(model, optimizer, num_play, num_game, gamma=0.9, penalty=-1.0):
    input_signature = (
        tf.TensorSpec(shape=[None, None, 6], dtype=tf.int32), # vs
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), # hs
        tf.TensorSpec(shape=[None], dtype=tf.int32), # ys
        tf.TensorSpec(shape=[None], dtype=tf.float64) # rs
    )
    @tf.function(input_signature=input_signature)
    def train_step(vs, hs, ys, rs):
        #tf.print(vs.shape, hs.shape, ys.shape, rs.shape)
        policy, value = model(vs, hs)
        tf.print(policy.shape, value.shape)
    # training loop
    for episode in data_gen(num_play, num_game):
        vs, hs, ys, rs = translate(episode, gamma=gamma, penalty=penalty)
        train_step(vs, hs, ys, rs)


if __name__ == "__main__":
    num_play = 2 # The number of rehearsals for each game
    num_game = 2 # The number of games for each number of the total players
    # model
    model_path = Path("model", "ac.keras")
    model = tf.keras.models.load_model(model_path)
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train(model, optimizer, num_play, num_game)