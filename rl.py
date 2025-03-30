import random

import numpy as np
import tensorflow as tf

from game import Game
from player import Player, RandomPlayer
from const import Action, CARDS, CIVS


class AIPlayer(RandomPlayer):
    def __init__(self, model):
        super().__init__()
        self.civs = [None] + list(CIVS)
        self.faces = [None] + ["Day", "Night"]
        self.cards = [None] + list(CARDS)
        self.action = [None] + list(Action)
        self.civ2idx = {civ: i for i, civ in enumerate(self.civs)}
        self.face2idx = {face: i for i, face in enumerate(self.faces)}
        self.card2idx = {card: i for i, card in enumerate(self.cards)}
        self.action2idx = {action: i for i, action in enumerate(self.action)}
        self.model = model

    def s2v(self, state, record):
        # determine the number of players
        n = len(state)
        # (turn, card, action, pos, civ, face)
        v = []
        # priming
        for pos, s in enumerate(state):
            i_civ = self.civ2idx[s["civ"]]
            i_face = self.face2idx[s["face"]]
            v.append([0, 0, 0, pos, i_civ, i_face])

        # other players
        for pos, s in enumerate(state):
            i_civ = self.civ2idx[s["civ"]]
            i_face = self.face2idx[s["face"]]
            if pos:
                for turn, card in s["built"]:
                    i_card = self.card2idx[card]
                    i_action = self.action2idx[Action.BUILD]
                    v.append([turn, i_card, i_action, pos, i_civ, i_face])
                for turn in s["wonder"]:
                    i_card = self.card2idx[None] # 0
                    i_action = self.action2idx[Action.WONDER]
                    v.append([turn, i_card, i_action, pos, i_civ, i_face])
                for turn in s["discard"]:
                    i_card = self.card2idx[None]
                    i_action = self.action2idx[Action.DISCARD]
                    v.append([turn, i_card, i_action, pos, i_civ, i_face])
            else:
                i_civ_player = self.civ2idx[s["civ"]]
                i_face_player = self.face2idx[s["face"]]
        # player itself
        for turn, card, action, hand in record:
            i_card = self.card2idx[card]
            i_action = self.action2idx[action]
            v.append([turn, i_card, i_action, 0, i_civ_player, i_face_player])
        
        # padding
        # (turn, card, action, pos, civ, face)
        m = 18 * n + 6 - len(v) # max total : Prime turn n + (18-1) turn * n player + 3 extra turn (Babylon) + 3 extra turn (Halikarnassos)
        for _ in range(m):
            v.append([19, 0, 0, n, 0, 0])

        v = np.asarray(v)
        return v


    def h2v(self, hand):
        h = np.asarray([self.card2idx[card] for card in hand])
        return h

    def c2v(self, coins):
        pass

    def send_face(self, state):
        return random.choice(["Day", "Night"])

    def send_move(self, state, record, hand, asked):
        if not asked:
            v = self.s2v(state, record) # shape (18 * n + 6, 6)
            h = self.h2v(hand)
            v = tf.expand_dims(v, axis=0)
            h = tf.expand_dims(h, axis=0)
            policy, value = self.model(v, h)
            # gumbal max trick to sample from policy distribution without replacement
            noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(policy))))
            logits = policy + noise
            self.buffer = tf.argsort(logits, axis=-1, direction='ASCENDING').numpy()[0].tolist()

        output = self.buffer.pop()
        i_card, i_action = divmod(output, len(Action))
        card = self.cards[i_card + 1]
        action = self.action[i_action + 1]
        return (card, action)

    def send_trade(self, state, record, coins):
        return coins[0]

    def recv_score(self, scores):
        pass

if __name__ == "__main__":
    from model import ActorCritic
    model = ActorCritic(len(CARDS), 100)
    n = 7
    game = Game(n)
    players = [RandomPlayer() for _ in range(n)]
    players[0] = AIPlayer(model)
    for i in range(n):
        game.register(i, players[i])

    game.run()
    model.summary()
