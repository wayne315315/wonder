import random

import numpy as np

from player import Player
from const import Action, CARDS, CIVS


class AIPlayer(Player):
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
        m = 19 * n + 6 - len(v) # max total : Prime turn n + 18 turn * n player + 3 extra turn (Babylon) + 3 extra turn (Halikarnassos)
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
        v = self.s2v(state, [])
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        self.record.append(["face", [state], [face], True])
        return face

    def send_move(self, state, record, hand, asked):
        if not asked:
            v = self.s2v(state, record) # shape (18 * n + 6, 6)
            h = self.h2v(hand)
            v = np.expand_dims(v, axis=0)
            h = np.expand_dims(h, axis=0)
            self.buffer = self.model.predict_move(v, h)[0].tolist()
        else:
            self.record[-1][-1] = False # last move is invalid

        output = self.buffer.pop()
        i_card, i_action = divmod(output, len(Action))
        pick = self.cards[i_card + 1]
        action = self.action[i_action + 1]
        self.record.append(["move", [state, record, hand], [pick, action], True])
        return (pick, action)

    def send_trade(self, state, record, coins):
        trade = coins[0]
        self.record.append(["trade", [state, record, coins], [trade], True])
        return trade

    def recv_score(self, scores):
        self.record.append(["score", [scores], [], True])

if __name__ == "__main__":
    import tensorflow as tf
    from player import RandomPlayer
    from model import ActorCritic
    from game import Game
    model = ActorCritic(len(CARDS), 100)
    n = 3
    game = Game(n)
    players = [RandomPlayer() for _ in range(n)]
    players[0] = AIPlayer(model)
    for i in range(n):
        game.register(i, players[i])

    game.run()
    model.summary()
    model.save("ac.keras")
    ####
    print("load model")
    new_model = tf.keras.models.load_model('ac.keras')
    game = Game(n, random_face=False)
    players = [RandomPlayer() for _ in range(n)]
    players[0] = AIPlayer(new_model)
    for i in range(n):
        game.register(i, players[i])

    game.run()
    model.summary()
