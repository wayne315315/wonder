import numpy as np

from const import Action, CARDS, CIVS

class Adaptor:
    def __init__(self):
        super().__init__()
        self.civs = [None] + sorted(list(CIVS))
        self.faces = [None] + ["Day", "Night"]
        self.cards = [None] + sorted(list(CARDS))
        self.action = [None] + [Action(i) for i in range(1,4)]
        self.pair = [(self.cards[1 + (i // len(Action))], self.action[1 + (i % len(Action))]) for i in range(len(CARDS) * 3)] # (pick, action)
        self.civ2idx = {civ: i for i, civ in enumerate(self.civs)}
        self.face2idx = {face: i for i, face in enumerate(self.faces)}
        self.card2idx = {card: i for i, card in enumerate(self.cards)}
        self.action2idx = {action: i for i, action in enumerate(self.action)}
        self.pair2idx = {pair: i for i, pair in enumerate(self.pair)}

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
        v = np.expand_dims(v, axis=0)
        return v
    
    def h2v(self, hand):
        h = np.asarray([self.card2idx[card] for card in hand])
        h = np.expand_dims(h, axis=0)
        return h
    
    def c2v(self, coins):
        pass
