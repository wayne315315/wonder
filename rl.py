import random

from helper import Adaptor
from player import Player


class AIPlayer(Player):
    def __init__(self, fn):
        Player.__init__(self)
        self.helper = Adaptor()
        self.fn = fn

    def _send_face(self, state):
        v = self.helper.s2v(state, [])
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        return face

    def _send_move(self, state, record, hand, asked):
        if not asked:
            v = self.helper.s2v(state, record) # shape (19 * n + 6, 7)
            h = self.helper.h2v(hand)
            _, moves = self.fn(v, h)
            self.buffer = moves.numpy()[0].tolist()
        else:
            self.record[-1][-1] = False # last move is invalid
        output = self.buffer.pop()
        pick, action = self.helper.pair[output]
        return (pick, action)

    def _send_trade(self, state, record, coins):
        trade = coins[0]
        return trade
