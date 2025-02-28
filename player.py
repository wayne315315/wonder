from abc import ABC, abstractmethod
import random
import itertools

from const import Action

class Player(ABC):

    def __init__(self):
        self.buffer = None

    @abstractmethod
    def send_face(self, state):
        pass
    
    @abstractmethod
    def send_move(self, state, record, hand, asked):
        pass

    @abstractmethod
    def send_trade(self, state, record, coins):
        pass

    @abstractmethod
    def recv_score(self, scores):
        pass

class RandomPlayer(Player):
    def send_face(self, state):
        return random.choice(["Day", "Night"])

    def send_move(self, state, record, hand, asked):
        if not asked:
            x = list(itertools.product(hand, [Action.DISCARD]))
            y = list(itertools.product(hand, [Action.BUILD, Action.WONDER]))
            random.shuffle(x)
            random.shuffle(y)
            self.buffer = x + y
        pick, action = self.buffer.pop()
        return pick, action

    def send_trade(self, state, record, coins):
        return random.choice(coins)

    def recv_score(self, scores):
        pass