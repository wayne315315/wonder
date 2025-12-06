from email import policy
import random
from collections import defaultdict
from threading import Thread, Lock
from queue import Queue
from concurrent.futures import Future

import numpy as np

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


class AIPlayer2(Player):
    fns = {}
    inputs = defaultdict(Queue) 
    timeout = 1e-1
    batch_size = 100
    shapes = [(v_dim, h_dim) for v_dim in [19 * n + 6 for n in range(3,8)] for h_dim in [7, 21]]
    threads = {}
    locks = defaultdict(Lock)
    def __init__(self, name, fn):
        Player.__init__(self)
        self.helper = Adaptor()
        self.name = name
        AIPlayer2.fns[name] = fn
        for v_dim, h_dim in AIPlayer2.shapes:
            key = (name, v_dim, h_dim)
            if key not in AIPlayer2.threads:
                t = Thread(target=AIPlayer2.loop, args=(key,), daemon=True)
                t.start()
                AIPlayer2.threads[key] = t

    def _send_face(self, state):
        v = self.helper.s2v(state, [])
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        return face

    def _send_move(self, state, record, hand, asked):
        if not asked:
            v = self.helper.s2v(state, record) # shape (19 * n + 6, 7)
            h = self.helper.h2v(hand)
            future = Future()
            AIPlayer2.inputs[(self.name, v.shape[1], h.shape[1])].put((v, h, future))
            _, moves = future.result()
            self.buffer = moves.tolist()
        else:
            self.record[-1][-1] = False # last move is invalid
        output = self.buffer.pop()
        pick, action = self.helper.pair[output]
        return (pick, action)

    def _send_trade(self, state, record, coins):
        trade = coins[0]
        return trade

    @classmethod
    def loop(cls, key):
        name, v_dim, h_dim = key
        fn = cls.fns[name]
        lock = cls.locks[name]
        timeout = cls.timeout if h_dim == 7 else 1e-3
        while True:
            batch_v = []
            batch_h = []
            futures = []
            try:
                while len(batch_v) < cls.batch_size:
                    v, h, future = cls.inputs[key].get(timeout=timeout)
                    batch_v.append(v)
                    batch_h.append(h)
                    futures.append(future)
            except:
                pass
            if batch_v:
                batch_v = np.vstack(batch_v)
                batch_h = np.vstack(batch_h)
                with lock:
                    _, batch_moves = fn(batch_v, batch_h)
                    batch_moves = batch_moves.numpy()
                for future, moves in zip(futures, batch_moves):
                    future.set_result((None, moves))
