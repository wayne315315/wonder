from abc import ABC, abstractmethod
from threading import Event
from pathlib import Path

import requests
import tensorflow as tf

from player import Player, RandomPlayer
from rl import AIPlayer
from const import Action
from model import ActorCritic


class WebPlayer(Player):
    def __init__(self, uid, url, events):
        Player.__init__(self)
        self.uid = uid
        self.url = url
        self.events = events # List[Event] only 1 event or 0
    def enqueue(self, task):
        endpoint = f"{self.url}/enqueue"
        while True:
            # send POST request with cookie to Flask server
            resp = requests.post(endpoint, json=task, cookies={"uid": self.uid})
            if resp.status_code == 200:
                break

    def execute(self):
        endpoint = f"{self.url}/execute"
        while True:
            resp = requests.get(endpoint, cookies={"uid": self.uid})
            if resp.status_code == 200:
                res = resp.json()
                break
        return res

    @abstractmethod
    def _send_face(self, state):
        pass
    
    @abstractmethod
    def _send_move(self, state, record, hand, asked): # pick, action
        pass

    @abstractmethod
    def _send_trade(self, state, record, coins): # trade: (left_coin, right_coin)
        pass

    def recv_notice(self, notice):
        self.enqueue(notice)

    def loop(self):
        pass


class WebHumanPlayer(WebPlayer):
    def _send_face(self, state):
        task = {
            "type": "FACE"
        }
        # ensure self.events is empty before appending
        e = Event()
        self.events.clear()
        self.events.append(e)
        self.enqueue(task)
        e.wait()
        res = self.execute()
        face = res["face"]
        return face
    
    def _send_move(self, state, record, hand, asked): # pick, action
        task = {
            "type": "MOVE",
            "hand": hand,
            "asked": asked
        }
        # ensure self.events is empty before appending
        e = Event()
        self.events.clear()
        self.events.append(e)
        self.enqueue(task)
        e.wait()
        res = self.execute()
        pick = res["pick"]
        action = res["action"]
        a2a = {"BUILD": Action.BUILD, "WONDER": Action.WONDER, "DISCARD": Action.DISCARD}
        action = a2a[action]
        return pick, action

    def _send_trade(self, state, record, coins): # trade: (left_coin, right_coin)
        task = {
            "type": "TRADE",
            "coins": coins
        }
        # ensure self.events is empty before appending
        e = Event()
        self.events.clear()
        self.events.append(e)
        self.enqueue(task)
        e.wait()
        res = self.execute()
        trade = tuple(res["trade"])
        return trade


class WebRandomPlayer(WebPlayer, RandomPlayer):
    def __init__(self, uid, url, events, verbose=False):
        super().__init__(uid, url, events)
        RandomPlayer.__init__(self, verbose=verbose)

    def _send_face(self, state):
        return RandomPlayer._send_face(self, state)
    
    def _send_move(self, state, record, hand, asked):
        return RandomPlayer._send_move(self, state, record, hand, asked)

    def _send_trade(self, state, record, coins): 
        return RandomPlayer._send_trade(self, state, record, coins)
    
    def loop(self):
        while True:
            endpoint = f"{self.url}/dequeue"
            resp = requests.get(endpoint, cookies={"uid": self.uid})
            task = resp.json()
            if task["type"] == "SCORE":
                print("Thread finished")
                break

class WebAIPlayer(AIPlayer, WebPlayer):
    def __init__(self, uid, url, events):
        model_path = Path("model", "base.keras")
        model = tf.keras.models.load_model(model_path)
        AIPlayer.__init__(self, model)
        WebPlayer.__init__(self, uid, url, events)
    
    def loop(self):
        while True:
            endpoint = f"{self.url}/dequeue"
            resp = requests.get(endpoint, cookies={"uid": self.uid})
            task = resp.json()
            if task["type"] == "SCORE":
                print("Thread finished")
                break
