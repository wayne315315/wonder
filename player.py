from abc import ABC, abstractmethod
import random
import itertools

from const import Action

class Player(ABC):

    def __init__(self):
        self.buffer = None
        self.record = [] # [[API_CALL, *args, *res, is_valid]]

    def show_state(self, state):
        print("")
        print("STATE")
        for item in state:
            built = item.pop("built")
            color = item.pop("color")
            coin = item.pop("coin")
            rsc = item.pop("rsc")
            rsc_tradable = item.pop("rsc_tradable")
            shield = item.pop("shield")
            symbol = item.pop("symbol")
            print("")
            print(item)
            print("built: ", built)
            print("--------------------")
            print("color: ", color)
            print("coin: ", coin)
            print("rsc: ", dict(rsc))
            print("rsc_tradable: ", dict(rsc_tradable))
            print("shield: ", shield)
            print("symbol: ", dict(symbol))
            item["built"] = built
            item["color"] = color
            item["coin"] = coin
            item["rsc"] = rsc
            item["rsc_tradable"] = rsc_tradable
            item["shield"] = shield
            item["symbol"] = symbol

    def show_record(self, record):
        print("")
        print("RECORD: ", [args for *args, _ in record])

    def show_hand(self, hand):
        print("")
        print("HAND: ", hand)

    def show_coins(self, coins):
        print("")
        print("Trade coins combination: ")
        for i, coin in enumerate(coins):
            print(f"{i}: {coin}")  

    @abstractmethod
    def _send_face(self, state):
        pass
    
    @abstractmethod
    def _send_move(self, state, record, hand, asked):
        pass

    @abstractmethod
    def _send_trade(self, state, record, coins):
        pass

    def send_face(self, state):
        face = self._send_face(state)
        self.record.append(["face", [state], [face], True])
        return face
    
    def send_move(self, state, record, hand, asked):
        pick, action = self._send_move(state, record, hand, asked)
        self.record.append(["move", [state, record, hand], [pick, action], True])
        return pick, action
    
    def send_trade(self, state, record, coins):
        trade = self._send_trade(state, record, coins)
        self.record.append(["trade", [state, record, coins], [trade], True])
        return trade
    
    def recv_score(self, scores):
        self.record.append(["score", [scores], [], True])

    def recv_notice(self, notice):
        pass


class RandomPlayer(Player):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def _send_face(self, state):
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        return face

    def _send_move(self, state, record, hand, asked):
        if self.verbose:
            self.show_state(state)
            self.show_record(record)
            self.show_hand(hand)
        if not asked:
            x = list(itertools.product(hand, [Action.DISCARD]))
            y = list(itertools.product(hand, [Action.BUILD, Action.WONDER]))
            random.shuffle(x)
            random.shuffle(y)
            self.buffer = x + y
        else:
            self.record[-1][-1] = False # last move is invalid
        pick, action = self.buffer.pop()
        return pick, action

    def _send_trade(self, state, record, coins):
        if self.verbose:
            self.show_state(state)
            self.show_record(record)
            self.show_coins(coins)
        trade = random.choice(coins)
        return trade


class HumanPlayer(Player):
    def _send_face(self, state):
        civs = [item["civ"] for item in state]
        print("All civs:", civs)
        print("Your civ:", civs[0]) 
        print("Day or Night?")
        while True:
            choice = input("type \"D\" for Day, \"N\" for Night : ").strip()
            if choice == "D":
                face = "Day"
                break
            elif choice == "N":
                face = "Night"
                break
            else:
                print("Invalid input. Please type \"D\" or \"N\" ")
        return face
    
    def _send_move(self, state, record, hand, asked):
        self.show_state(state)
        self.show_record(record)
        self.show_hand(hand)
        if not asked:
            self.buffer = set()
        else:
            self.record[-1][-1] = False # last move is invalid
            print("")
            print("** Previous move is invalid. Retry...")
        while True:
            pick = input("Choose a card from the hand (type the name of the card) : ").strip()
            action = input("Choose an action from one of this (type B for BUILD; W for WONDER; D for DISCARD) : ").strip()
            if pick not in hand or action not in ["B", "W", "D"]:
                print("** ERROR: Invalid input. Please try again")
            elif (pick, action) in self.buffer:
                print("** ERROR: You have already made this move and failed. Please try again")
            else:
                self.buffer.add((pick, action))
                break
        a2a = {"B": Action.BUILD, "W": Action.WONDER, "D": Action.DISCARD}
        action = a2a[action]
        return pick, action

    def _send_trade(self, state, record, coins):
        self.show_state(state)
        self.show_record(record)
        self.show_coins(coins)
        while True:
            i = input("Choose a trade combination (type the number) : ").strip()
            try:
                trade = coins[int(i)]
            except:
                print("** ERROR: Invalid input. Please try again")
            else:
                break
        return trade
    
    def recv_notice(self, notice):
        pass