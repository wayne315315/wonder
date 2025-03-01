from abc import ABC, abstractmethod
import random
import itertools

from const import Action

class Player(ABC):

    def __init__(self):
        self.buffer = None

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
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
    def send_face(self, civ):
        return random.choice(["Day", "Night"])

    def send_move(self, state, record, hand, asked):
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
        pick, action = self.buffer.pop()
        return pick, action

    def send_trade(self, state, record, coins):
        if self.verbose:
            self.show_state(state)
            self.show_record(record)
            self.show_coins(coins)
        return random.choice(coins)

    def recv_score(self, scores):
        pass

class HumanPlayer(Player):
    def send_face(self, civ):
        print("civ: ", civ) 
        print("Day or Night?")
        while True:
            choice = input("type \"D\" for Day, \"N\" for Night : ").strip()
            if choice == "D":
                return "Day"
            elif choice == "N":
                return "Night"
            else:
                print("Invalid input. Please type \"D\" or \"N\" ")
    
    def send_move(self, state, record, hand, asked):
        self.show_state(state)
        self.show_record(record)
        self.show_hand(hand)
        if not asked:
            self.buffer = set()
        else:
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

    def send_trade(self, state, record, coins):
        self.show_state(state)
        self.show_record(record)
        self.show_coins(coins)
        while True:
            i = input("Choose a trade combination (type the number) : ").strip()
            try:
                coin = coins[int(i)]
            except:
                print("** ERROR: Invalid input. Please try again")
            else:
                break
        return coin

    def recv_score(self, scores):
        pass
