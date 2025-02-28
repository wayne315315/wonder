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
    def send_face(self, civ):
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
        print("state: ")
        for item in state:
            built = item.pop("built")   
            print("")
            print(item)
            print("built: ", built)
            item["built"] = built
        print("")
        print("record: ", [args for *args, _ in record])
        print("")
        print("hand: ", hand)
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
        print("state: ")
        for item in state:
            built = item.pop("built")   
            print("")
            print(item)
            print("built: ", built)
            item["built"] = built
        print("")
        print("record: ", [args for *args, _ in record])
        print("")
        print("trade coins combination: ")
        for i, coin in enumerate(coins):
            print(f"{i}: {coin}")
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
