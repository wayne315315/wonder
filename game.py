# brown
# - cost : coin
# - func : rsc
# grey
# - cost : coin
# - func : rsc
# blue
# - cost : rsc
# - func : score
# red
# - cost : rsc
# - func : shield
# green
# - cost : rsc
# - func : symbol
# yellow
# - cost : rsc
# - func : coin, score, rsc, trade
# purple
# - cost : rsc
# - func : score, symbol, others
# wonder
# - cost : rsc
# - func : coin, score, rsc, shield, symbol, others
""" 
# notice format
SETTING
{
    "type": "SETTING",
    "faces": None,
    "civs": ["Olympia", "Babylon", "Gizah"]
}
AGE
{
    "type": "AGE",
    "age": 1
}
UPDATE
{
    "type": "UPDATE",
    "scavenge": False,
    "moves": [
        {"pick": "Pantheon", "action": "WONDER", "trade": [1, 2]},
        {"pick": "Arena", "action": "DISCARD", "trade": [0, 0]},
        {"pick": "Palace", "action": "BUILD", "trade": [3, 0]},
    ],
    "coins": [1,6,0]
}
CLEAR
{
    "type": "CLEAR"
}
BATTLE
{
    "type": "BATTLE",
    "battle": [[0,3],[-3,0],[0,0]]
}
SCORE
{
    "type": "SCORE",
    "civilian": [5, 12, 7],
    "conflict": [-6, 6, 18],
    "science": [10, 0, 0],
    "commerce": [0, 0, 0],
    "guild": [0, 0, 2],
    "wonder": [0, 0, 0],
    "wealth": [1, 2, 3],
    "total": [10, 20, 30],
    "coin": [4, 6, 10]
} 
"""
import logging
import sys
import random
import itertools
from copy import deepcopy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from const import CARDS, CIVS, DECK, Action, ALL_COLORS, ALL_RSC, RARE_RSC, BASIC_RSC, ALL_SYMBOL

class Game:
    def __init__(self, n, civs=None, faces=None, deck=None, random_face=True):
        # civs
        if not civs:
            civs = random.sample(list(CIVS), n)
        assert n == len(civs) == len(set(civs))
        
        # deck
        if not deck:
            deck = [[name for name, count in DECK[str(n)][str(age)].items() for _ in range(count)] for age in range(1, 4)]
            # add guild cards into the deck of Age III
            guilds = random.sample(list(name for name in CARDS if "Guild" in name), n + 2)
            deck[2].extend(guilds)
            for d in deck:
                random.shuffle(d)
        
        # assert
        if not random_face:
            try:
                assert faces is None
            except AssertionError:
                raise ValueError("If faces are given, random_face should be always set to True")

        # basics
        self.n = n
        self.random_face = random_face
        self.civs = civs
        self.faces = [random.choice(["Day", "Night"]) for _ in range(n)] if random_face and not faces else faces
        self.turn = 1
        self.discard = [] # for Halikarnassos
        self.hands = [[] for _ in range(n)]
        self.deck = deck
        self.players = [None] * n
        self.state = None
        self.record = [[] for _ in range(n)] # each player can only access its own record (turn, pick, action, hand)

        # extra
        self.rsc = [Counter({CIVS[civ]["rsc"]: 1}) for civ in self.civs]
        self.rsc_tradable = [Counter({CIVS[civ]["rsc"]: 1}) for civ in self.civs]
        self.built = [set() for _ in range(n)] # only card names with 7 colors, no wonder
        self.color = [{color: 0 for color in ALL_COLORS} for _ in range(n)] # keys: 7 colors + wonder
        self.symbol = [Counter() for _ in range(n)]
        self.shield = [0] * n
        
        # special
        self.free_first = None # only free for builings; not for wonders; only for Olympia
        self.free_last = None # only free for builings; not for wonders; only for Olympia
        self.free_color = None # only free for builings; not for wonders; only for Olympia
        self.seven = None # only for Babylon
        self.scavenge = None # only for Halikarnassos

        # score
        self.civilian = [0] * n # blue
        self.conflict = [0] * n # red
        self.science = [0] * n # green
        self.commerce = [0] * n # yellow
        self.guild = [0] * n # purple
        self.wonder = [0] * n # wonder
        self.wealth = [0] * n # coin / 3
        self.total = [0] * n

        # logger
        # debug, info, warning, error, critical
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

    def export(self):
        return self.n, self.civs, self.faces, self.deck, self.random_face

    def register(self, i, player):
        self.players[i] = player

    def unregister(self):
        self.players = [None] * self.n

    def rsc_compute(self, i, cost, own=True):
        rsc_seed = self.rsc[i] if own else self.rsc_tradable[i]
        rsc_gen = map(lambda x: x.split("|"), rsc_seed.elements())
        rsc_cnt = sorted({tuple(Counter(rscs)[rsc] for rsc in ALL_RSC) for rscs in itertools.product(*rsc_gen)})
        rsc_counters = [Counter(dict(zip(ALL_RSC, cnt))) for cnt in rsc_cnt]
        rsc_needed = []
        for rsc_counter_prev in sorted([cost - rsc_counter for rsc_counter in rsc_counters], key=lambda x: x.total()):
            flag = True
            for rsc_counter in rsc_needed:
                if rsc_counter_prev >= rsc_counter:
                    flag = False
            if flag:
                rsc_needed.append(rsc_counter_prev)
        return rsc_needed

    def rsc_trade(self, i, cost):
        # rsc_trade = [(Counter(left), Counter(right)), ...]
        l = (i - 1) % self.n
        r = (i + 1) % self.n
        rsc_needed = self.rsc_compute(i, cost)
        rsc_trade = []
        for needed in rsc_needed:
            for cnts in itertools.product(*map(lambda rsc: list(range(1+needed[rsc])), ALL_RSC)):
                left_needed = Counter(dict(zip(ALL_RSC, cnts))) - Counter()
                right_needed = needed - left_needed
                if not self.rsc_compute(l, left_needed, own=False)[0] and not self.rsc_compute(r, right_needed, own=False)[0]:
                    rsc_trade.append((left_needed, right_needed))
        return rsc_trade
    
    def coin(self):
        return [self.state[i]["coin"] for i in range(self.n)]

    def coin_trade(self, i, pick, action, free):
        # Step 1. If the Action is DISCARD, return []
        if action == Action.DISCARD:
            return []
        # Step 2. check if the player hasn't built the card(Action.BUILD) or there is available wonder to build (Action.WONDER)
        # Step 2. (continue) If not, return None
        # Step 3. Check if the card can be built with upgrade or free. If yes, return []
        coin = self.state[i]["coin"]
        wonders = CIVS[self.civs[i]]["wonders"][self.faces[i]]
        stage = self.color[i]["wonder"]
        color = CARDS[pick]["color"]
        prev = CARDS[pick]["prev"]

        if action == Action.BUILD:
            if pick in self.built[i]:
                return None
            elif free:
                return []
            elif prev and set(prev) & self.built[i]: # upgrade
                return []
            elif color in {"brown", "grey"}:
                if coin >= CARDS[pick]["cost"]["coin"]:
                    return []
                else:
                    return None
            else:
                cost = Counter(CARDS[pick]["cost"]) - Counter()
        elif action == Action.WONDER:
            if stage == len(wonders):
                return None
            else:
                cost = Counter(wonders[stage]["cost"]) - Counter()
        # Step 4. Check all possible combinations of resource needed to trade. 
        # Step 4. (continue) If the neighbors don't have enough resource, return None
        # Step 4. (continue) If the player has enough resource, return []
        rsc_trade = self.rsc_trade(i, cost) # [(Counter(left), Counter(right)), ...]
        if not rsc_trade:
            # neighbors don't have enough resource
            return None
        elif sum(left.total() + right.total() for left, right in rsc_trade) == 0:
            # player has enough resource
            return []
        # Step 5. Check if the player has any trading privilege (East/West Trading Post,  Marketplace)
        # Step 6. Find out all possible combinations of coins to trade
        # Step 7. If the player can't afford the least amount of total coins needed, return None
        # Step 8. Ask the player to choose one of the combinations [(coinToPrev, coinToNext), ...]
        coin_per_basic_left = 1 if "West Trading Post" in self.built[i] else 2
        coin_per_basic_right = 1 if "East Trading Post" in self.built[i] else 2
        coin_per_rare = 1 if "Marketplace" in self.built[i] else 2
        coins = set()
        for left, right in rsc_trade:
            left_coin = sum(left[rsc] * coin_per_basic_left for rsc in BASIC_RSC) 
            left_coin += sum(left[rsc] * coin_per_rare for rsc in RARE_RSC)
            right_coin = sum(right[rsc] * coin_per_basic_right for rsc in BASIC_RSC)
            right_coin += sum(right[rsc] * coin_per_rare for rsc in RARE_RSC)
            if left_coin + right_coin <= coin:
                coins.add((left_coin, right_coin))
        coins = sorted(coins, key=lambda x:(sum(x), x[0]))
        return coins if coins else None

    def send_notice(self, notice):
        for i in range(self.n):
            # player-centric notice
            notice_i = deepcopy(notice)
            for key, val in notice_i.items():
                if type(val) == list:
                    for _ in range(i):
                        val.append(val.pop(0))
            # if notice type is UPDATE, mask others' pick if their action is DISCARD or WONDER
            if notice_i["type"] == "UPDATE":
                for j in range(1, self.n): # skip self
                    move = notice_i["moves"][j]
                    if move and move["action"] != "BUILD":
                        move["pick"] = None
            # send notice to player i
            self.players[i].recv_notice(notice_i)
            

    def recv_face(self, i):
        player = self.players[i]
        # initialize the state
        state_og = [
            {
                "civ": name, 
                "face": None, 
                "rsc": self.rsc[j],
                "rsc_tradable": self.rsc_tradable[j],
                "shield": 0,
                "symbol": self.symbol[j],
                "color": self.color[j],
                "coin": 3,
                "built": [],
                "wonder": [],
                "discard": []
            } 
            for j, name in zip(range(self.n), self.civs)
        ]
        # player-centric state
        for _ in range(i):
            state_og.append(state_og.pop(0))
        # ask player
        while True:
            state = deepcopy(state_og)
            face = player.send_face(state)
            if face in ["Day", "Night"]:
                break
        return face
    
    def recv_move(self, i, scavenge=False):
        player = self.players[i]
        hand_og = deepcopy(self.hands[i]) if not scavenge else deepcopy(self.discard)
        record_og = deepcopy(self.record[i])
        state_og = deepcopy(self.state)
        # player-centric state
        for _ in range(i):
            state_og.append(state_og.pop(0))
        # shuffle the hand to avoid information leakage
        random.shuffle(hand_og)

        asked = False
        # ask player to make the move, repeat until the player make a legit move
        while True:
            state = deepcopy(state_og)
            record = deepcopy(record_og)
            hand = deepcopy(hand_og)
            pick, action = player.send_move(state, record, hand, asked)
            asked = True
            # Ensure no illegal pick or action
            if pick not in hand_og or action not in list(Action):
                continue
            # Ensure no wonder built in scavenging round
            if scavenge and action == Action.WONDER:
                continue
            # check free condition
            free = False
            free |= (self.free_first == i) and (self.turn % 6 == 1)
            free |= (self.free_last == i) and (self.turn % 6 == 0)
            free |= (self.free_color == i) and (self.color[i][CARDS[pick]["color"]] == 0)
            free |= scavenge

            # check if the player has enough coins to trade
            coins_og = self.coin_trade(i, pick, action, free)
            if coins_og is not None:
                break

        # coin trade
        while True:
            state = deepcopy(state_og)
            record = deepcopy(record_og)
            coins = deepcopy(coins_og)
            trade = player.send_trade(state, record, coins) if coins else (0, 0)
            if trade in coins_og or trade == (0, 0):
                break
        return pick, action, trade
    
    def update(self, moves, scavenge=False):
        # moves = [(pick, action, (#coin left, #coin right)), ...
        # scavenge discard doesn't generate 3 coins
        # brown card might cost coins instead of resources, need to deal seperately

        # Coin transfer (self.coin, self.state)
        # Card transfer basics (self.state, self.record, self.built, self.color, self.hands, self.discard)
        # Card-specific (self.rsc, self.rsc_tradable, self.symbol, self.shield)
        # Wonder-specific (self.free_first, self.free_last, self.free_color, self.seven, self.scavenge)
        # record (turn, pick, action, hand)

        # FIRST : coin, rsc, symbol, shield, specific
        for i, x in enumerate(moves):
            if not x:
                continue
            pick, action, (coin_left, coin_right) = x
            # coin transfer coin
            l = (i - 1) % self.n
            r = (i + 1) % self.n
            self.state[i]["coin"] -= coin_left
            self.state[i]["coin"] -= coin_right
            self.state[l]["coin"] += coin_left
            self.state[r]["coin"] += coin_right
            # discard
            if action == Action.DISCARD:
                if not scavenge:
                    self.state[i]["coin"] += 3
            elif action == Action.WONDER:
                stage = self.color[i]["wonder"]
                wonder = CIVS[self.civs[i]]["wonders"][self.faces[i]][stage]
                if wonder["func"]["coin"]:
                    self.state[i]["coin"] += wonder["func"]["coin"]
                if wonder["func"]["rsc"]:
                    rsc, cnt = wonder["func"]["rsc"]
                    self.rsc[i] += Counter({rsc: cnt})
                if wonder["func"]["shield"]:
                    self.shield[i] += wonder["func"]["shield"]
                if wonder["func"]["symbol"]:
                    self.symbol[i] += Counter({wonder["func"]["symbol"]:1})
                if wonder["func"]["others"]:
                    if wonder["func"]["others"] == "free_first":
                        self.free_first = i
                    if wonder["func"]["others"] == "free_last":
                        self.free_last = i
                    if wonder["func"]["others"] == "free_color":
                        self.free_color = i        
                    if wonder["func"]["others"] == "seven":
                        self.seven = i
                    if wonder["func"]["others"] == "scavenge":
                        self.scavenge = i
            ### Action.BUILD 
            # brown, grey : rsc, rsc_tradable (pay coin)
            # blue : None
            # red : shield
            # green : symbol
            # yellow: coin, rsc
            # purple: symbol
            else:
                assert action == Action.BUILD
                card = CARDS[pick]
                color = card["color"]
                if color in {"brown", "grey"}:
                    if not scavenge:
                        self.state[i]["coin"] -= card["cost"]["coin"]
                    rsc, cnt = card["func"]["rsc"]  
                    self.rsc[i] += Counter({rsc: cnt})
                    self.rsc_tradable[i] += Counter({rsc: cnt})
                elif color == "red":
                    self.shield[i] += card["func"]["shield"]
                elif color == "green":
                    self.symbol[i] += Counter({card["func"]["symbol"]:1})
                elif color == "yellow":
                    if card["func"]["rsc"]:
                        rsc, cnt = card["func"]["rsc"]
                        self.rsc[i] += Counter({rsc: cnt})
                    x = card["func"]["coin"]
                    if type(x) == int:
                        self.state[i]["coin"] += x
                    elif type(x) == list:
                        offsets, colors, coin_per_card = x
                        cnt = sum(self.color[(i + offset) % self.n][color] for offset, color in itertools.product(offsets, colors))
                        cnt += int("yellow" in colors) # yellow card itself
                        self.state[i]["coin"] += cnt * coin_per_card
                elif color == "purple":
                    if card["func"]["symbol"]:
                        self.symbol[i] += Counter({card["func"]["symbol"]:1})
                else:
                    assert color == "blue"
        # SECOND : self.state, self.record, self.built, self.color, self.hands, self.discard
        for i, x in enumerate(moves):
            if not x:
                continue
            pick, action, (coin_left, coin_right) = x
            self.record[i].append((self.turn, pick, action, deepcopy(self.hands[i])))
            if not scavenge:
                self.hands[i].remove(pick)
            if action == Action.DISCARD:
                if not scavenge:
                    self.discard.append(pick)
                    self.state[i]["discard"].append(self.turn)
            elif action == Action.WONDER:
                self.state[i]["wonder"].append(self.turn)
                self.color[i]["wonder"] += 1
            else:
                assert action == Action.BUILD
                self.state[i]["built"].append((self.turn, pick))
                self.built[i].add(pick)
                self.color[i][CARDS[pick]["color"]] += 1
        # THIRD : self.state[i] (rsc, rsc_tradable, symbol, shield)
        for i in range(self.n):
            ###self.state[i]["rsc"] = self.rsc[i]
            ###self.state[i]["rsc_tradable"] = self.rsc_tradable[i]
            ###self.state[i]["symbol"] = self.symbol[i]
            self.state[i]["shield"] = self.shield[i]
        # send update notice to all players
        moves = deepcopy(moves)
        for i, move in enumerate(moves):
            if move:
                pick, action, trade = move
                moves[i] = {"pick": pick, "action": action.name, "trade": list(trade)}
        notice = {
            "type": "UPDATE",
            "scavenge": scavenge,
            "moves": moves,
            "coins": self.coin()
        }
        self.send_notice(notice)

    def clear(self):
        # send setting notice to all players
        notice = {
            "type": "CLEAR"
        }
        self.send_notice(notice)
        for hand in self.hands:
            while hand:
                self.discard.append(hand.pop())

    def battle(self):
        # each player only battle with its neighbors
        # Age I : WIN 1, LOSE -1, TIE 0
        # Age II: WIN 3, LOSE -1, TIE 0
        # Age III: WIN 5, LOSE -1, TIE 0
        WIN = 2 * (self.turn // 6) - 1
        LOSE = -1
        TIE = 0
        scores = [[0, 0] for _ in range(self.n)]
        for i in range(self.n):
            l = (i - 1) % self.n
            r = (i + 1) % self.n
            for j, x in zip([0, 1], [l, r]):
                if self.shield[i] > self.shield[x]:
                    score = WIN
                elif self.shield[i] < self.shield[x]:
                    score = LOSE
                else:
                    score = TIE
                self.conflict[i] += score
                scores[i][j] = score
        # send battle notice to all players
        notice = {
            "type": "BATTLE",
            "battle": scores
        }
        self.send_notice(notice)
    
    def calculate(self):
        for i in range(self.n):
            l = (i - 1) % self.n
            r = (i + 1) % self.n
            wonders = CIVS[self.civs[i]]["wonders"][self.faces[i]]
            # wealth score
            self.wealth[i] = self.state[i]["coin"] // 3
            # wonder score
            for stage in range(self.color[i]["wonder"]): 
                s = wonders[stage]["func"]["score"]
                if s:
                    self.wonder[i] += s
            # science score
            symbol_seed = self.symbol[i]
            symbol_gen = map(lambda x: x.split("|"), symbol_seed.elements())
            symbol_cnt = sorted({tuple(Counter(symbols)[symbol] for symbol in ALL_SYMBOL) for symbols in itertools.product(*symbol_gen)})
            for cnts in symbol_cnt:
                science = 0
                science += sum([cnt ** 2 for cnt in cnts])
                science += 7 * min(cnts)
                self.science[i] = max(self.science[i], science)            
            # blue, yellow, purple score
            for name in self.built[i]:
                card = CARDS[name]
                color = card["color"]
                if color in {"blue", "yellow", "purple"}:
                    x = card["func"]["score"]
                if color == "blue":
                    self.civilian[i] += x
                elif color == "yellow":
                    if type(x) == int:
                        self.commerce[i] += x
                    elif type(x) == list:
                        offsets, colors, score_per_card = x
                        cnt = sum(self.color[(i + offset) % self.n][color] for offset, color in itertools.product(offsets, colors))
                        self.commerce[i] += cnt * score_per_card
                elif color == "purple":
                    if type(x) == int:
                        self.guild[i] += x
                    elif type(x) == list:
                        offsets, colors, score_per_card = x
                        cnt = sum(self.color[(i + offset) % self.n][color] for offset, color in itertools.product(offsets, colors))
                        self.guild[i] += cnt * score_per_card
                    # Decorators Guild
                    if name == "Decorators Guild" and self.color[i]["wonder"] == len(wonders):
                        self.guild[i] += 7
            # total score
            self.total[i] = sum([self.civilian[i], self.conflict[i], self.science[i], self.commerce[i], self.guild[i], self.wonder[i], self.wealth[i]])
        scores = {
            "civilian": self.civilian,
            "conflict": self.conflict,
            "science": self.science,
            "commerce": self.commerce,
            "guild": self.guild,
            "wonder": self.wonder,
            "wealth": self.wealth,
            "total": self.total,
            "coin": self.coin()
        }
        # send score notice to all players
        notice = {"type": "SCORE"}
        notice.update(scores)
        self.send_notice(notice)
        return scores

    def init(self):
        # send setting notice to all players
        notice = {
            "type": "SETTING",
            "civs": self.civs,
            "faces": self.faces
        }
        self.send_notice(notice)
        # determine the face for each civ
        if not self.faces:
            # concurrent face selection
            with ThreadPoolExecutor() as executor:
                self.faces = list(executor.map(self.recv_face, range(self.n)))

            # send setting notice to all players
            notice = {
                "type": "SETTING",
                "civs": self.civs,
                "faces": self.faces
            }
            self.send_notice(notice)

        # initialize the state
        self.state = [
            {
                "civ": name, 
                "face": face, 
                "rsc": self.rsc[i],
                "rsc_tradable": self.rsc_tradable[i],
                "shield": 0,
                "symbol": self.symbol[i],
                "color": self.color[i],
                "coin": 3,
                "built": [],
                "wonder": [],
                "discard": []
            } 
            for i, name, face in zip(range(self.n), self.civs, self.faces)
        ]

    def run(self, verbose=logging.INFO):
        # set the logger
        self.logger.setLevel(verbose)
        # check if all players are registered
        assert all(self.players)
        # initialize the game
        self.init()
        # routine
        while self.turn <= 18:
            ### distribute cards in the beginning of each age
            if self.turn % 6 == 1:
                # send age notice to all players
                notice = {
                    "type": "AGE",
                    "age": self.turn // 6 + 1
                }
                self.send_notice(notice)
                deck = self.deck[self.turn // 6]
                for j, name in enumerate(deck):
                    self.hands[j % self.n].append(name)
            ### regular turn 
            # players make moves (pick, action, trade)
            # concurrent move selection
            with ThreadPoolExecutor(self.n) as executor:
                moves = list(executor.map(self.recv_move, range(self.n)))
            # update 
            self.update(moves)
            # additional move ("seven" for Babylon)
            if self.seven is not None and self.turn % 6 == 0:
                moves = [None] * self.n
                moves[self.seven] = self.recv_move(self.seven)
                self.update(moves)
            # clear desk at the end of each age
            if self.turn % 6 == 0:
                self.clear()
                self.logger.info("Remove all hands")
            # additional move ("scavenge" for Halikarnassos)
            if self.scavenge is not None:
                if self.discard:
                    pick, action, trade = self.recv_move(self.scavenge, scavenge=True)
                    moves = [None] * self.n
                    moves[self.scavenge] = (pick, action, trade)
                    self.update(moves, scavenge=True)
                    if action == Action.BUILD:
                        self.discard.remove(pick)
                else:
                    self.logger.warning("Sorry ~ No card to scavenge")
                self.scavenge = None
            # rotate the hands
            # clockwise (Age I, III)
            # counter-clockwise (Age II)
            if (self.turn // 6) % 2 == 1: # counter-clockwise
                self.hands.append(self.hands.pop(0))
            else: # clockwise
                self.hands.insert(0, self.hands.pop())
            ### military conflict at the end of each age
            if self.turn % 6 == 0:
                self.logger.info("Age %d finished" % (self.turn // 6)) ###
                self.logger.info("Battle") ###
                self.battle()
            # go to next turn
            self.turn += 1

        # Calculate score
        scores = self.calculate()
        # Show score
        items = ["civilian", "conflict", "science", "commerce", "guild", "wonder", "wealth", "coin", "total"]
        for i in range(self.n):
            state = self.state[i]
            color_num = self.color[i]
            civ = state["civ"]
            face = state["face"]
            self.logger.info("")
            self.logger.info("=" * 20 + " %s %s " % (civ, face) + "=" * 20)
            self.logger.info("Scores: %s" % {item: scores[item][i] for item in items})
            self.logger.debug("Colors: %s" % color_num)
            for color in ALL_COLORS[:-1]:
                x = [(turn, pick) for turn, pick in state["built"] if CARDS[pick]["color"] == color]
                self.logger.debug("%s %d %s" % (color, len(x), x))
            self.logger.debug("-----------")
            self.logger.debug("wonder %s" % state["wonder"])
            self.logger.debug("discard %s" % state["discard"])
            self.logger.debug("-----------")
            self.logger.debug("coin %s" % state["coin"])
            self.logger.debug("rsc %s" % self.rsc[i])
            self.logger.debug("rsc_tradable %s" % self.rsc_tradable[i])
            self.logger.debug("shield %s" % self.shield[i])
            self.logger.debug("symbol %s" % self.symbol[i])
        # Send player-centric score to players
        for i in range(self.n):
            scores_i = deepcopy(scores)
            for item in items:
                for _ in range(i):
                    scores_i[item].append(scores_i[item].pop(0))
            self.players[i].recv_score(scores_i)
        # Collect history from registered players
        history = [self.players[i].record for i in range(self.n)]
        # Remove player's game record
        for i in range(self.n):
            self.players[i].record = []
        ### Reset game
        # export game configuration
        n, civs, faces, deck, random_face = self.export()
        # preserve players
        players = self.players
        # if random_face is set to False, remove the faces
        if not random_face:
            faces = None
        self.__init__(n, civs=civs, faces=faces, deck=deck, random_face=random_face)
        self.players = players
        return history


if __name__ == "__main__":
    from player import RandomPlayer, HumanPlayer
    n = 3
    #['Alexandria', 'Babylon', 'Éphesos', 'Gizah', 'Halikarnassos', 'Olympia', 'Rhódos']
    game = Game(n, random_face=False)
    players = [RandomPlayer() for _ in range(n)]
    for i in range(n):
        game.register(i, players[i])
    game.run(verbose=logging.DEBUG)
