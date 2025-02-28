import random
import json
from enum import Enum

ALL_COLORS = ["brown", "grey", "blue", "red", "green", "yellow", "purple", "wonder"]
ALL_RSC = ["wood", "stone", "brick", "ore", "glass", "papyrus", "cloth"]
RARE_RSC = ["glass", "papyrus", "cloth"]
BASIC_RSC = ["wood", "stone", "brick", "ore"]
ALL_SYMBOL = ["compass", "tombstone", "wheel"]

class Action(Enum):
    BUILD = 1
    WONDER = 2
    DISCARD = 3

with open("card.json", "rt") as f:
    CARDS = json.load(f)

with open("civ.json", "rt") as f:
    CIVS = json.load(f)

with open("deck.json", "rt") as f:
    DECK = json.load(f)
