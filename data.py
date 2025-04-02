import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from player import RandomPlayer
from rl import AIPlayer
from game import Game


def get_reward_old(scores, exp=2):
    x = list(zip(scores["total"], scores["coin"])) # eg. [(42, 4), (61,2), (42, 5)]
    n = len(x)
    y = [0] * n
    for i in range(n):
        for j in range(n):
            if x[i] > x[j]:
                y[i] += 1
            elif x[i] < x[j]:
                y[i] -= 1
    y = np.asarray(y)
    y = np.sign(y) * np.power(np.abs(y), exp)
    y = y / np.std(y)
    y += np.log(1 + np.asarray(scores["total"])) # encourage to pursue higher score
    reward = y[0]
    return reward

def get_reward(scores):
    totals, coins = scores["total"], scores["coin"]
    x = list(zip(totals, coins)) # eg. [(42, 4), (61,2), (42, 5)]
    n = len(x)
    y = [0] * n
    for i in range(n):
        for j in range(n):
            if x[i] > x[j]:
                y[i] += 1
            elif x[i] < x[j]:
                y[i] -= 1
    y = np.asarray(y)
    y = np.sign(y) * np.power(np.abs(y), 2)
    reward = y[0] + sum([totals[0] - totals[i] for i in range(n)])
    reward /= n
    return reward

def extract(history):
    n = len(history)
    rec_valid = [[] for _ in range(n)]
    rec_invalid = [[] for _ in range(n)]
    # (api, args, res, is_valid)
    for i in range(n):
        _, [scores, *_], _, _ = history[i].pop() # (api, args, res, is_valid) : (str, list, list, bool)
        reward = get_reward(scores)
        for rec in history[i]:
            api, args, res, is_valid = rec
            if is_valid == True:
                rec = [api, args, res, reward]
                rec_valid[i].append(rec) # rec: (api, args, res, reward)
            else:
                rec = [api, args, res]
                rec_invalid[i].append(rec) # rec: (api, args, res)
    episodes = [(rec_valid[i], rec_invalid[i]) for i in range(n)]
    return episodes

def epi_gen(arg):
    n, num_play, *extras = arg
    game = Game(n, random_face=False)
    players = [RandomPlayer() for _ in range(n)]
    if extras:
        model = extras[0]
        players[0] = AIPlayer(model)
    episodes = []
    for i, player in enumerate(players):
        game.register(i, player)
    for epoch in range(num_play):
        history = game.run(verbose=50)
        episodes.extend(extract(history))
    random.shuffle(episodes)
    return episodes

def data_gen(num_play, num_game, model=None, shard=5, size=5, max_workers=None):
    data = [[] for _ in range(shard)]
    args = [(n, num_play, model) if model else (n, num_play) for n in range(3,8) for _ in range(num_game)]
    random.shuffle(args)
    if model:
        for arg in args:
            for episode in epi_gen(arg):
                i = random.randint(0, shard-1)
                data[i].append(episode)
                # shard reach size
                if len(data[i]) >= size:
                    random.shuffle(data[i])
                    while data[i]:
                        yield data[i].pop()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for episodes in executor.map(epi_gen, args):
                for episode in episodes:
                    i = random.randint(0, shard-1)
                    data[i].append(episode)
                    # shard reach size
                    if len(data[i]) >= size:
                        random.shuffle(data[i])
                        while data[i]:
                            yield data[i].pop()
    random.shuffle(data)
    for datum in data:
        random.shuffle(datum)
        while datum:
            yield datum.pop()

# data generation
# Same game plays for 100 times
# Same player number plays for 100 times
# Player number : 3 - 7
# total data : (3+4+5+6+7) * 100 * 100 = 250000 at least

if __name__ == "__main__":
    from tqdm import tqdm
    from helper import Adaptor
    adaptor = Adaptor()
    num_play = 10 # The number of rehearsals for each game
    num_game = 30 # The number of games for each number of the total players
    data = data_gen(num_play, num_game)
    for episode in tqdm(data, total=num_play * num_game * 25):
        rec_valid, rec_invalid = episode
        print("")
        print(len(rec_valid))
        for rec in rec_valid:
            api, args, res, reward = rec
            print(api, reward)
        print("")
        print(len(rec_invalid))
        for rec in rec_invalid:
            api, args, res = rec
