import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np

from player import RandomPlayer
from rl import AIPlayer, AIPlayer2
from game import Game


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
    reward_min = float("inf")
    reward_max = float("-inf")
    # (api, args, res, is_valid)
    for i in range(n):
        _, [scores, *_], _, _ = history[i].pop() # (api, args, res, is_valid) : (str, list, list, bool)
        reward = get_reward(scores)
        if i:
            if reward < reward_min:
                reward_min = reward
                i_reward_min = i
            if reward > reward_max:
                reward_max = reward
                i_reward_max = i
        for rec in history[i]:
            api, args, res, is_valid = rec
            if is_valid == True:
                rec = [api, args, res, reward]
                rec_valid[i].append(rec) # rec: (api, args, res, reward)
            else:
                rec = [api, args, res]
                rec_invalid[i].append(rec) # rec: (api, args, res)
    
    #selected = [0, i_reward_max, i_reward_min]
    # selected players: include player 0, the best player and the worst player
    selected = [0] # only select the player 0
    episodes = [(rec_valid[i], rec_invalid[i]) for i in selected]
    return episodes


def epi_gen(game):
    history = game.run(verbose=50)
    episodes = extract(history)
    return episodes


def data_gen(num_play, num_game, model=None, serve_name=None, serve_version=None, exploited=None):
    # Create all games & plays
    games = [Game(n, random_face=False) for n in range(3, 8) for _ in range(num_game)]
    games = [deepcopy(g) for g in games for _ in range(num_play)]
    for game in games:
        players = [RandomPlayer() for _ in range(game.n)]
        if model:
            players[0] = AIPlayer(model)
            # the exploiter will be trained against all exploited models
            for i in range(1, game.n):
                players[i] = AIPlayer2(exploited) if exploited else random.choice([AIPlayer(model), players[i]])
        elif serve_name:
            players[0]  = AIPlayer2(serve_name, serve_version=serve_version) 
            for i in range(1, game.n):
                players[i] = AIPlayer2(exploited) if exploited else random.choice([AIPlayer2(serve_name, serve_version=serve_version), players[i]])
        for i in range(game.n):
            game.register(i, players[i])
    max_workers = min(len(games), 1024)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2n = {executor.submit(epi_gen, game): game.n for game in games}
        for f in as_completed(f2n):
            episodes = f.result()
            for episode in episodes:
                yield episode

# data generation
# Same game plays for 10 times
# Same player number plays for 10 times
# Player number : 3 - 7
# total data : 3 * 5 * 10 * 10 = 1500 at least

if __name__ == "__main__":
    from tqdm import tqdm
    from helper import Adaptor
    adaptor = Adaptor()
    num_play = 2 # The number of rehearsals for each game
    num_game = 2 # The number of games for each number of the total players
    data = data_gen(num_play, num_game)
    for episode in tqdm(data, total=num_play * num_game * 5):
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
