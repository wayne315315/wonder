import random

from player import RandomPlayer
from game import Game, CARDS
from tqdm import tqdm


def extract(history):
    n = len(history)
    rec_valid = [[] for _ in range(n)]
    rec_invalid = [[] for _ in range(n)]
    # (api, args, res, is_valid)
    for i in range(n):
        _, [scores, *_], _, _ = history[i].pop() # (api, args, res, is_valid) : (str, list, list, bool)
        rewards = list(zip(scores["total"], scores["coin"]))
        for rec in history[i]:
            api, args, res, is_valid = rec
            if is_valid == True:
                idx = len(rec_valid[i])
                rec.append(idx)
                rec.append(rewards)
                rec_valid[i].append(rec) # rec: (api, args, res, is_valid, idx, rewards)
            else:
                rec_invalid[i].append(rec) # rec: (api, args, res, is_valid)
    recs = []
    for i in range(n):
        for rec in rec_valid[i]:
            recs.append(rec)
        for rec in rec_invalid[i]:
            recs.append(rec)
    return recs

def data_gen(num_play, num_game):
    pbar = tqdm(total=num_game * num_play * 5)
    data = []
    for n in range(3,8):
        for num in range(num_game):
            game = Game(n, random_face=False)
            players = [RandomPlayer() for _ in range(n)]
            for i, player in enumerate(players):
                game.register(i, player)
            for epoch in range(num_play):
                history = game.run(verbose=50)
                data.extend(extract(history))
                pbar.update(1)
    random.shuffle(data)
    pbar.close()
    return data

"""   
def meow(x, exp=2):
    x = np.asarray(x)
    x = np.sign(x) * np.power(np.abs(x), exp)
    return x / np.std(x)
"""

# data generation
# Each game has around 18 moves for each player
# Same game plays for 100 times
# Same player number plays for 100 times
# Player number : 3 - 7
# total data : 18 * (3+4+5+6+7) * 100 * 100 = 4500000 at least

if __name__ == "__main__":
    num_play = 2 # The number of rehearsals for each game
    num_game = 2 # The number of games for each number of the total players
    data = data_gen(num_play, num_game)
    for rec in data:
        api, args, res, is_valid, *extras = rec
        if is_valid:
            idx, rewards = extras
            print(api,res,idx,rewards)
