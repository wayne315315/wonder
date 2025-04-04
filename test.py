import numpy as np
import tensorflow as tf
from player import RandomPlayer
from rl import AIPlayer
from game import Game


def test(modelpath, num_play=10, num_game=10):
    model = tf.keras.models.load_model(modelpath)
    for n in range(3,8):
        players = [RandomPlayer() for _ in range(n)]
        players[0] = AIPlayer(model)
        totals = []
        ranks = []
        for epoch in range(num_game):
            game = Game(n, random_face=False)
            # register players
            for i in range(n):
                game.register(i, players[i])
            # run same game multiple times
            for _ in range(num_play):
                history = game.run(verbose=20)
                _, [scores, *_], _, _ = history[0].pop()
                total = scores["total"]
                rank = [0] * n
                for i in range(n):
                    for j in range(n):
                        if total[i] > total[j]:
                            rank[i] += 1
                        elif total[i] < total[j]:
                            rank[i] -= 1
                totals.append(total)
                ranks.append(rank)
        totals = np.asarray(totals)
        totals = np.mean(totals, axis=0)
        ranks = np.asarray(ranks)
        ranks = np.mean(ranks, axis=0)
        print(totals, ranks)
        

if __name__ == "__main__":
    from pathlib import Path
    import logging

    model_dir = Path("model")
    model_path = Path(model_dir, "ac.keras")

    # set 
    logpath = Path("test.log")
    if logpath.exists():
        logpath.unlink()
    logging.basicConfig(filename=logpath)
    test(model_path)
   