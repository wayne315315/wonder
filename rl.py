import random

from helper import Adaptor
from player import Player


class AIPlayer(Player):
    def __init__(self, model):
        super().__init__()
        self.helper = Adaptor()
        self.model = model

    def send_face(self, state):
        v = self.helper.s2v(state, [])
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        self.record.append(["face", [state], [face], True])
        return face

    def send_move(self, state, record, hand, asked):
        if not asked:
            v = self.helper.s2v(state, record) # shape (19 * n + 6, 7)
            h = self.helper.h2v(hand)
            self.buffer = self.model.predict_move(v, h).numpy()[0].tolist()
        else:
            self.record[-1][-1] = False # last move is invalid

        output = self.buffer.pop()
        pick, action = self.helper.pair[output]
        self.record.append(["move", [state, record, hand], [pick, action], True])
        return (pick, action)

    def send_trade(self, state, record, coins):
        trade = coins[0]
        self.record.append(["trade", [state, record, coins], [trade], True])
        return trade

    def recv_score(self, scores):
        self.record.append(["score", [scores], [], True])


if __name__ == "__main__":
    from pathlib import Path
    import tensorflow as tf
    from player import RandomPlayer
    from model import ActorCritic
    from game import Game, CARDS

    model_dir = Path("model")
    if not model_dir.exists():
        model_dir.mkdir()
    model_path = Path(model_dir, "toy.keras")

    model = ActorCritic(len(CARDS), 128)
    n = 3
    game = Game(n)
    players = [RandomPlayer() for _ in range(n)]
    players[0] = AIPlayer(model)
    for i in range(n):
        game.register(i, players[i])

    game.run()
    model.save(model_path)
    ####
    print("load model")
    new_model = tf.keras.models.load_model(model_path)
    game = Game(n, random_face=False)
    players = [RandomPlayer() for _ in range(n)]
    players[0] = AIPlayer(new_model)
    for i in range(n):
        game.register(i, players[i])

    game.run()
