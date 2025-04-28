import random

from helper import Adaptor
from player import Player
from serving import send_grpc_request


class AIPlayer(Player):
    def __init__(self, model):
        super().__init__()
        self.helper = Adaptor()
        self.model = model

    def _send_face(self, state):
        v = self.helper.s2v(state, [])
        face = random.choice(["Day", "Night"]) # [[API_CALL, *args, *res, is_valid]]
        return face

    def _send_move(self, state, record, hand, asked):
        if not asked:
            v = self.helper.s2v(state, record) # shape (19 * n + 6, 7)
            h = self.helper.h2v(hand)
            self.buffer = self.model.predict_move(v, h).numpy()[0].tolist()
        else:
            self.record[-1][-1] = False # last move is invalid
        output = self.buffer.pop()
        pick, action = self.helper.pair[output]
        return (pick, action)

    def _send_trade(self, state, record, coins):
        trade = coins[0]
        return trade

    def _recv_score(self, scores):
        pass


class AIPlayer2(AIPlayer):
    def __init__(self, serve_name, serve_version=None):
        Player.__init__(self)
        self.helper = Adaptor()
        self.serve_name = serve_name
        self.serve_version = serve_version

    def _send_move(self, state, record, hand, asked):
        if not asked:
            self.buffer = None
            v = self.helper.s2v(state, record) # shape (19 * n + 6, 7)
            h = self.helper.h2v(hand)
            while self.buffer is None:
                self.buffer = send_grpc_request(v, h, self.serve_name, serve_version=self.serve_version)
                self.buffer = self.buffer[0].tolist() if self.buffer is not None else None
        else:
            self.record[-1][-1] = False # last move is invalid
        output = self.buffer.pop()
        pick, action = self.helper.pair[output]
        return (pick, action)


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

    model = ActorCritic(len(CARDS))
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
