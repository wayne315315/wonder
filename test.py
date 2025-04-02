if __name__ == "__main__":
    from pathlib import Path
    import tensorflow as tf
    from player import RandomPlayer
    from rl import AIPlayer
    from game import Game

    n = 4
    num_play = 10
    epoch = 10
    model_dir = Path("model")
    model_path = Path(model_dir, "ac.keras")
    print("load model")
    model = tf.keras.models.load_model(model_path)
    for _ in range(epoch):
        game = Game(n, random_face=False)
        players = [RandomPlayer() for _ in range(n)]
        players[0] = AIPlayer(model)
        for i in range(n):
            game.register(i, players[i])
        for _ in range(num_play):
            game.run()