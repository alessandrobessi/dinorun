from dinorun.game import Game
from dinorun.agent import Agent
from dinorun.state import State
from dinorun.model import build_model
from dinorun.train import train
from dinorun.helpers import init_cache


def play_game(observe=False):
    game = Game()
    dino = Agent(game)
    game_state = State(dino, game)
    model = build_model()
    try:
        train(model, game_state, observe=observe)
    except StopIteration:
        game.end()


if __name__ == '__main__':
    init_cache()
    play_game()
