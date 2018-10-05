from dinorun.agent import Agent
from dinorun.game import Game
from dinorun.helpers import init_cache
from dinorun.model import Model
from dinorun.state import State
from dinorun.train import train


def play_game(observe=False):
    game = Game()
    dino = Agent(game)
    game_state = State(dino, game)
    model = Model()
    try:
        train(model, game_state, observe)
    except StopIteration:
        game.end()


if __name__ == '__main__':
    init_cache()
    play_game()
