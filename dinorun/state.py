import pandas as pd

from .helpers import *

config = configparser.ConfigParser()
config.read('./config.ini')
c = config['CONFIG']


class State:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        # display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_img()
        # initiliaze the display coroutine
        self._display.__next__()
        self._initialize_log_files()

    def _initialize_log_files(self):
        self.loss_df = pd.read_csv(c['loss_file_path']) if \
            os.path.isfile(c['loss_file_path']) else pd.DataFrame(columns=['loss'])
        self.scores_df = pd.read_csv(c['scores_file_path']) if \
            os.path.isfile(c['scores_file_path']) else pd.DataFrame(columns=['scores'])
        self.actions_df = pd.read_csv(c['actions_file_path']) if \
            os.path.isfile(c['actions_file_path']) else pd.DataFrame(columns=['actions'])
        self.q_values_df = pd.read_csv(c['q_values_file_path']) if \
            os.path.isfile(c['q_values_file_path']) else pd.DataFrame(columns=['q_values'])

    def get_state(self, actions):
        self.actions_df.loc[len(self.actions_df)] = actions[1]
        score = self._game.get_score()
        reward = 0.1
        is_over = False  # game over
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver)
        self._display.send(image)  # display the image on screen
        if self._agent.is_crashed():
            self.scores_df.loc[
                len(self.loss_df)] = score  # log the score when game is over
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over  # return the Experience tuple

    def update_loss(self, loss):
        self.loss_df.loc[len(self.loss_df)] = loss

    def update_q_values(self, q_value):
        self.q_values_df.loc[len(self.q_values_df)] = q_value

    def save(self):
        self.loss_df.to_csv(c['loss_file_path'], index=False)
        self.scores_df.to_csv(c['scores_file_path'], index=False)
        self.actions_df.to_csv(c['actions_file_path'], index=False)
        self.q_values_df.to_csv(c['q_values_file_path'], index=False)
