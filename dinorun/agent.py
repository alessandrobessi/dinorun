class Agent:
    def __init__(self, game):
        self.game = game
        self.jump()

    def is_running(self):
        return self.game.get_playing()

    def is_crashed(self):
        return self.game.get_crashed()

    def jump(self):
        self.game.press_up()

    def duck(self):
        self.game.press_down()
