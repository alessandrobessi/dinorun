import configparser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from .canvas import canvas

config = configparser.ConfigParser()
config.read('./config.ini')
c = config['CONFIG']


class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('disable-infobars')
        chrome_options.add_argument('--mute-audio')
        self.driver = webdriver.Chrome(executable_path=c['chrome_driver_path'],
                                       chrome_options=chrome_options)
        self.driver.set_window_position(x=-10, y=0)
        self.driver.get(c['game_url'])
        self.driver.execute_script('Runner.config.ACCELERATION=0')
        self.driver.execute_script(canvas['init_script'])

    def get_crashed(self):
        return self.driver.execute_script('return Runner.instance_.crashed')

    def get_playing(self):
        return self.driver.execute_script('return Runner.instance_.playing')

    def restart(self):
        self.driver.execute_script('Runner.instance_.restart()')

    def press_up(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self.driver.execute_script('return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.driver.execute_script('return Runner.instance_.stop()')

    def resume(self):
        return self.driver.execute_script('return Runner.instance_.play()')

    def end(self):
        self.driver.close()
