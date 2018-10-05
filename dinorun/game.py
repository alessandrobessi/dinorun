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
        self._driver = webdriver.Chrome(executable_path=c['chrome_driver_path'],
                                        chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(c['game_url'])
        self._driver.execute_script('Runner.config.ACCELERATION=0')
        self._driver.execute_script(canvas['init_script'])

    def get_crashed(self):
        return self._driver.execute_script('return Runner.instance_.crashed')

    def get_playing(self):
        return self._driver.execute_script('return Runner.instance_.playing')

    def restart(self):
        self._driver.execute_script('Runner.instance_.restart()')

    def press_up(self):
        self._driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        """
        The javascript object is of type array with score
        in the format [1,0,0], that is 100
        """

        score_array = self._driver.execute_script(
            'return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self._driver.execute_script('return Runner.instance_.stop()')

    def resume(self):
        return self._driver.execute_script('return Runner.instance_.play()')

    def end(self):
        self._driver.close()
