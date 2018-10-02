'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the broswer window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''
import configparser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

config = configparser.ConfigParser()
config.read('./config.ini')


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument('disable-infobars')
        chrome_options.add_argument('--mute-audio')
        self._driver = webdriver.Chrome(
            executable_path=config['CONFIG']['chrome_driver_path'],
            chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(config['CONFIG']['game_url'])
        self._driver.execute_script('Runner.config.ACCELERATION=0')
        self._driver.execute_script(config['CANVAS']['init_script'])

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
        score_array = self._driver.execute_script(
            'return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_array)
        # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script('return Runner.instance_.stop()')

    def resume(self):
        return self._driver.execute_script('return Runner.instance_.play()')

    def end(self):
        self._driver.close()
