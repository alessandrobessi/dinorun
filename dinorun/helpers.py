import base64
import configparser
import os
import pickle
from collections import deque
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from .canvas import canvas
from .settings import settings

config = configparser.ConfigParser()
config.read('./config.ini')


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_b64 = _driver.execute_script(canvas['get_base64_script'])
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = 'logs' if graphs else 'game_play'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def init_cache():
    """initial variable caching, done only once"""
    if not os.path.exists(os.path.join(os.getcwd(), 'objects')):
        os.makedirs(os.path.join(os.getcwd(), 'objects'))
    save_obj(settings['first_epsilon'], 'epsilon')
    t = 0
    save_obj(t, 'time')
    replay_memory = deque()
    save_obj(replay_memory, 'replay_memory')
