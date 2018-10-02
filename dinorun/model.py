import configparser
import os

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from .settings import settings

config = configparser.ConfigParser()
config.read('./config.ini')


def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8),
                     padding='same',
                     strides=(4, 4),
                     input_shape=(settings['img_cols'],
                                  settings['img_rows'],
                                  settings['img_channels'])))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(settings['num_actions']))
    adam = Adam(lr=settings['learning_rate'])
    model.compile(loss='mse', optimizer=adam)

    if not os.path.isfile(config['CONFIG']['loss_file_path']):
        model.save_weights('model.h5')
    return model
