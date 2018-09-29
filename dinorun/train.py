'''
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''
import random
import time
import json
import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard

from .helpers import *
from .settings import settings


def train(model, game_state, observe=False):
    last_time = time.time()
    # store the previous observations in replay memory
    replay_memory = load_obj('replay_memory')  # load from file system
    # get the first state by doing nothing
    init_action = np.zeros(settings['num_actions'])
    init_action[0] = 1  # 0 => do nothing, 1=> jump

    x_t, r_0, terminal = game_state.get_state(init_action)  # get next step after action

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

    initial_state = s_t

    if observe:
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = settings['final_epsilon']
    else:  # We go to training mode
        OBSERVE = settings['observation']
        epsilon = load_obj('epsilon')

    model.load_weights('model.h5')
    adam = Adam(lr=settings['learning_rate'])
    model.compile(loss='mse', optimizer=adam)

    t = load_obj('time')  # resume from the previous time step stored in file system
    while True:  # endless running

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # reward at 4
        a_t = np.zeros([settings['num_actions']])  # action at t

        # choose an action epsilon greedy
        if t % settings['frames_per_action'] == 0:  # parameter to skip frames for actions
            if random.random() <= epsilon:  # randomly explore an action
                print("----------Random Action----------")
                action_index = random.randrange(settings['num_actions'])
                a_t[action_index] = 1
            else:  # predict the output
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)  # chosing index with maximum q value
                action_index = max_Q
                a_t[action_index] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > settings['last_epsilon'] and t > OBSERVE:
            epsilon -= (settings['first_epsilon'] - settings['last_epsilon']) / settings['explore']

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        # append the new image to input stack and remove the first one

        # store the transition in D
        replay_memory.append((s_t, action_index, r_t, s_t1, terminal))
        if len(replay_memory) > settings['len_replay_memory']:
            replay_memory.popleft()

        # only train if done observing
        if t > OBSERVE:

            # sample a minibatch to train on
            mini_batch = random.sample(replay_memory, settings['batch_size'])
            inputs = np.zeros((settings['batch_size'], s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], settings['num_actions']))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(mini_batch)):
                state_t = mini_batch[i][0]  # 4D stack of images
                action_t = mini_batch[i][1]  # This is action index
                reward_t = mini_batch[i][2]  # reward at state_t due to action_t
                state_t1 = mini_batch[i][3]  # next state
                terminal = mini_batch[i][4]  # whether the agent died or survided due the action

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # predict q values for next step

                if terminal:
                    targets[i, action_t] = reward_t  # if terminated, only equals reward
                else:
                    targets[i, action_t] = reward_t + settings['gamma'] * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            game_state.update_loss(loss)
            game_state.update_q_values(np.max(Q_sa))
        s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            game_state._game.pause()  # pause game while saving to filesystem
            model.save_weights('model.h5', overwrite=True)
            save_obj(replay_memory, 'replay_memory')  # saving episodes
            save_obj(t, "time")  # caching time steps
            save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions

            game_state.save()

            with open('model.json', 'w') as f:
                json.dump(model.to_json(), f)

            game_state._game.resume()
        # print info
        state = ""
        if t <= OBSERVE:
            state = 'observe'
        elif OBSERVE < t <= OBSERVE + settings['explore']:
            state = 'explore'
        else:
            state = 'train'

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)
