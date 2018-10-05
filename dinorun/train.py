import random
import time

import torch
import torch.nn as nn

from .helpers import *


def train(model, game_state, observe):
    last_time = time.time()
    # store the previous observations in replay memory
    replay_memory = load_obj('replay_memory')  # load from file system
    # get the first state by doing nothing
    init_action = np.zeros(settings['num_actions'])
    init_action[0] = 1  # 0 => do nothing, 1=> jump

    x_t, r_0, terminal = game_state.get_state(init_action)  # get next step after action

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    initial_state = s_t

    if observe:
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = settings['final_epsilon']
    else:
        OBSERVE = settings['observation']
        epsilon = load_obj('epsilon')

    if os.path.isfile(config['CONFIG']['loss_file_path']):
        model.load_state_dict(torch.load('model_weights.h5'))

    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    mse = nn.MSELoss()

    t = load_obj('time')  # resume from the previous time step stored in file system
    while True:  # endless running

        optimizer.zero_grad()
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
                q = model(torch.Tensor(s_t))  # input a stack of 4 images,
                # get the prediction
                q = q.detach().numpy()
                max_Q = np.argmax(q)  # chosing index with maximum q value
                action_index = max_Q
                a_t[action_index] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > settings['last_epsilon'] and t > OBSERVE:
            epsilon -= (settings['first_epsilon'] -
                        settings['last_epsilon']) / \
                       settings['explore']

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate
        last_time = time.time()
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])  # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
        # append the new image to input stack and remove the first one

        # store the transition in D
        replay_memory.append((s_t, action_index, r_t, s_t1, terminal))
        if len(replay_memory) > settings['len_replay_memory']:
            replay_memory.popleft()

        # only train if done observing
        if t > OBSERVE:

            print('---> TRAINING')
            loss = 0
            # sample a minibatch to train on
            mini_batch = random.sample(replay_memory, settings['batch_size'])
            inputs = np.zeros((settings['batch_size'], s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], settings['num_actions']))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(mini_batch)):
                state_t = mini_batch[i][0]  # 4D stack of images
                action_t = mini_batch[i][1]  # This is action index
                reward_t = mini_batch[i][2]  # reward at state_t due to action_t
                state_t1 = mini_batch[i][3]  # next state
                terminal = mini_batch[i][4]  # whether the agent died or survided due the action

                inputs[i:i + 1] = state_t
                targets[i] = model(torch.Tensor(state_t)).detach().numpy()  # predicted q values
                Q_sa = model(torch.Tensor(state_t1)).detach().numpy()  # predict q values for next

                targets[i, action_t] = reward_t if terminal else reward_t + settings[
                    'gamma'] * np.max(Q_sa)

            predictions = model(torch.Tensor(inputs))

            mse_loss = mse(predictions, torch.Tensor(targets))
            mse_loss.backward()
            optimizer.step()

            loss += float(mse_loss.data)

            game_state.update_loss(loss)
            game_state.update_q_values(np.max(Q_sa))

        s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            game_state._game.pause()  # pause game while saving to filesystem
            save_obj(replay_memory, 'replay_memory')  # saving episodes
            save_obj(t, 'time')  # caching time steps
            save_obj(epsilon, 'epsilon')  # cache epsilon to avoid repeated randomness in actions
            game_state.save()
            torch.save(model.state_dict(), 'model_weights.h5')
            game_state._game.resume()

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)
