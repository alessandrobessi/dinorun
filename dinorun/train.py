import logging
import random
import time

import torch
import torch.nn as nn

from .helpers import *

config = configparser.ConfigParser()
config.read('./config.ini')
c = config['CONFIG']

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler = logging.FileHandler(os.path.join(os.getcwd(), 'training.log'))
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


def train(model, game_state, observe):
    last_time = time.time()
    replay_memory = load_object('replay_memory')

    init_action = np.zeros(settings['num_actions'])
    init_action[0] = 1

    x_t, r_0, terminal = game_state.get_state(init_action)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    initial_state = s_t

    if observe:
        observation_time = 9999999
        epsilon = 0
    else:
        observation_time = settings['observation']
        epsilon = load_object('epsilon')

    if os.path.isfile(c['model_weights_file_path']):
        model.load_state_dict(torch.load(c['model_weights_file_path']))

    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    mse = nn.MSELoss()

    t = load_object('time')

    while True:

        optimizer.zero_grad()
        loss = 0
        q = 0
        a_index = 0
        a_t = np.zeros([settings['num_actions']])

        if t % settings['frames_per_action'] == 0:
            if random.random() <= epsilon:
                logging.info("RANDOM ACTION")
                a_index = random.randrange(settings['num_actions'])
                a_t[a_index] = 1
            else:
                logging.info("POLICY ACTION")
                q = model(torch.Tensor(s_t)).detach().numpy()
                a_index = np.argmax(q)
                a_t[a_index] = 1

        if epsilon > settings['last_epsilon'] and t > observation_time:
            epsilon -= (settings['first_epsilon'] - settings['last_epsilon']) / settings['explore']

        x_next, r_t, terminal = game_state.get_state(a_t)
        logging.info('fps: {}'.format(1 / (time.time() - last_time)))
        last_time = time.time()
        x_next = x_next.reshape(1, 1, x_next.shape[0], x_next.shape[1])
        s_next = np.append(x_next, s_t[:, :3, :, :], axis=1)

        replay_memory.append((s_t, a_index, r_t, s_next, terminal))
        if len(replay_memory) > settings['len_replay_memory']:
            replay_memory.popleft()

        if t > observation_time:

            logging.info("TRAINING")
            loss = 0

            mini_batch = random.sample(replay_memory, settings['batch_size'])
            inputs = np.zeros((settings['batch_size'], s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], settings['num_actions']))

            for i in range(0, len(mini_batch)):
                state = mini_batch[i][0]
                action = mini_batch[i][1]
                reward = mini_batch[i][2]
                state_next = mini_batch[i][3]
                terminal = mini_batch[i][4]

                inputs[i:i + 1] = state
                targets[i] = model(torch.Tensor(state)).detach().numpy()
                q = model(torch.Tensor(state_next)).detach().numpy()

                targets[i, action] = reward if terminal else reward + settings['gamma'] * np.max(q)

            predictions = model(torch.Tensor(inputs))

            mse_loss = mse(predictions, torch.Tensor(targets))
            mse_loss.backward()
            optimizer.step()

            loss += float(mse_loss.data)

            game_state.update_loss(loss)
            game_state.update_q_values(np.max(q))

        s_t = initial_state if terminal else s_next
        t = t + 1

        if t % 100 == 0:
            game_state.game.pause()
            save_object(replay_memory, 'replay_memory')
            save_object(t, 'time')
            save_object(epsilon, 'epsilon')
            game_state.save()
            torch.save(model.state_dict(), c['model_weights_file_path'])
            game_state.game.resume()

        logging.info("time {} | loss {}".format(t, loss))
