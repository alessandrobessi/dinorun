settings = {
    'num_actions': 2,
    'gamma': 0.99,
    'observation': 100,  # time steps to observe before training
    'explore': 100000,  # frames over which anneal epsilon
    'last_epsilon': 1e-4,
    'first_epsilon': 0.1,
    'len_replay_memory': 50000,  # number of previous transitions to remember
    'batch_size': 16,
    'frames_per_action': 1,
    'learning_rate': 1e-4,
    'img_rows': 40,
    'img_cols': 40,
    'img_channels': 4,
    'reward': 0.1
}
