config = {
    # gomoku
    'n': 15,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts
    #'libtorch_use_gpu' : True,                  # libtorch use cuda
    #'num_mcts_threads': 4,                      # mcts threads number
    #'num_mcts_sims': 1600,                      # mcts simulation times
    #'c_puct': 5,                                # puct coeff
    #'c_virtual_loss': 3,                        # virtual loss coeff

    # neural_network
    'train_use_gpu': True,                     # train neural network using cuda
    'lr': 0.001,                                # learning rate
    'lr_milestones': [200, 400, 800],           # lr 阶梯衰减触发代际
    'lr_gamma': 0.5,                            # lr 每次衰减系数
    'l2': 0.0001,                               # L2
    'num_channels': 256,                        # convolution neural network channel size
    'num_layers' : 4,                           # residual layer number
    'epochs': 5,                                # train epochs
    'batch_size': 512,                          # batch size
    'input_channel_size':3,                            # board(cnn) input channel

    # train
    'num_iters': 10000,                         # train iterations
    'num_eps': 10,                              # self play times in per iter
    'num_train_threads': 10,                    # self play in parallel
    'num_explore': 5,                           # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.3,                     # action noise in self play games
    'update_threshold': 0.55,                   # update model threshold
    'num_contest': 10,                          # new/old model compare times
    'check_freq': 20,                           # test model frequency
    'examples_buffer_max_len': 20,              # 回放窗口(轮数):训练时使用最近 N 轮自对弈数据
    'games_per_iter': 16,                       # 每轮自对弈局数,须与 src/configuration.rs 的 NUM_2_SELF_PLAY
                                                # 及 train/train_loop.sh 的 STEP 保持一致

    # test
    'human_color': 1                            # human player's color
}

# action size
config['action_size'] = config['n'] ** 2
