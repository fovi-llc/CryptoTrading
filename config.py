from pandas import Timestamp

run1 = {
    'folder': 'raw_data_4_hour/',
    # 'folder' : 'raw_data_1_hour/',
    # 'folder' : 'raw_data_30_min/',
    #'folder' : 'raw_data_1_day/',

    'models': 'models/',

    'reports': 'reports/',
    'alpha': 0.038,  # computed in determine_alpha.py
    'beta': 0.24,  # ignore sample greater than beta in percent of change
    'seed': 353598215,
    'commission fee': 0.001,  # 0.0004,  # 0.001,

    'b_window': 5,
    'f_window': 2,

    # used in define the grid for searching backward and forward window
    'b_lim_sup_window': 6,
    'f_lim_sup_window': 6,

    'back_test_start': Timestamp("2022-01-1"),
    'back_test_end': Timestamp("2022-03-31"),
    'suffix': 'ncr',

    'stop_loss': 0.05,

    'off_label_set': [],  # ['BTCUSDT', 'ETHUSDT', 'ALGOUSDT']  # list of coin to be excluded from training/test set. Used in backtesting

    'balance_algo': 'srs',  # 'ncr', 'srs', None
    'loss_func': 'categorical',  # 'focal', 'categorical'

    'epochs': 32  # how many epochs spent in training neural network
}

RUN = run1
