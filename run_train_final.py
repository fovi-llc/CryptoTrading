from compute_indicators_labels_lib import preprocess
from model_train_test_lib import train_test
from config import RUN
from copy import deepcopy
from collections import defaultdict
from pandas import Timestamp
import pandas as pd


# train final models for backtesting and charting

run_conf = deepcopy(RUN)

run_conf['suffix'] = "bt_1"
run_conf['epochs'] = 50
run_conf['off_label_set'] = ['BTCUSDT', 'ETHUSDT', 'ALGOUSDT']

# train model on whole dataset excluding coins above putting start in the future , end in the past
# this avoid removing backtest data from the dataset
run_conf['back_test_start'] = Timestamp("2025-01-01")  # in the future
run_conf['back_test_end'] = Timestamp("2015-01-01")  # in the past


# here the backward and forward windows combination to train
top_most = [(2, 2), (2, 1), (4, 2), (5, 2), (3, 2), (5, 1)]



for bf in top_most:
    run_conf['b_window'] = bf[0]
    run_conf['f_window'] = bf[1]
    rep_rows, rep_fields = train_test(run_conf, "model_final_%d_%d.h5" % (run_conf['b_window'], run_conf['f_window']))
    df = pd.DataFrame(rep_rows, columns=rep_fields)
    df.to_excel("reports/final_model_%d_%d.xlsx" % (run_conf['b_window'], run_conf['f_window']))




