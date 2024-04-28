from backtest_coins_lib import backtest_single_coin
from config import RUN
from copy import deepcopy
from collections import defaultdict
from pandas import Timestamp
import pandas as pd


run_conf = deepcopy(RUN)
final_rep = defaultdict(list)

# remember to set fee in config.py

run_conf['off_label_set'] = ['BTCUSDT', 'ETHUSDT', 'ALGOUSDT']
# period = "short"  # "long"


# run_conf['back_test_start'] = Timestamp("2022-05-15")
# run_conf['back_test_end'] = Timestamp("2022-12-31")
# run_conf['back_test_start'] = Timestamp("2015-05-15")
# run_conf['back_test_end'] = Timestamp("2025-12-31")

# here the backward and forward windows combination to train
top_most = [(2, 2), (2, 1), (4, 2), (5, 2), (3, 2), (5, 1)]

report = []
columns = ["model", "period", "bw", "fw", "coin", "final cap", "num op", "min drawdown ", "max gain ", "good ops"]

for period in ("short", "long"):
    for bf in top_most:
        run_conf['b_window'] = bf[0]
        run_conf['f_window'] = bf[1]
        
        # set backtest period, short for last six month of 2022, long for all data
        if period == "short":
            run_conf['back_test_start'] = Timestamp("2022-05-15")
            run_conf['back_test_end'] = Timestamp("2022-12-31")
        elif period == "long":
            run_conf['back_test_start'] = Timestamp("2015-05-15")
            run_conf['back_test_end'] = Timestamp("2025-12-31")
        
        for c in run_conf['off_label_set']:
            res = backtest_single_coin(run_conf, "%s.csv" % c, 
                                       mdl_name="model_final_%d_%d.keras" % (run_conf['b_window'], run_conf['f_window']), 
                                       suffix=period)
            
            rep = ["NN", period, bf[0], bf[1], c, res['nn'][0], res['nn'][1], res['nn'][2], res['nn'][3], res['nn'][4]]
            report.append(rep)
            rep = ["DU", period, bf[0], bf[1], c, res['du'][0], res['du'][1], res['du'][2], res['du'][3], res['du'][4]]
            report.append(rep)


rep = pd.DataFrame(report, columns=columns)

rep.to_excel(run_conf['reports'] + "backtest_final.xlsx")


    
    
