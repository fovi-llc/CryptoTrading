from compute_indicators_labels_lib import preprocess
from model_train_test_lib import train_test
from backtest_coins_lib import backtest_all_coins
from config import RUN
from copy import deepcopy
from collections import defaultdict
import numbers
import pandas as pd

"""
grid search forward and backward window
"""

run_conf = deepcopy(RUN)

final_rep = defaultdict(list)

for bw in range(1, run_conf['b_lim_sup_window']):
    for fw in range(1, run_conf['f_lim_sup_window']):
        run_conf['b_window'] = bw
        run_conf['f_window'] = fw
        for i in range(0, 3):
            
            rep_res, rep_fields = train_test(run_conf)
            res_strat = backtest_all_coins(run_conf)
            
            res_strat.drop(columns=['asset_name'], inplace=True)
            
            print(rep_res)
            
            for rep in rep_res:
                for k, v in zip(rep_fields, rep):
                    final_rep[k].append("%.3f" % v if isinstance(v, numbers.Number) else str(v))
            
            # add stats for strategy
            
            cols = list(res_strat.columns)
            
            min_d = res_strat.min(axis=0)
            max_d = res_strat.max(axis=0)
            med_d = res_strat.median(axis=0)
            mean_d = res_strat.mean(axis=0)
            std_d = res_strat.std(axis=0)
            
            # strategy report header
            row = ['stat']
            row.extend(cols)
            row.extend([""] * (len(final_rep) - len(cols) - 1))
            
            for k, v in zip(rep_fields, row):
                final_rep[k].append(v)
                
            # strategy report values
            
            stats = {'min': min_d, 'max': max_d, 'mean': mean_d, 'median': med_d, 'stDev': std_d}
            
            for r in stats:
                row = [r]
                row.extend(stats[r].tolist())
                row.extend([""] * (len(final_rep) - len(cols) - 1))
    
                for k, v in zip(rep_fields, row):
                    final_rep[k].append(v)
            
            # add blank line
            for k in final_rep:
                final_rep[k].append("")
                
        # add 2 blank lines
        for j in range(0, 2):
            for k in final_rep:
                final_rep[k].append("")

        df = pd.DataFrame(final_rep)
        df.to_excel(f"{RUN['reports']}final_{RUN['suffix']}_1_part.xlsx", float_format="%.3f")
        
        
    
df = pd.DataFrame(final_rep)
df.to_excel(f"{RUN['reports']}final_{RUN['suffix']}_1.xlsx", float_format="%.3f")
              
       
        
        
        
        
        
        
            
