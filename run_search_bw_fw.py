from datetime import datetime
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


def search_bw_fw(run_conf) -> list:
    run_conf["run_id"] = f"{run_conf['suffix']}_{run_conf['b_window']}_{run_conf['f_window']}_{datetime.now(datetime.UTC)}"
    run_conf["model_path"] = f"{run_conf['models']}model_{run_conf['run_id']}.keras"
    report = train_test(run_conf)
    print(report)
    res_strat = backtest_all_coins(run_conf)
    print(res_strat.info())

    res_strat.drop(columns=["asset_name"], inplace=True)

    final_rep = defaultdict(list)
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
    row = ["stat"]
    row.extend(cols)
    row.extend([""] * (len(final_rep) - len(cols) - 1))

    for k, v in zip(rep_fields, row):
        final_rep[k].append(v)

        # strategy report values

    stats = {
        "min": min_d,
        "max": max_d,
        "mean": mean_d,
        "median": med_d,
        "stDev": std_d,
    }

    for r in stats:
        row = [r]
        row.extend(stats[r].tolist())
        row.extend([""] * (len(final_rep) - len(cols) - 1))

        for k, v in zip(rep_fields, row):
            final_rep[k].append(v)

            # add blank line
    for k in final_rep:
        final_rep[k].append("")
    
    return final_rep


def main(run_conf, repetitions=1):
    final_rep = defaultdict(list)

    for bw in range(1, run_conf["b_lim_sup_window"]):
        for fw in range(1, run_conf["f_lim_sup_window"]):
            run_conf = deepcopy(run_conf)

            run_conf["b_window"] = bw
            run_conf["f_window"] = fw
            for i in range(0, repetitions):
                search_report = search_bw_fw(run_conf)
                for k in search_report:
                    final_rep[k].extend(search_report[k])

            # add 2 blank lines
            for j in range(0, 2):
                for k in final_rep:
                    final_rep[k].append("")

            df = pd.DataFrame(final_rep)
            df.to_excel(
                f"{RUN['reports']}final_{RUN['suffix']}_1_part.xlsx", float_format="%.3f"
            )

    df = pd.DataFrame(final_rep)
    df.to_excel(f"{RUN['reports']}final_{RUN['suffix']}_1.xlsx", float_format="%.3f")


if __name__ == "__main__":
    main(RUN)
