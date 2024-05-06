import pandas as pd
import numpy as np

import compute_indicators_labels_lib
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from sklearn.preprocessing import StandardScaler
from NNModel_lib import NNModel
from sklearn.model_selection import train_test_split
import os
import traceback
import sys
from config import RUN as run_conf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

"""
backtest strategy based on coins excluded from training and test set

"""

def calc_cum_ret_s3(x, stop_loss, fee, f_win):
    
    def correct_labels(labels):
        """
        Correct labels in a way that the one at the start of the forward window must be -1 
        the other, to the end of window must be 0
        :param labels: 
        :return: 
        """
        arr = np.zeros(len(labels))
        result = np.where(labels == -1)
        idxs = result[0]

        pos = 0
        try:
            for idx in idxs:
                if idx <= pos:
                    continue

                arr[idx] = -1
                pos = idx + f_win
        except Exception as ex:
            print(ex)

        arr[len(labels) - 1] = 0

        return arr
    
    x['label'] = correct_labels(x['label'])
    x['CloseP'] = x['Close'].shift(-1 * f_win)
    x['Ret'] = ((x['CloseP'] * (1 - fee)) -
                (x['Close'] * (1 + fee))) / \
               (x['Close'] * (1 + fee)) * -x['label']

    # history, capital, num_ops, min_drowdown, max_gain, good_ops

    return list[x['Ret']], np.prod(x[x['Ret'] != 0]['Ret'].dropna() + 1), len(x[x['label'] == -1]),\
           np.min(x[x['Ret'] != 0]['Ret'].dropna()), np.max(x[x['Ret'] != 0]['Ret'].dropna()), -1



def calc_cum_ret_s1(x, stop_loss, fee):
    """
    compute cumulative return strategy s1.
    compute the multiplication factor of original capital after n iteration,
    reinvesting the gains.
    It search for BUY order, execute them and close them when stop_loss hits happen or when reversal/SELL signal.
    Finds the pairs BUY/SELL orders and compute the cumulative return.
    If no BUY order is encountered, it does nothing.
    If no SELL order is encountered, once a BUY order was issued, a dummy SELL order of the position is issued at the end of the period.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss:
    :param fee: commission fee applied
    :return: history, capital, num_op, min_drowdown, max_gain
    """

    order_pending = 0
    price_start = 0
    price_end = 0
    capital = 1
    history = []
    labs = x['label'].values  # debug

    min_drowdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    for row in x.itertuples():

        # handle stop loss
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                order_pending = 0

                price_end = price_start * (1 - stop_loss)

                pct_chg = (price_end - price_start) / price_start
                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg
                
                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0

        history.append(capital)

        if row.label == BUY:
            if order_pending:
                continue

            else:
                order_pending = 1
                price_start = row.Close
                num_ops += 1
                continue

        if row.label == HOLD:
            continue

        if row.label == SELL:
            if order_pending:
                price_end = row.Close
                pct_chg = (price_end - price_start) / price_start
                if pct_chg > 0:
                    good_ops += 1

                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg

                if pct_chg > max_gain:
                    max_gain = pct_chg

                order_pending = 0
                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0
                continue

            else:
                continue

    # handle last candle
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)

        if pct_chg < min_drowdown:
            min_drowdown = pct_chg

        if pct_chg > max_gain:
            max_gain = pct_chg

        capital *= 1 + (((price_end * (1 - fee)) -
                         (price_start * (1 + fee))) /
                        (price_start * (1 + fee)))
    

    return history, capital, num_ops, min_drowdown, max_gain, good_ops



def calc_cum_ret_s2(x, stop_loss, fee):
    """
    compute cumulative return strategy s2.
    close issued BUY order if stop loss hit or if at end of forward window is HOLD or SELL label
    :param x: 
    :param stop_loss: 
    :param fee: 
    :return: 
    """

    order_pending = 0
    price_start = 0
    price_end = 0
    capital = 1
    history = []
    labs = x['label'].values  # debug

    min_drawdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    fw_pos = 0
    for row in x.itertuples():
        history.append(capital)
        # handle stop loss
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                order_pending = 0

                price_end = price_start * (1 - stop_loss)

                pct_chg = (price_end - price_start) / price_start
                if pct_chg < min_drawdown:
                    min_drawdown = pct_chg

                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0
                fw_pos = 0

        # handle fw window
        if order_pending:
            if fw_pos < run_conf['f_window']:
                fw_pos += 1
                continue
            elif fw_pos == run_conf['f_window']:
                if row.label == BUY:
                    fw_pos = 0
                    continue
        

        if row.label == BUY:
            if order_pending:
                fw_pos += 1
                continue

            else:
                order_pending = 1
                price_start = row.Close
                num_ops += 1
                fw_pos = 0
                continue


        if row.label == SELL or row.label == HOLD:
            if order_pending:
                price_end = row.Close
                pct_chg = (price_end - price_start) / price_start
                if pct_chg > 0:
                    good_ops += 1

                if pct_chg < min_drawdown:
                    min_drawdown = pct_chg

                if pct_chg > max_gain:
                    max_gain = pct_chg

                order_pending = 0
                fw_pos = 0
                capital *= 1 + (((price_end * (1 - fee)) -
                                 (price_start * (1 + fee))) /
                                (price_start * (1 + fee)))
                price_start = price_end = 0
                continue

            else:
                continue

    # handle last candle
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)

        if pct_chg < min_drawdown:
            min_drawdown = pct_chg

        if pct_chg > max_gain:
            max_gain = pct_chg

        capital *= 1 + (((price_end * (1 - fee)) -
                         (price_start * (1 + fee))) /
                        (price_start * (1 + fee)))

    return history, capital, num_ops, min_drawdown, max_gain, good_ops



def backtest_single_coin(RUN, filename, mdl_name="model.h5", suffix=""):
    """
    Backtest a coin whose timeseries is contained in filename.
    It uses last model trained.
    Backtest period selected in RUN config dictionary
    :param suffix: 
    :param mdl_name: 
    :param RUN: 
    :param filename: 
    :return: a dictionary with dummy (du) and neural net (nn) statistic of backtest
    tuple composed by: (final capital, num operaatioins completed, min drowdown, max_gain, positive ops)
    """

    data1 = compute_indicators_labels_lib.get_dataset(RUN)
    data1.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', "Asset_name", "Date"], inplace=True)
    data1.replace([np.inf, -np.inf], np.nan, inplace=True)
    data1.dropna(inplace=True)
    X_scaler = data1.iloc[:, :-1]
    nr = StandardScaler()
    nr.fit(X_scaler)
    X, y = data1.iloc[:, :-1], data1.iloc[:, -1]
    Xs = nr.transform(X)  # used for dummy classifier
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, stratify=y)  # , random_state=RUN['seed'])
    
    
    try:
        data = pd.read_csv(f"{RUN['folder']}{filename}")
        data['Date'] = pd.to_datetime(data['Date'])

        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)

        data = data[data['Date'] >= RUN['back_test_start']]
        data = data[data['Date'] <= RUN['back_test_end']]
        if len(data.index) == 0:
            raise ValueError("Void dataframe")
        
        data.set_index('Date', inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data_pred = data.copy()
        data_pred.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', "Asset_name"],
                       inplace=True)
        if len(data_pred.index) == 0:
            raise ValueError("Void dataframe")
        
        Xs = nr.transform(data_pred)
        model = NNModel(Xs.shape[1], 3)
        model.dummy_train(X_train, y_train)
        model.load(mdl_name)
        labels = model.predict(Xs)
        data['label'] = labels
        hist_nn, cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn = calc_cum_ret_s1(data, RUN['stop_loss'], RUN['commission fee'])
        
        
        labels = model.dummy_predict(Xs)
        data['label'] = labels
        hist_du, cap_du, num_op_du, min_drawdown_du, max_gain_du, g_ops_du = calc_cum_ret_s1(data, RUN['stop_loss'], RUN['commission fee'])

        # hist_du.pop()

        dates = list(data.index)
        
        ya = np.array(hist_nn)
        ya = np.log(ya)
        # ya = (ya - ya.mean()) / ya.std()
        # ya = ya - ya[0]
        
        yda = np.array(hist_du)
        yda = np.log(yda)
        # yda = (yda - yda.mean()) / yda.std()
        # yda = yda - yda[0]
        
        prices = data['Close']
        
        # prices = np.array(list((data['Close'] - data['Close'].mean()) / data['Close'].std()))
        # prices = prices - prices[0]

        plt.rcParams['font.size'] = 14
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        fig, axs = plt.subplots(2, 1)
        fig = plt.gcf()
        fig.set_size_inches(8, 8)

        f = RUN['f_window']
        b = RUN['b_window']
        
        ax = axs[0]
        ax.set_facecolor('#eeeeee')
        box = ax.get_position()
        box.y0 = box.y0 + 0.03
        ax.set_position(box)
        ax.plot(dates, ya, label='MLP', color='green')
        ax.plot(dates, yda, label='Dummy', color='red')
        ax.set(xlabel='', ylabel='Log(Return)',
               title=filename.split('.')[0] + " " + "backW=%d, forW=%d" % (b, f))
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.grid()
        ax.legend()
        
        ax = axs[1]
        ax.set_facecolor('#eeeeee')
        ax.plot(dates, prices)
        ax.set(xlabel='', ylabel='Price',
               title="")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.grid()

        fig.savefig(RUN["reports"] + filename.split('.')[0] + "_b%d_f%d_%s.png" % (b, f, suffix))
        plt.show()
        
        return {'du': (cap_du, num_op_du, min_drawdown_du, max_gain_du, g_ops_du),
                'nn': (cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn)}

    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)

    

def backtest_all_coins(RUN):
    data1 = compute_indicators_labels_lib.get_dataset(RUN)
    data1.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', "Asset_name", "Date"], inplace=True)
    data1.replace([np.inf, -np.inf], np.nan, inplace=True)
    data1.dropna(inplace=True)
    X_scaler = data1.iloc[:, :-1]
    nr = StandardScaler()
    nr.fit(X_scaler)
    X, y = data1.iloc[:, :-1], data1.iloc[:, -1]
    Xs = nr.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, stratify=y)  # , random_state=RUN['seed'])

    filenames = os.listdir(RUN['folder'])
    rets_list = []

    start = ""
    end = ""
    for i in range(len(filenames)):
        try:
            print(f"{round((i / len(filenames)) * 1000) / 10}% of files analyzed")
            data = pd.read_csv(f"{RUN['folder']}{filenames[i]}")
            data['Date'] = pd.to_datetime(data['Date'])

            start = data['Date'].iloc[0].strftime("%d-%m-%Y")
            end = data['Date'].iloc[-1].strftime("%d-%m-%Y")
            data = TecnicalAnalysis.compute_oscillators(data)
            data = TecnicalAnalysis.find_patterns(data)
            data = TecnicalAnalysis.add_timely_data(data)

            data = data[data['Date'] >= RUN['back_test_start']]
            data = data[data['Date'] <= RUN['back_test_end']]
            if len(data.index) == 0:
                continue
            
            data.set_index('Date', inplace=True)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)
            data_pred = data.copy()
            data_pred.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', "Asset_name"], inplace=True)
            if len(data_pred.index) == 0:
                continue
            Xs = nr.transform(data_pred)
            model = NNModel(Xs.shape[1], 3)
            model.dummy_train(X_train, y_train)
            model.load('model.h5')
            labels = model.predict(Xs)
            data['label'] = labels
            hist_nn, cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn = calc_cum_ret_s1(data, RUN['stop_loss'],
                                                                                                 RUN['commission fee'])
                        
            labels = model.dummy_predict(Xs)
            data['label'] = labels
            hist_du, cap_du, num_op_du, min_drawdown_du, max_gain_du, g_ops_du = calc_cum_ret_s1(data, RUN['stop_loss'],
                                                                                                 RUN['commission fee'])
            
            rets_list.append(
                {'asset_name': data['Asset_name'].iloc[0], 'profit_nn': cap_nn, 'profit_dummy': cap_du,
                 'operations': num_op_nn, 'max_dropdown': min_drawdown_nn, 'max_gain': max_gain_nn})

        except Exception:
            print("Exception in user code:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)

    fw = RUN['f_window']
    bw = RUN['b_window']

    res_df = pd.DataFrame(rets_list)

    f_save = ""
    for i in range(1, 20):
        f_save = f'{RUN["reports"]}{start}_{end}_{bw}_{fw}_{RUN["suffix"]}_{i}.xlsx'
        if os.path.isfile(f_save):
            continue
        else:
            break


    res_df.to_excel(f_save, float_format="%.3f")

    return res_df



if __name__ == "__main__":
    backtest_single_coin(run_conf, 'BTCUSDT.csv')
