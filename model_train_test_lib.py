import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import compute_indicators_labels_lib
import imbalanced_lib
from NNModel_lib import NNModel
import tensorflow as tf
from sklearn.utils import shuffle
import random

from config import RUN as run_conf
from numpy.random import seed
from tensorflow import random as tf_rand
from imbalanced_lib import get_sampler


def train_test(RUN, save_to="model.keras"):
    random.seed(RUN['seed'])
    seed(RUN['seed'] + 254923845)
    tf_rand.set_seed(RUN['seed'] + 984573)
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    scaler = StandardScaler()
    sampler = get_sampler(run_conf['balance_algo'])
    data = compute_indicators_labels_lib.get_dataset(RUN)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[
        (data['Date'] < RUN['back_test_start']) |
        (data['Date'] > RUN['back_test_end'])]  # exclude backtest data from trainig/test set

    data = data[data['pct_change'] < RUN['beta']]  # remove outliers

    print(data['label'].value_counts())

    labels = data['label'].copy()
    data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', "Asset_name", "label"], inplace=True)
    columns = data.columns
    index = data.index
    X = scaler.fit_transform(data.values)

    data = pd.DataFrame(X, columns=columns, index=index)
    data['label'] = labels
    # data.dropna(inplace=True)

    data = shuffle(data, random_state=RUN['seed'] + 3434534)
    data = sampler(data)

    Xs, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, stratify=y,
                                                        random_state=RUN['seed'] + 5467458)
    
    print(len(X_train))
    model = NNModel(X_train.shape[1], 3, imbalanced_lib.get_loss(RUN['loss_func']), epochs=RUN['epochs'])
    model.train(X_train, y_train)
    model.save(save_to)
    model.dummy_train(X_train, y_train)
    
    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)
    preds_dummy = model.dummy_predict(X_test)
    dummy_rep = classification_report(y_test, preds_dummy, digits=3, output_dict=True)
    train_rep = classification_report(y_train, preds_train, digits=2, output_dict=True)
    test_rep = classification_report(y_test, preds_test, digits=2, output_dict=True)
    

    rep_fields = ['', 'bw', 'fw', 'tf', 'alpha', 'beta', 'fee', 'epchs', 'bt_start', 'bt_end', 'ba_alg', 'acc', 'wa_prec', 'wa_rec', 'wa_f1',
                  '-1 prec', '0 prec', '1 prec', '-1 rec', '0 rec', '1 rec', '-1 f1', '0 f1', '1 f1', '-1 supp', '0 supp', '1 supp']
    rep_rows = []
    

    ds = {"TestRep": test_rep, "TrainRep": train_rep, "DummyRep": dummy_rep}
    for d in ds:
        rep = ds[d]
        row = [d, RUN['b_window'], RUN['f_window'], RUN['folder'], RUN['alpha'], RUN['beta'], RUN['commission fee'], RUN['epochs'], 
               RUN['back_test_start'], RUN['back_test_end'], RUN['balance_algo'],
               rep['accuracy'], rep['weighted avg']['precision'], rep['weighted avg']['recall'], rep['weighted avg']['f1-score']]
        for k in ['precision', 'recall', 'f1-score', 'support']:
            row.append(rep[str(-1)][k])
            row.append(rep[str(0)][k])
            row.append(rep[str(1)][k])

        rep_rows.append(row)
    
    return rep_rows, rep_fields
    

if __name__ == '__main__':
    train_test(run_conf)
