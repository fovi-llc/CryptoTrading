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

from config import RUN
from numpy.random import seed
from tensorflow import random as tf_rand
from imbalanced_lib import get_sampler


def train_test(run_conf):
    random.seed(run_conf["seed"])
    seed(run_conf["seed"] + 254923845)
    tf_rand.set_seed(run_conf["seed"] + 984573)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    scaler = StandardScaler()
    sampler = get_sampler(run_conf["balance_algo"])
    data = compute_indicators_labels_lib.get_dataset(run_conf)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[
        (data["Date"] < run_conf["back_test_start"])
        | (data["Date"] > run_conf["back_test_end"])
    ]  # exclude backtest data from training/test set

    data = data[data["pct_change"] < run_conf["beta"]]  # remove outliers

    print(data["label"].value_counts())

    labels = data["label"].copy()
    data.drop(
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Asset_name",
            "label",
        ],
        inplace=True,
    )
    columns = data.columns
    index = data.index
    X = scaler.fit_transform(data.values)

    data = pd.DataFrame(X, columns=columns, index=index)
    data["label"] = labels
    # data.dropna(inplace=True)

    data = shuffle(data, random_state=run_conf["seed"] + 3434534)
    data = sampler(data)

    Xs, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, stratify=y, random_state=run_conf["seed"] + 5467458
    )

    print(len(X_train))
    model = NNModel(
        X_train.shape[1],
        3,
        imbalanced_lib.get_loss(run_conf["loss_func"]),
        epochs=run_conf["epochs"],
    )
    model.train(X_train, y_train)
    model.save(run_conf["model_path"])
    model.dummy_train(X_train, y_train)

    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)
    preds_dummy = model.dummy_predict(X_test)
    dummy_rep = classification_report(y_test, preds_dummy, digits=3, output_dict=True)
    train_rep = classification_report(y_train, preds_train, digits=2, output_dict=True)
    test_rep = classification_report(y_test, preds_test, digits=2, output_dict=True)

    report = pd.DataFrame(
        # columns=[
        #     "",
        #     "run",
        #     "bw",
        #     "fw",
        #     "tf",
        #     "alpha",
        #     "beta",
        #     "fee",
        #     "epchs",
        #     "bt_start",
        #     "bt_end",
        #     "ba_alg",
        #     "acc",
        #     "wa_prec",
        #     "wa_rec",
        #     "wa_f1",
        #     "-1 prec",
        #     "0 prec",
        #     "1 prec",
        #     "-1 rec",
        #     "0 rec",
        #     "1 rec",
        #     "-1 f1",
        #     "0 f1",
        #     "1 f1",
        #     "-1 supp",
        #     "0 supp",
        #     "1 supp",
        # ]
    )

    for lab, rep in {
        "TestRep": test_rep,
        "TrainRep": train_rep,
        "DummyRep": dummy_rep,
    }.items():
        report = report.append(
            {
                "": lab,
                "run": run_conf["run_id"],
                "bw": run_conf["b_window"],
                "fw": run_conf["f_window"],
                "tf": run_conf["folder"],
                "alpha": run_conf["alpha"],
                "beta": run_conf["beta"],
                "fee": run_conf["commission fee"],
                "epchs": run_conf["epochs"],
                "bt_start": run_conf["back_test_start"],
                "bt_end": run_conf["back_test_end"],
                "ba_alg": run_conf["balance_algo"],
                "acc": rep["accuracy"],
                "wa_prec": rep["weighted avg"]["precision"],
                "wa_rec": rep["weighted avg"]["recall"],
                "wa_f1": rep["weighted avg"]["f1-score"],
                "-1 prec": rep[str(-1)]["precision"],
                "0 prec": rep[str(0)]["precision"],
                "1 prec": rep[str(1)]["precision"],
                "-1 rec": rep[str(-1)]["recall"],
                "0 rec": rep[str(0)]["recall"],
                "1 rec": rep[str(1)]["recall"],
                "-1 f1": rep[str(-1)]["f1-score"],
                "0 f1": rep[str(0)]["f1-score"],
                "1 f1": rep[str(1)]["f1-score"],
                "-1 supp": rep[str(-1)]["support"],
                "0 supp": rep[str(0)]["support"],
                "1 supp": rep[str(1)]["support"],
            }
        )

    return report


if __name__ == "__main__":
    print(train_test(RUN))
