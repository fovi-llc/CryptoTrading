import pandas as pd
import numpy as np
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf
import random
import tensorflow as tf
from compute_indicators_labels_lib import get_dataset
import shap
import pickle
from imbalanced_lib import get_sampler


samples = 50000

tf.keras.backend.clear_session()

random.seed(run_conf['seed'])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
scaler = StandardScaler()
data = get_dataset(run_conf)
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', "Asset_name"], inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

X, y = data.iloc[:, :-1], data.iloc[:, -1]

X = scaler.fit_transform(X, y)
data = pd.DataFrame(X, columns=data.columns[:-1])
data['label'] = y
data.dropna(inplace=True)

sampler = get_sampler(run_conf['balance_algo'])
data = sampler(data)
data = data.sample(n=samples, axis=0)

X_train, y = data.iloc[:, :-1], data.iloc[:, -1]

print(len(X))
model = NNModel(X_train.shape[1], len(y.unique()))
model.load("model_final_%d_%d.keras" % (run_conf['b_window'], run_conf['f_window']))

explainer = shap.Explainer(model.predict, masker=X_train, algorithm='permutation', feature_names=data.columns)
ex = explainer(X_train)
print(ex)
shap.plots.beeswarm(ex, max_display=10)
shap.plots.bar(ex, max_display=10)

with open("explainer_%d_%d_%d.pickle" % (run_conf['b_window'], run_conf['f_window'], samples), 'wb') as handle:
    pickle.dump(ex, handle, protocol=pickle.HIGHEST_PROTOCOL)
