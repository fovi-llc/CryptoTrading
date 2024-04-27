import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.dummy import DummyClassifier


class NNModel:

    def __init__(self, in_dim, n_classes, loss='categorical_crossentropy', epochs=32):
        self.dummy_clf = DummyClassifier(strategy="stratified", random_state=2987)
        model = Sequential()
        model.add(Dense(128, activation=keras.layers.LeakyReLU(alpha=0.01), input_dim=in_dim))
        model.add(Dense(64, activation=keras.layers.LeakyReLU(alpha=0.01)))
        model.add(Dense(32, activation=keras.layers.LeakyReLU(alpha=0.01)))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam',
                      loss=[loss],
                      metrics=[keras.metrics.CategoricalAccuracy()])
        self.model = model
        self.epochs = epochs

    def train(self, train_data, y):
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        dummy_y = np_utils.to_categorical(encoded_Y)
        checkpoint = ModelCheckpoint(
            'model_best_weights.h5',
            monitor='loss',
            verbose=4,
            save_best_only=True,
            mode='min'
        )
        self.model.fit(train_data, dummy_y, batch_size=64, epochs=self.epochs, use_multiprocessing=True, verbose=1,
                       callbacks=[checkpoint])
        return True

    def predict(self, pred_data):
        y = self.model.predict(pred_data, verbose=0)
        max_data = np.argmax(y, axis=1)
        return max_data - 1

    def load(self, filename):
        self.model = keras.models.load_model(filename)

    def save(self, filename):
        self.model.save(filename)

    def dummy_train(self, train_data, y):
        self.dummy_clf.fit(train_data, y)

    def dummy_predict(self, pred_data):
        preds = self.dummy_clf.predict(pred_data)
        return preds
