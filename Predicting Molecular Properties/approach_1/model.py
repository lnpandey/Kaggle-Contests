# import matplotlib.pyplot as plt
import keras as K
# import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate
from keras.models import Model
# from keras import optimizers
# from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from code import *

inp_dim = len(X.columns)
mid_dim = len(X_mid.columns)
out_dim = len(Y_mid.columns)
inp = Input(shape=[inp_dim], dtype='float')
nn = BatchNormalization()(inp)
nn = Dense(50, activation='relu')(nn)
nn = Dropout(0.2)(nn)
nn = BatchNormalization()(nn)
nn = Dense(50, activation='relu')(nn)
nn = Dropout(0.2)(nn)
nn = BatchNormalization()(nn)
nn = Dense(40, activation='relu')(nn)
nn = Dropout(0.2)(nn)
nn = BatchNormalization()(nn)
out_mid = Dense(mid_dim, activation='linear', name='out_mid')(nn)
inp_mid = Input(shape=[mid_dim], dtype='float')
nn = Concatenate()([inp, inp_mid])
nn = BatchNormalization()(nn)
nn = Dense(80, activation='relu')(nn)
nn = Dropout(0.2)(nn)
nn = BatchNormalization()(nn)
nn = Dense(40, activation='relu')(nn)
nn = Dropout(0.2)(nn)
nn = BatchNormalization()(nn)
nn = Dense(20, activation='relu')(nn)
nn = BatchNormalization()(nn)
out = Dense(out_dim, activation='linear', name='out')(nn)
model1 = Model(inputs=[inp], outputs=[out_mid])
model2 = Model(inputs=[inp, inp_mid], outputs=[out_mid, out])


def loss(y_true, y_pred):
    return K.backend.mean(K.backend.log(K.losses.mean_squared_error(y_true, y_pred)))


losses = {'out_mid': 'mse', 'out': loss}
model1.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
model2.compile(optimizer='adam', loss=losses, metrics=["accuracy"])
model1.summary()
model2.summary()

# x_train, x_test, y_train, y_test = train_test_split(X, X_mid, test_size=0.1)
# checkpoint = ModelCheckpoint(f'{file_folder}/model3_1', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# hist1 = model1.fit([X], [X_mid], epochs=25, batch_size=512, validation_split=0.15, callbacks=[checkpoint])
# checkpoint = ModelCheckpoint(f'{file_folder}/model3', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# hist = model2.fit([X, X_mid], [X_mid, Y_mid], epochs=25, batch_size=512, validation_split=0.15, callbacks=[checkpoint])

X_test = X_test.fillna(X_test.mean())
# X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())
# model1.load_weights(f'{file_folder}/model3_1')
# model2.load_weights(f'{file_folder}/model3')



X_test_mid = model1.predict(X_test)
pred_y = model2.predict([X_test, X_test_mid])
y_pred = pred_y[1].sum(axis=1)
