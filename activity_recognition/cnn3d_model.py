# 3d cnn model
import tensorflow as tf
from keras import models, Model
from keras.layers import Add, Activation, Conv2D, Conv3D, Concatenate, Dense, Dropout, Flatten, Input, LSTM, MaxPooling3D, GlobalMaxPool3D, GlobalMaxPool2D, ConvLSTM2D, ReLU, TimeDistributed

def _conv3d(in_tensor, kernels, k, s, pool=True):
    conv = Conv3D(kernels, kernel_size=k, strides=s, padding='same')(in_tensor)
    activated =  ReLU()(conv)
    if pool:
        return MaxPooling3D(pool_size=(4,4,2), strides=2, padding='same')(activated)
    else:
        return activated

def _model_architecture(in_shape, n_classes):
    in_layer = Input(in_shape)
    # stream 1 - temporal
    convx_1 = _conv3d(in_layer,32,(5,5,5),(1,2,2), True)
    convx_2 = _conv3d(convx_1, 64,(3,3,3),(1,1,1))
    convx_3 = _conv3d(convx_2,64,(3,3,3),(1,2,2))
    convx_4 = _conv3d(convx_3,128,(3,3,3),(1,1,1), True)
    convx_5 = _conv3d(convx_4,128,(3,3,3),(1,1,1))
    pool = GlobalMaxPool3D()(convx_5)
    output_temporal = Dense(n_classes*4, activation='relu')(pool)
    # stream 2 - lstm
    time_1 = TimeDistributed(Conv2D(24, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))(in_layer)
    lstm_1 = ConvLSTM2D(24, kernel_size=(3,3), strides=2, padding='same', activation='relu', return_sequences=False)(time_1)
    flatten_1 = Flatten()(lstm_1)
    output_lstm = Dense(n_classes*4, activation='relu')(flatten_1)
    # aggregation
    summation = Add()([output_temporal, output_lstm])
    # output layer
    output = Dense(n_classes, activation='softmax')(summation)
    model = Model(in_layer, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def _basic_architecture(in_shape, n_classes):
    in_layer = Input(in_shape)
    convx_1 = _conv3d(in_layer,32,5,2)
    pool_1 = GlobalMaxPool3D()(convx_1)
    output = Dense(n_classes, activation='softmax')(pool_1)
    model = Model(in_layer, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def model3d(in_shape=(16,192,256,3), n_classes=50):
    return _model_architecture(in_shape, n_classes)

def basic_model(in_shape=(16, 192, 256, 3), n_classes=50):
    return _basic_architecture(in_shape, n_classes)