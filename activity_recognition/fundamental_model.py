import tensorflow as tf
from tensorflow.keras.layers import add, multiply, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Add, Activation, Conv2D, Conv3D, Concatenate, Dense, Dropout, Flatten, Input, LSTM, MaxPooling3D, GlobalMaxPool3D, GlobalMaxPool2D, ConvLSTM2D, ReLU, TimeDistributed

class simple_model(tf.keras.Model):
    def __init__(self, k = 64, d=8, label_size=50, output_activation='softmax'):
        super(simple_model, self).__init__()
        self.conv1 = Conv3D(k, kernel_size=(7,7,7), strides=(2,2,2), padding='same', input_shape=(16, 192, 256, 3))
        self.conv2 = Conv3D(k, kernel_size=(3,3,3), strides=(2,2,2), padding='same')
        self.conv3 = Conv3D(k, kernel_size=(3,3,3), strides=(2,2,2), padding='same')
        self.classifier = Dense(label_size, activation=output_activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = MaxPooling3D(pool_size=(4,4,4), strides=2, padding='same')(x)
        x = self.conv3(x)
        x = GlobalMaxPool3D()(x)
        return self.classifier(x)