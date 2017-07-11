import os
import json
import numpy as np

from gbdx_task_interface import GbdxTaskInterface
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


class TrainCnn(GbdxTaskInterface):

    def invoke(self):

        # Get string inputs
        nb_epoch = int(self.get_input_string_port('nb_epoch', default = '10'))
        bit_depth = int(self.get_input_string_port('bit_depth', default = '8'))

        # Get training from input data dir
        train = self.get_input_data_port('train_data')
        X_train = np.load(os.path.join(train, 'X.npz'))['arr_0']
        y_train = np.load(os.path.join(train, 'y.npz'))['arr_0']
        nb_classes = len(np.unique(y_train))

        # Reshape for input to net, normalize based on bit_depth
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_train = X_train.astype('float32')
        X_train /= float((2 ** bit_depth) - 1)
	X_train = np.swapaxes(X_train, 1, -1)	

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)

        # Create basic Keras model
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='valid',
                                input_shape=(X_train.shape[1:])))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        # Fit model on input data
        model.fit(X_train, Y_train, batch_size=128, nb_epoch=nb_epoch,
              verbose=1)

        # Create the output directory
        output_dir = self.get_output_data_port('trained_model')
        os.makedirs(output_dir)

        # Save the model architecture and weights to output dir
        model.save(os.path.join(output_dir, 'model.h5'))


if __name__ == '__main__':
    with TrainCnn() as task:
        task.invoke()
