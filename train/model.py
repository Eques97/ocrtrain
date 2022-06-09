from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import (
        Conv2D,
        BatchNormalization,
        ReLU,
        MaxPool2D,
        Reshape,
        Bidirectional,
        LSTM,
        Softmax,
    )
from train import CHAR_COUNT

def createModel(decode: bool=False) -> Sequential:
    kernel_initializer = TruncatedNormal(stddev=0.1)
    model = Sequential(
        [
            # convolution layers
            # first layer
            Conv2D(
                name="01-1",
                filters=32,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
                input_shape=(128, 32, 1),
            ),
            BatchNormalization(name="01-2"),
            ReLU(name="01-3"),
            MaxPool2D(name="01-4", pool_size=(2, 2)),

            # second layer
            Conv2D(
                name="02-1",
                filters=64,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(name="02-2"),
            ReLU(name="02-3"),
            MaxPool2D(name="02-4", pool_size=(2, 2)),

            # third layer
            Conv2D(
                name="03-1",
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(name="03-2"),
            ReLU(name="03-3"),
            MaxPool2D(name="03-4", pool_size=(1, 2)),

            # fourth layer
            Conv2D(
                name="04-1",
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(name="04-2"),
            ReLU(name="04-3"),
            MaxPool2D(name="04-4", pool_size=(1, 2)),

            # fifth layer
            Conv2D(
                name="05-1",
                filters=256,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
            ),
            BatchNormalization(name="05-2"),
            ReLU(name="05-3"),
            MaxPool2D(name="05-4", pool_size=(1, 2)),

            # squeeze layer
            Reshape(name="06-1", target_shape=(-1, 256)),

            # bidirectional LSTM layers
            Bidirectional(
                name="07-1",
                layer=LSTM(units=256, return_sequences=True)
            ),
            Bidirectional(
                name="07-2",
                layer=LSTM(units=256, return_sequences=True)
            ),

            # expand layer
            Reshape(name="08-1", target_shape=(-1, 1, 512)),

            # atorus convolution layer
            Conv2D(
                name="09-1",
                filters=CHAR_COUNT,
                kernel_size=1,
                dilation_rate=1,
                padding="same",
                kernel_initializer=kernel_initializer,
            ),

            # squeeze layer
            Reshape(name="10-1", target_shape=(-1, CHAR_COUNT)),

            # softmax layer
            Softmax(name="11-1")
        ],
        name="OCR"
    )
    return model
