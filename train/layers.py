from tensorflow.keras.layers import Layer
from tensorflow.sparse import to_dense, SparseTensor
from tensorflow import constant, int64

from train import (
    BATCH_SIZE,
    FRAME_SIZE,
    CHAR_COUNT,
)
from evaluate.decode import ctcdecode

class CTCdecode(Layer):
    def __init__(self, name='ctc_greedy_decoder'):
        super(CTCdecode, self).__init__()
        self._name = name

    def call(self, inputs):
        sparse = ctcdecode(inputs)
        dense_shape = constant([BATCH_SIZE, FRAME_SIZE], dtype=int64)
        sparse = SparseTensor(
            indices=sparse.indices,
            values=sparse.values,
            dense_shape=dense_shape,
        )
        return to_dense(sparse, default_value=CHAR_COUNT - 1)
