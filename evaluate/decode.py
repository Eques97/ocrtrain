from tensorflow.nn import ctc_greedy_decoder
from tensorflow import transpose, constant

from train import (
    BATCH_SIZE,
    FRAME_SIZE,
)

def ctcdecode(inputs):
    inputs = transpose(inputs, [1, 0, 2])
    sequence_length = constant(FRAME_SIZE, shape=(BATCH_SIZE,))
    decoded, _ = ctc_greedy_decoder(inputs=inputs, sequence_length=sequence_length)
    sparse = decoded[0]
    return sparse
