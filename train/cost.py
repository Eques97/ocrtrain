from tensorflow import constant
from tensorflow.keras.backend import ctc_batch_cost

from train import BATCH_SIZE, FRAME_SIZE

def ctcloss(y_true, y_pred):
    input_length = constant(FRAME_SIZE, shape = (BATCH_SIZE, 1))
    label_length = constant(FRAME_SIZE, shape = (BATCH_SIZE, 1))
    loss_batch = ctc_batch_cost(
        y_true = y_true,
        y_pred = y_pred,
        input_length = input_length,
        label_length = label_length)
    return loss_batch
