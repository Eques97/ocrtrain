from tensorflow import constant
from tensorflow.nn import ctc_loss

from train import BATCH_SIZE, FRAME_SIZE

def ctcloss(labels, logits):
    label_length = constant(FRAME_SIZE, shape=(BATCH_SIZE,))
    logit_length = constant(FRAME_SIZE, shape=(BATCH_SIZE,))
    loss = ctc_loss(
        labels,
        logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=None,
    )
    return loss
