from tensorflow.math import less, multiply
from tensorflow.sparse import from_dense, reset_shape
from tensorflow import edit_distance, cast, int64, where

from train import CHAR_COUNT
from evaluate.decode import ctcdecode

def characc(y_true, y_predict):
    hypothesis = ctcdecode(y_predict)
    y_true = cast(y_true, dtype=int64)
    condition = less(y_true, CHAR_COUNT - 1)
    mask = cast(where(condition, 1, 0), dtype=int64)
    y_true = multiply(y_true, mask)
    sparse = from_dense(y_true)
    truth = reset_shape(sparse)
    edist = edit_distance(
        hypothesis=hypothesis,
        truth=truth,
        normalize=True,
    )
    acc = 100 - edist * 100
    return acc
