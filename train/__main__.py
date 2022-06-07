import json

from tensorflow.keras.optimizers import RMSprop
from tensorflow.io import read_file, decode_jpeg
from tensorflow.image import resize_with_pad, transpose, per_image_standardization
from tensorflow.data import Dataset

from train.model import createModel
from train.cost import ctcloss
from train import (
    BATCH_SIZE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    COLOR_CHANNEL,
    ALPHA,
    EPOCHS,
    CHAR_COUNT,
)

def processimg(filepath: str):
    image = read_file(filepath)
    image = decode_jpeg(image, channels=COLOR_CHANNEL)
    image = resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image = transpose(image)
    return per_image_standardization(image)

def createds(ds) -> Dataset:
    ds = Dataset.from_tensor_slices(tuple(ds))
    ds = ds.map(lambda x, y: (processimg(x), y))
    return ds.batch(BATCH_SIZE)

if __name__ == "__main__":
    model = createModel(CHAR_COUNT)
    model.compile(
        optimizer=RMSprop(learning_rate=ALPHA),
        loss=ctcloss,
    )
    model.summary()

    with open("dataset.json") as f:
        ds = json.load(f)
    print(f"samples count: {len(ds[0])}")

    tds = [ds[0][:1000], ds[1][:1000]]
    print(f"train samples count: {len(tds[0])}")
    tds = createds(tds)

    vds = [ds[0][-1000:], ds[1][-1000:]]
    print(f"validate samples count: {len(vds[0])}")
    vds = createds(vds)

    model.fit(
        x=tds,
        epochs=EPOCHS,
        validation_data=vds
    )
