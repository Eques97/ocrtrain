import json
from pathlib import Path

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
    DATA_PATH,
)

def processimg(filepath: str):
    image = read_file(filepath)
    image = decode_jpeg(image, channels=COLOR_CHANNEL)
    image = resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image = transpose(image)
    return per_image_standardization(image)

def createds(datapath: Path) -> Dataset:
    with open(datapath / "dataset.json") as f:
        ds = json.load(f)
    ds = [ds[0][:100], ds[1][:100]]
    paths = [str(datapath / x) for x in ds[0]]
    ds = Dataset.from_tensor_slices((paths, ds[1]))
    ds = ds.map(lambda x, y: (processimg(x), y))
    return ds.batch(BATCH_SIZE)

if __name__ == "__main__":

    # create model
    model = createModel(CHAR_COUNT)
    model.compile(
        optimizer=RMSprop(learning_rate=ALPHA),
        loss=ctcloss,
    )
    model.summary()

    # create dataset
    datapath = Path.cwd() / DATA_PATH
    ds = createds(datapath)

    # train model
    model.fit(
        x=ds,
        epochs=EPOCHS,
        # validation_data=vds
    )
