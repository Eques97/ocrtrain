from pathlib import Path
import tarfile
import json

from tensorflow.io import read_file, decode_jpeg
from tensorflow.image import resize_with_pad, transpose, per_image_standardization
from tensorflow.data import Dataset

from train import (
    BATCH_SIZE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    COLOR_CHANNEL,
    DATA_FILE,
)

def extract() -> Path:
    datapath = Path.cwd() / DATA_FILE
    if datapath.exists():
        return datapath
    tardir = str(datapath) + ".tar.xz"
    with tarfile.open(tardir) as f:
        f.extractall(Path.cwd())
    return datapath

def processimg(filepath: str):
    image = read_file(filepath)
    image = decode_jpeg(image, channels=COLOR_CHANNEL)
    image = resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image = transpose(image)
    return per_image_standardization(image)

def createds(isval=False) -> Dataset:
    datapath = extract()
    filename = "datasetv.json" if isval else "dataset.json"
    with open(datapath / filename) as f:
        ds = json.load(f)
    dsize = len(ds[0])
    cutidx = dsize % BATCH_SIZE
    if cutidx > 0:
        ds = [ds[0][:-cutidx], ds[1][:-cutidx]]
    paths = [str(datapath / x) for x in ds[0]]
    ds = Dataset.from_tensor_slices((paths, ds[1]))
    ds = ds.map(lambda x, y: (processimg(x), y))
    return ds.batch(BATCH_SIZE)
