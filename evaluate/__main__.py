from train.model import createmodel
from train.dataset import createds
from train.cost import ctcloss
from train.metrics import characc

if __name__ == "__main__":

    ds = createds(isval=True)

    model = createmodel(decode=False)
    metrics = [
        characc,
    ]
    model.compile(
        loss=ctcloss,
        metrics=metrics,
    )
    model.load_weights("ckpt/008")
    model.evaluate(ds)
