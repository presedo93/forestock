from models.core import CoreForestock
from models.bbands import BBandsForestock
from models.ohlc import OHLCForestock
from models.emas import EmasForestock


MODELS = {"bbands": BBandsForestock, "ohlc": OHLCForestock, "emas": EmasForestock}


# TODO: raise not available model
def model_picker(model_name: str) -> CoreForestock:
    return MODELS[model_name.lower()]
