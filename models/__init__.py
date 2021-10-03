from models.core import CoreForestock
from models.bbands import BBandsForestock


MODELS = {"bbands": BBandsForestock}


# TODO: raise not available model
def model_picker(model_name: str) -> CoreForestock:
    return MODELS[model_name]
