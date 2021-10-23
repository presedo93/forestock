from typing import Any, Dict, KeysView
from models.core import CoreForestock
from models.bbands import BBandsForestock, BBANDS_DESC
from models.ohlc import OHLCForestock, OHLC_DESC
from models.emas import EmasForestock, EMA_DESC
from models.linears import LinearsForestock, LINEARS_DESC


MODELS: Dict[str, Dict[str, Any]] = {
    "bbands": {"model": BBandsForestock, "desc": BBANDS_DESC},
    "ohlc": {"model": OHLCForestock, "desc": OHLC_DESC},
    "emas": {"model": EmasForestock, "desc": EMA_DESC},
    "linears": {"model": LinearsForestock, "desc": LINEARS_DESC},
}


def available_models() -> KeysView[str]:
    return MODELS.keys()


# TODO: raise not available model
def model_picker(model_name: str) -> CoreForestock:
    return MODELS[model_name.lower()]["model"]


# TODO: raise not available model
def desc_picker(model_name: str) -> str:
    return MODELS[model_name.lower()]["desc"]
