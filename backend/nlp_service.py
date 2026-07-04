import os
import sys

_MODEL_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "model_engine")
if _MODEL_ENGINE_PATH not in sys.path:
    sys.path.insert(0, _MODEL_ENGINE_PATH)

from pipeline import TextProcessingPipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TextProcessingPipeline()
    return _pipeline


def process_text(raw_text):
    return get_pipeline().process(raw_text)
