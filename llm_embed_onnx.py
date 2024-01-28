import llm
import logging
from onnx_embedding_models.model import ONNXEmbeddingModel
from onnx_embedding_models.registry import registry


# Disable warning about lack of PyTorch which is not relevant here
class SpecificMessageFilter(logging.Filter):
    def filter(self, record):
        return "None of PyTorch, TensorFlow" not in record.getMessage()


logger = logging.getLogger("transformers")
logger.addFilter(SpecificMessageFilter())


@llm.hookimpl
def register_embedding_models(register):
    for key in registry.keys():
        register(OnnxModel(key))


class OnnxModel(llm.EmbeddingModel):
    def __init__(self, key):
        self.key = key
        self.model_id = "onnx-" + key
        self._model = None

    def ensure_model_path(self):
        # Downloads model if it is not yet downloaded
        onnx_dir = llm.user_dir() / "llm_embed_onnx"
        onnx_dir.mkdir(exist_ok=True)
        model_path = onnx_dir / self.key
        model_path.mkdir(exist_ok=True)
        if not list(model_path.glob("*.onnx")):
            ONNXEmbeddingModel.download_from_registry(
                self.key, destination=str(model_path)
            )
        return model_path

    def get_model(self):
        if self._model is None:
            model_path = self.ensure_model_path()
            self._model = ONNXEmbeddingModel(
                onnx_path=str(model_path / "model.onnx"),
                tokenizer_path=str(model_path),
                max_length=registry[self.key]["max_length"],
                pooling_strategy=registry[self.key]["pooling_strategy"],
                normalize=True,  # Is this right?
            )
        return self._model

    def embed_batch(self, texts):
        return self.get_model().embed_batch(texts=list(texts))
