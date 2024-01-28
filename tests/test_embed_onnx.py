import llm


def test_gte_tiny():
    model = llm.get_embedding_model("onnx-gte-tiny")
    hello1 = model.embed("hello")
    hello2 = model.embed("hello")
    assert hello1 == hello2
    assert len(hello1) == 384
