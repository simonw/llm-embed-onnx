import llm


def test_gte_tiny():
    model = llm.get_embedding_model("onnx-gte-tiny")
    hello1 = model.embed("hello")
    hello2 = model.embed("hello")
    assert hello1 == hello2
    assert len(hello1) == 384


def test_multi_gte_tiny():
    model = llm.get_embedding_model("onnx-gte-tiny")

    def generate():
        yield "hello"
        yield "hello"

    two_embeds = model.embed_batch(generate())
    assert len(two_embeds) == 2
    assert len(two_embeds[0]) == 384
    assert two_embeds[0] == two_embeds[1]
