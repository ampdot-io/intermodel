from intermodel.callgpt import parse_model_string, Model, max_token_length_inner


def test_plain():
    assert parse_model_string("davinci-002") == Model(
        "davinci-002", "davinci-002", max_token_length_inner("davinci-002")
    )


def test_both():
    assert parse_model_string("davinci-002^davinci@32000") == Model(
        "davinci-002", "davinci", 32000
    )


def test_tokenizer():
    assert parse_model_string("davinci-002^davinci") == Model(
        "davinci-002", "davinci", max_token_length_inner("davinci")
    )


def test_length():
    assert parse_model_string("davinci-002@12") == Model(
        "davinci-002", "davinci-002", 12
    )
