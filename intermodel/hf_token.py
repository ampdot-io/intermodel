import typing

if typing.TYPE_CHECKING:
    from tokenizers import Tokenizer

hf_tokenizers = {}


def get_hf_tokenizer(hf_name) -> "Tokenizer":
    if hf_name in hf_tokenizers:
        return hf_tokenizers[hf_name]
    else:
        from tokenizers import Tokenizer

        hf_tokenizers[hf_name] = Tokenizer.from_pretrained(
            hf_name, token=get_hf_auth_token()
        )  # log in with "huggingface-cli login"
        return hf_tokenizers[hf_name]


def get_hf_auth_token():
    import huggingface_hub

    try:
        get_token = huggingface_hub.get_token
    except AttributeError:
        from intermodel.hf_auth_token import get_token
    return get_token()
