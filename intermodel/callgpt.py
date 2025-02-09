#!/usr/bin/env python
import dataclasses
import json
import re
import sys
import warnings
import traceback
import asyncio
import cmd
import datetime
import os
import uuid
import hashlib
from typing import Union, List, Optional, Iterable

import aiohttp
import tenacity
from huggingface_hub.utils import LocalEntryNotFoundError

import intermodel.callgpt_faker

from dotenv import load_dotenv

from intermodel.hf_token import get_hf_tokenizer, get_hf_auth_token

load_dotenv()

MODEL_ALIASES = {}
untokenizable = set()

session: Optional[aiohttp.ClientSession] = None


@tenacity.retry(
    retry=tenacity.retry_if_exception(
        lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status in (429,)
    ),
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_attempt(6),
)
async def complete(
    model,
    prompt=None,
    temperature=None,
    top_p=None,
    max_tokens=None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Union[float, int] = 0,
    presence_penalty: Union[float, int] = 0,
    num_completions: int = None,
    top_k=None,
    repetition_penalty: Union[float, int] = 1,
    tfs=1,
    user_id=None,
    logit_bias=None,
    vendor=None,
    vendor_config=None,
    **kwargs,
):
    tokenize_as = parse_model_string(MODEL_ALIASES.get(model, model)).tokenize_as
    model = parse_model_string(MODEL_ALIASES.get(model, model)).model
    # todo: multiple completions, top k, logit bias for all vendors
    # todo: detect model not found on all vendors and throw the same exception
    if vendor is None:
        vendor = pick_vendor(model, vendor_config)
    if vendor_config is not None and vendor in vendor_config:
        kwargs = {**vendor_config[vendor]["config"], **kwargs}
    if vendor.startswith("openai"):
        global session
        if session is None:
            session = aiohttp.ClientSession()
        if user_id is None:
            hashed_user_id = None
        else:
            hash_object = hashlib.sha256()
            hash_object.update(os.getenv("INTERMODEL_HASH_SALT", "").encode("utf-8"))
            hash_object.update(str(user_id).encode("utf-8"))
            hashed_user_id = hash_object.hexdigest()

        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        rest = dict(kwargs)
        headers = {
            "Content-Type": "application/json",
        }
        if (api_key := rest.pop("openai_api_key", None)) is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        api_base = rest.pop("api_base", "https://api.openai.com/v1")
        if model.startswith("o1"):
            stop = []
        api_arguments = {
            "model": model,
            "prompt": "<|endoftext|>" if prompt is None else prompt,
            "temperature": temperature,
            (
                "max_completion_tokens" if model.startswith("o1") else "max_tokens"
            ): max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop if stop != [] else None,
            "user": hashed_user_id,
            "logit_bias": logit_bias,
            "n": num_completions,
            **rest,
        }
        # remove None values, OpenAI API doesn't like them
        for key, value in dict(api_arguments).items():
            if value is None:
                del api_arguments[key]
        if (
            model.startswith("gpt-3.5")
            or model.startswith("gpt-4")
            or model.startswith("o1")
            or model.startswith("openpipe:")
            or model.startswith("gpt4")
        ) and not model.endswith("-base"):
            api_arguments["messages"] = format_messages(api_arguments["prompt"], "user")
            if "prompt" in api_arguments:
                del api_arguments["prompt"]
            if "logprobs" in api_arguments:
                del api_arguments["logprobs"]
            api_suffix = "/chat/completions"
        else:
            api_suffix = "/completions"
        async with session.post(
            api_base + api_suffix, headers=headers, json=api_arguments
        ) as response:
            response.raise_for_status()
            api_response = await response.json()
        try:
            return {
                "prompt": {"text": prompt if prompt is not None else "<|endoftext|>"},
                "completions": [
                    {
                        "text": (
                            completion["text"]
                            if api_suffix == "/completions"
                            else completion["message"]["content"]
                        ),
                        "finish_reason": {
                            "reason": completion.get("finish_reason", "unknown")
                        },
                    }
                    for completion in api_response["choices"]
                ],
                "model": api_response.get("model"),
                "id": api_response.get("id"),
                "created": api_response.get("created"),
                "usage": {
                    # "prompt_tokens": api_response.usage.prompt_tokens,
                    # # if the completion is empty, the value will be missing
                    # "completion_tokens": api_response.usage.get("completion_tokens", 0),
                    # "charged_tokens": api_response.usage.total_tokens,
                    "vendor": vendor,
                },
            }
        except KeyError:
            if "error" in api_response:
                raise ValueError(
                    "API returned error: " + json.dumps(api_response["error"])
                )
            else:
                raise ValueError("API responded with: " + json.dumps(api_response))
    elif vendor == "ai21":
        import httpx

        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api.ai21.com/studio/v1/{model}/complete",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("ai21_api_key", os.environ.get("AI21_API_KEY"))
                },
                json={
                    "prompt": prompt,
                    "numResults": num_completions or 1,
                    "maxTokens": max_tokens,
                    # "stopSequences": stop,
                    "topP": top_p,
                    "temperature": temperature,
                    "frequencyPenalty": {"scale": frequency_penalty},
                    "presencePenalty": {"scale": presence_penalty},
                    **kwargs,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        completion_tokens = sum(
            [
                len(completion["data"]["tokens"])
                for completion in api_response["completions"]
            ]
        )
        return {
            "prompt": {"text": api_response["prompt"]["text"]},
            "completions": [
                {
                    "text": completion["data"]["text"],
                    "finish_reason": completion["finishReason"]["reason"],
                }
                for completion in api_response["completions"]
            ],
            "model": model,
            "id": api_response["id"],
            "usage": {
                "prompt_tokens": len(api_response["prompt"]["tokens"]),
                "completion_tokens": completion_tokens,
                "charged_tokens": completion_tokens,
                "vendor": vendor,
            },
        }
    elif vendor == "textsynth":
        pass
    elif vendor == "huggingface":
        import httpx

        # todo: freq and pres penalties
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get(
                        "huggingface_api_key", os.environ.get("HUGGINGFACE_API_KEY")
                    )
                },
                json={
                    "inputs": prompt,
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    **kwargs,
                },
            )
    elif vendor == "forefront":
        import httpx

        if "t5-20b" in model:
            prompt = prompt + " <extra_id_0>"
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://shared-api.{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("forefront_api_key", os.getenv("FOREFRONT_API_KEY"))
                },
                json={
                    "text": prompt,
                    "top_p": top_p,
                    "top_k": 50400 or top_k,
                    "temperature": temperature,
                    "tfs": tfs,
                    "length": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "stop": stop,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        return {
            "prompt": {"text": prompt},
            "model": api_response["model"],
            "completions": [
                {"text": output["completion"]} for output in api_response["result"]
            ],
            "created": api_response["timestamp"],
            # forefront bills both the prompt and completion
            "usage": NotImplemented,
        }
    elif vendor == "anthropic":
        import anthropic

        if num_completions not in [None, 1]:
            raise NotImplementedError("Anthropic only supports num_completions=1")
        client = anthropic.Client(
            api_key=kwargs.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        )
        if "anthropic_api_key" in kwargs:
            del kwargs["anthropic_api_key"]

        # remove None values, Anthropic API doesn't like them
        for key, value in dict(kwargs).items():
            if value is None:
                del kwargs[key]
        response = await client.acompletion(
            model=model,
            prompt=prompt or "\n\nHuman:",
            max_tokens_to_sample=max_tokens or 16,
            temperature=temperature or 1,
            top_p=top_p or 1,
            # top_k=top_k or -1,
            stop_sequences=stop or list(),
            disable_checks=True,
            **kwargs,
        )
        if response["stop_reason"] == "stop_sequence":
            finish_reason = "stop"
        elif response["stop_reason"] == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "unknown"
        return {
            "prompt": {
                "text": prompt,
            },
            "completions": [
                {"text": response["completion"], "finish_reason": finish_reason}
            ],
            "model": model,
            "id": uuid.uuid4(),
            "created": datetime.datetime.now(),
            "usage": {
                "charged_tokens": 0,
                "vendor": vendor,
            },
        }
    elif vendor == "replicate":
        import httpx

        async with httpx.AsyncClient() as client:
            initial_response = await client.post(
                "https://api.replicate.com/v1/predictions",
                json={
                    "version": model.split(":")[1],
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop_sequence": stop[0],
                },
            )
            initial_response.raise_for_status()
            initial_response_json = initial_response.json()
            response = initial_response
            while response.json()["status"] != "succeeded":
                await asyncio.sleep(0.25)
                response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{initial_response_json['id']}"
                )
                response.raise_for_status()
        return {"prompt": {"text": prompt}, "completions": {}}
    elif vendor == "fake-local":
        return intermodel.callgpt_faker.fake_local(
            model=tokenize_as,
            vendor=vendor,
            prompt=prompt,
            max_tokens=max_tokens,
            num_completions=num_completions,
        )
    else:
        raise NotImplementedError(f"Unknown vendor {vendor}")


SWITCH_ROLE_START = ["<|start_header_id|>", "<|im_start|>"]
SWITCH_ROLE_END = ["<|end_header_id|>", "<|im_sep|>"]
END_TURN = ["<|eot_id|>", "<|im_end|>"]
SET_NAME_START = ["<|name_start|>"]
SET_NAME_END = ["<|name_end|>"]
ALL_DELIMITERS = (
    SWITCH_ROLE_START + SWITCH_ROLE_END + END_TURN + SET_NAME_START + SET_NAME_END
)


def split_many(string: str, delimiters: Iterable[str]) -> List[str]:
    # Escape special regex characters and join delimiters with '|'
    pattern = "(" + "|".join(re.escape(d) for d in delimiters) + ")"
    # Split the string
    result = re.split(pattern, string)
    # Remove empty strings from the result
    return [part.strip() for part in result if part.strip()]


def format_messages(
    string: str, initial_role, initial_name=None, sticky_name=True
) -> list:
    role = initial_role
    name = initial_name
    substrings = split_many(string, ALL_DELIMITERS)
    i = 0
    messages = []
    cur_message_content = ""
    while i < len(substrings):
        substring = substrings[i]
        if substring in SWITCH_ROLE_START:
            sofar = ""
            j = i + 1
            while j < len(substrings):
                search = substrings[j]
                if search in SWITCH_ROLE_END:
                    role = sofar
                    break
                else:
                    sofar += substrings[j]
                j += 1
            i = j
        elif substring in SET_NAME_START:
            sofar = ""
            j = i + 1
            while j < len(substrings):
                search = substrings[j]
                if search in SET_NAME_END:
                    name = sofar
                    break
                else:
                    sofar += substrings[j]
                j += 1
            i = j
        elif substring in END_TURN:
            message = {"content": cur_message_content, "role": role}
            if name is not None:
                message["name"] = name
            messages.append(message)
            if not sticky_name:
                name = None
            cur_message_content = ""
        else:
            cur_message_content += substring
        i += 1
    if cur_message_content != "":
        message = {"content": cur_message_content, "role": role}
        if name is not None:
            message["name"] = name
        messages.append(message)
    return messages


def complete_sync(*args, **kwargs):
    return asyncio.run(complete(*args, **kwargs))


def tokenize(model: str, string: str) -> List[int]:
    import tiktoken

    model = parse_model_string(MODEL_ALIASES.get(model, model)).tokenize_as
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    # actual tokenizer for claude 3.x models is unknown
    if vendor == "openai" or model == "gpt2" or model.startswith("claude-3"):
        # tiktoken internally caches loaded tokenizers
        if model.startswith("claude-3"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("o1"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = tiktoken.encoding_for_model(model)
        # encode special tokens as normal
        # XXX: make this an option
        return tokenizer.encode(string, allowed_special="all")
    elif vendor == "anthropic":
        # all anthropic tokenzers are private now
        return tokenize("gpt2", string)
    elif model in untokenizable:
        return tokenize("gpt2", string)
    else:
        try:
            tokenizer = get_hf_tokenizer(model)
        except Exception as e:
            message = (
                f"Failed to download tokenizer by looking up {model} as a huggingface model ID."
                "To override the tokenizer, put the correct huggingface ID ^ after the model name and follow it with a max token cap."
                "For example, to use OpenRouter Gemini while using the Gemma tokenizer with a maximum context window of 2 million Gemma tokens, use `google/gemini-pro-1.5^google/gemma-7b@2000000`"
                "Using the incorrect tokenizer may lead to flakiness and intermittent failures. Some proprietary models have openly available tokenizers."
            )
            if e.__class__.__name__ == "GatedRepoError":
                warnings.warn(
                    message
                    + f" Cause: GatedRepoError. Log in with huggingface-cli and accept the model ToS at https://huggingface.co/{model} to access the model tokenizer"
                )
            elif e.__class__.__name__ == "RepositoryNotFoundError":
                warnings.warn(message + f" Cause: Not found on huggingface")
            else:
                warnings.warn(message)
            return tokenize("gpt2", string)
        else:
            return tokenizer.encode(string).ids


@dataclasses.dataclass
class Model:
    model: str
    tokenize_as: str
    max_token_len: int


def parse_model_string(model: str):
    # returns: model, tokenization model, max token length
    match = re.match(r"^(.+?)(?:\^(.+?))?(?:@(\d+))?$", model)
    tokenize_as = match.group(2) or match.group(1)
    return Model(
        model=match.group(1),
        tokenize_as=tokenize_as,
        max_token_len=(
            int(match.group(3))
            if match.group(3) is not None
            else max_token_length_inner(tokenize_as)
        ),
    )


def count_tokens(model: str, string: str) -> int:
    return len(tokenize(model, string))


def untokenize(model: str, token_ids: List[int]) -> str:
    model = MODEL_ALIASES.get(model, model)
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    if vendor == "openai" or model == "gpt2":
        import tiktoken

        # tiktoken internally caches loaded tokenizers
        tokenizer = tiktoken.encoding_for_model(model)
        return tokenizer.decode(token_ids)
    elif vendor == "anthropic":
        import anthropic

        # anthropic caches the tokenizer
        # XXX: this may send synchronous network requests, could be downloaded as part of build
        tokenizer = anthropic.get_tokenizer()
        encoded_text = tokenizer.decode(token_ids)
        return encoded_text.ids
    else:
        try:
            tokenizer = get_hf_tokenizer(model)
        except Exception as e:
            if e.__class__.__name__ == "GatedRepoError":
                raise ValueError(f"Log in with huggingface-cli to access {model}")
            elif e.__class__.__name__ == "RepositoryNotFoundError":
                raise NotImplementedError(f"I don't know how to tokenize {model}")
            else:
                raise
        else:
            return tokenizer.decode(token_ids)
        raise NotImplementedError(f"I don't know how to tokenize {model}")


def pick_vendor(model, custom_config=None):
    if custom_config is not None:
        for vendor_name, vendor in custom_config.items():
            if vendor["provides"] is not None:
                for pattern in vendor["provides"]:
                    if re.fullmatch(pattern, model):
                        return vendor_name
    model = MODEL_ALIASES.get(model, model)
    if (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
        or "text-moderation-" in model
        or model.startswith("ft-")
        or model.startswith("gpt-4")
        or model.startswith("gpt-3.5-")
        or model.startswith("o1-")
    ):
        return "openai"
    elif "j1-" in model or model.startswith("j2-"):
        return "ai21"
    elif "forefront" in model:
        return "forefront"
    elif model.startswith("claude-"):
        return "anthropic"
    elif "/" in model:
        return "huggingface"
    else:
        raise NotImplementedError("Unknown model")


def max_token_length(model):
    return parse_model_string(model).max_token_len


def max_token_length_inner(model):
    """
    The maximum number of tokens in the prompt and completion
    """
    if model == "gpt2":
        return 1024
    elif model == "gpt-4-32k":
        return 32769
    elif model.startswith("o1"):
        return 128_000
    elif model.startswith("gpt-4"):
        return 8193
    elif model == "code-davinci-002":
        return 8001
    elif model.startswith("code"):
        raise ValueError("Unknown maximum")
    elif model == "gpt-3.5-turbo-16k":
        return 16385
    elif model in ("babbage-002", "davinci-002"):
        return 16385
    elif model == "gpt-3.5-turbo":
        return 4097
    elif model == "gpt-3.5-turbo-instruct":
        return 4097
    elif model == "text-davinci-003" or model == "text-davinci-002":
        return 4097
    elif model in (
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ):
        return 8191
    elif model.startswith("text-embedding-") and model.endswith("-001"):
        return 2046
    elif model == "claude-3-sonnet-20240229-steering-preview":
        return int(18_000 * 0.7)
    elif model.startswith("claude-3"):
        return 200_000 * 0.7
    elif model.startswith("claude-2.1"):
        return 200_000 * 0.7
    elif model.startswith("claude-2"):
        return 100_000 * 0.7
    elif model.startswith("claude"):
        return 100_000 * 0.7
    elif (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
    ):
        return 2049
    else:
        try:
            import huggingface_hub

            try:
                fname = huggingface_hub.hf_hub_download(
                    model,
                    "config.json",
                    token=get_hf_auth_token(),
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                fname = huggingface_hub.hf_hub_download(
                    model,
                    "config.json",
                    token=get_hf_auth_token(),
                    local_files_only=False,
                )

            with open(fname, "r") as f:
                import json

                data = json.load(f)
            return data["max_position_embeddings"] + 1
        except Exception as e:
            raise NotImplementedError(
                f"Unable to download {model} from HuggingFace. "
                f"Are you logged into HuggingFace (`rye run huggingface-cli login`) and have you agreed to the model"
                f"license at `https://huggingface.co/{model}`?"
            )


class InteractiveIntermodel(cmd.Cmd):
    prompt = "intermodel> "

    def do_c(self, arg):
        """
        Send a completion request to a model
        Usage: c <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(complete_sync(model, pick_vendor(model), prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_t(self, arg):
        """
        Tokenize a model
        Usage: t <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(tokenize(model, prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_EOF(self, arg):
        """Exit"""
        return True


if __name__ == "__main__":
    InteractiveIntermodel().cmdloop()
