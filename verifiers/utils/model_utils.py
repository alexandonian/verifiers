from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None


def if_flash_attn_supported(min_compute_capability: float = 8.0) -> bool:
    """
    Check if the current GPU setup supports Flash Attention.

    Flash Attention typically requires:
    1. CUDA GPU with compute capability >= 8.0 (NVIDIA Ampere architecture or newer)
    2. PyTorch with CUDA support
    3. Flash Attention package installed

    Args:
        min_compute_capability: Minimum CUDA compute capability required (default: 8.0)

    Returns:
        bool: True if Flash Attention is supported, False otherwise
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return False

    # Check for GPU Compute capability
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return False

    for device_idx in range(device_count):
        device = torch.cuda.get_device_properties(device_idx)
        compute_capability = device.major + device.minor / 10.0
        if compute_capability < min_compute_capability:
            return False
    return True


def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
            if if_flash_attn_supported()
            else "eager",
            use_cache=False,
        )
    if not is_liger_available():
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    print("Using Liger kernel")
    from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore

    return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_tokenizer(model_name: str) -> Any:
    # tokenizer = None
    # if "Instruct" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    # else:
    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)# + "-Instruct")
    #     except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if not hasattr(tokenizer, "pad_token"):
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(
            f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute."
        )
    return tokenizer


def get_model_and_tokenizer(
    model_name: str, model_kwargs: Union[Dict[str, Any], None] = None
) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer
