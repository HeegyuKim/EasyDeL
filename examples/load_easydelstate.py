import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
from jax.sharding import PartitionSpec
from typing import Sequence, Optional
from transformers import LlamaForCausalLM as ModuleTorch, AutoConfig, AutoTokenizer
from lib.python.EasyDel import (
    AutoEasyDelModelForCausalLM,
    EasyDelState,
    easystate_to_huggingface_model
)


def load_model(
        config_model_path: str,
        checkpoint_path: str,
        verbose: bool = True,
        state_shard_fns=None,  # You can pass that
        init_optimizer_state: bool = False
):
    config = AutoConfig.from_pretrained(config_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config_model_path)

    with jax.default_device(jax.devices("cpu")[0]):
        state = EasyDelState.load_state(
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            state_shard_fns=state_shard_fns,  # You can pass that
            init_optimizer_state=init_optimizer_state
        )
        model = easystate_to_huggingface_model(
            state=state,
            base_huggingface_module=ModuleTorch,
            config=config
        )

        print(model)

    # print(state)

    test_input = tokenizer("안녕하세요", return_tensors="pt")
    outs = model.generate(**test_input, max_new_tokens=16, early_stopping=False)[0]
    print(tokenizer.decode(outs))


if __name__ == "__main__":
    load_model(
        "heegyu/ko-llama-46M-random",
        "lib/python/tiny-kollama-ckpt/46M/ko-llama-pretrain-46M/ko-llama-pretrain-46M-S13276_13276.easy"
    )
