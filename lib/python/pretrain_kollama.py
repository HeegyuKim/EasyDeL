from EasyDel import (
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    CausalLanguageModelPretrainer,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    EasyDelState,
    EasyDeLXRapTureConfig,
    get_modules_by_type,
    easystate_to_huggingface_model
)
from datasets import load_from_disk

from flax.core import FrozenDict, unfreeze
from transformers import AutoTokenizer
from jax import numpy as jnp
import jax
from transformers import GemmaForCausalLM as ModuleTorch
import fire



def main(size: str, lr: float, batch_size: int):
    run_name = f"ko-llama-pretrain-{size}"
    dataset_dir = "/data-plm/gemma-adapt/train/"
    pretrained_model_name_or_path = f"heegyu/ko-llama-{size}-random"
    push2hub = f"heegyu/ko-llama-{size}-test"

    batch_total_tokens = 1 * 1024 * 1024
    max_length = 2048
    # batch_size = 1
    total_batch_size = batch_total_tokens // max_length

    # pretrained_model_name_or_path = "google/gemma-2b-it"

    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=(1, 1, 1, -1)
    )

    config = model.config

    model_parameters = FrozenDict({"params": params})

    dtype = jnp.bfloat16
    config.add_basic_configurations(
        attn_mechanism="normal",
        block_b=1,
        block_q=128,
        block_k=128,
        block_k_major=128,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )


    configs_to_initialize_model_class = {
        'config': config,
        'dtype': dtype,
        'param_dtype': dtype,
        'input_shape': (1, max_length)
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(
        dataset_dir
    )

    item = dataset[0]
    for k, v in item.items():
        print(k, len(v))

    train_args = TrainArguments(

        model_class=get_modules_by_type(config.model_type)[1],
        configs_to_initialize_model_class=configs_to_initialize_model_class,
        custom_rule=config.get_partition_rules(True),

        model_name=run_name,

        num_train_epochs=4,
        learning_rate=lr, # 5e-5,
        learning_rate_end=0.1 * lr,
        warmup_steps=1000,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0.02,
        total_batch_size=batch_size,
        gradient_accumulation_steps=total_batch_size // batch_size,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=(1, -1, 1, 1),
        use_pjit_attention_force=False,

        init_input_shape=(1, max_length),

        dtype=dtype,
        param_dtype=dtype,

        step_start_point=0,

        wandb_entity=None
    )

    trainer = CausalLanguageModelPretrainer(
        train_args,
        dataset,
        checkpoint_path=None
    )

    model_parameters = model_parameters if not use_lora else None

    output = trainer.train(
        model_parameters=model_parameters,
        state=None
    )

    output.state = output.state.replace(params=FrozenDict(params))
    output.state.save_state("Jupyter-State.easy")
    with jax.default_device(jax.devices("cpu")[0]):
        model = easystate_to_huggingface_model(
            state=EasyDelState.load_state(
                "Jupyter-State.easy"
            ),
            base_huggingface_module=ModuleTorch,
            config=config
        )

    # model = model.half()
    model.push_to_hub(push2hub)
    tokenizer.push_to_hub(push2hub)


if __name__ == "__main__":
    # tyro.cli(main)
    fire.Fire(main)
