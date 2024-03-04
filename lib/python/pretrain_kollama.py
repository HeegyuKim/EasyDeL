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
from transformers import LlamaForCausalLM as ModuleTorch
import fire

import gcsfs
from datasets import load_dataset, Dataset, IterableDataset


def iter_gcs_files(dirname):
    fs = gcsfs.GCSFileSystem(project='gpt-tpu-370700')
    files = fs.ls(dirname)
    files = [f"gs://{f}" for f in files if f.endswith('.arrow')]
    print(f"{len(files)} files found")

    for file in files:
        with fs.open(file) as f:
            ds = Dataset.from_buffer(f.read())
            yield from ds


SHARDING_AXIES = {
    "dp": (-1, 1, 1, 1),
    "fsdp": (1, -1, 1, 1),
    "mp": (1, 1, 1, -1)
}


def main(size: str, 
         lr: float, 
         batch_size: int, 
         save_dir: str, 
         sharding: str = "dp", 
         save_tokens_in_billion: float = 5,
         total_train_tokens_in_billion: float = 50,
         ):
    sharding_axis_dims = SHARDING_AXIES[sharding]
    run_name = f"ko-llama-pretrain-{size}"
    dataset_dir = "gs://ko-llama-tiny/data-plm-v1/ko-tiny-llama/train"
    pretrained_model_name_or_path = f"heegyu/ko-llama-{size}-random"
    push2hub = f"heegyu/ko-llama-{size}-v1"

    batch_total_tokens = 1 * 1024 * 1024
    max_length = 2048
    # batch_size = 1
    total_batch_size = batch_total_tokens // max_length
    # save every n-billion tokens
    save_steps = int(save_tokens_in_billion * 1024 ** 3 // max_length // batch_size)
    
    max_training_steps = int(total_train_tokens_in_billion * 1024 ** 3 // max_length // batch_size)
    # 10B 토큰까지 LR 감소 이후 최소치 유지 
    max_schedule_steps = int(min(10, total_train_tokens_in_billion) * 1024 ** 3 // max_length // batch_size)

    print(f"save every {save_steps} steps ({save_tokens_in_billion}B tokens)")

    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=sharding_axis_dims
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

    dataset = IterableDataset.from_generator(
        iter_gcs_files,
        gen_kwargs={"dirname": dataset_dir},
        )


    train_args = TrainArguments(

        model_class=get_modules_by_type(config.model_type)[1],
        configs_to_initialize_model_class=configs_to_initialize_model_class,
        custom_rule=config.get_partition_rules(True),

        model_name=run_name,
        num_train_epochs=100,
        max_training_steps=max_training_steps,
        max_scheduler_steps=max_schedule_steps,
        learning_rate=lr, # 5e-5,
        learning_rate_end=0.1 * lr,
        warmup_steps=1000,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0.002,
        total_batch_size=batch_size,
        gradient_accumulation_steps=total_batch_size // batch_size,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=sharding_axis_dims,
        use_pjit_attention_force=False,

        init_input_shape=(1, max_length),
        save_dir=save_dir,
        save_steps=save_steps,
        save_temp_dir=True,

        dtype=dtype,
        param_dtype=dtype,

        step_start_point=0,

        wandb_entity=None,
        loss_remat='',

        push_to_hub=True,
        push_to_hub_id=push2hub,
        push_to_hub_hf_pt_model_cls=ModuleTorch,
    )

    trainer = CausalLanguageModelPretrainer(
        train_args,
        dataset,
        checkpoint_path=None
    )

    output = trainer.train(
        model_parameters=model_parameters,
        state=None
    )

    # output.state = output.state.replace(params=FrozenDict(params))
    # output.state.save_state("Jupyter-State.easy")
    # with jax.default_device(jax.devices("cpu")[0]):
    #     model = easystate_to_huggingface_model(
    #         state=EasyDelState.load_state(
    #             "Jupyter-State.easy"
    #         ),
    #         base_huggingface_module=ModuleTorch,
    #         config=config
    #     )

    # model = model.half()
    # model.push_to_hub(push2hub)
    # tokenizer.push_to_hub(push2hub)


if __name__ == "__main__":
    # tyro.cli(main)
    fire.Fire(main)
