import jax
jax.distributed.initialize()  # Should not produce any error

from EasyDel import (
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    CausalLanguageModelTrainer,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    EasyDelState,
    EasyDeLXRapTureConfig,
    get_modules_by_type,
    easystate_to_huggingface_model
)
from dataset import DatasetLoader, DatasetArguments
from typing import Optional

from flax.core import FrozenDict, unfreeze
from transformers import AutoTokenizer
from jax import numpy as jnp
import fire


SHARDING_AXIES = {
    "dp": (-1, 1, 1, 1),
    "fsdp": (1, -1, 1, 1),
    "mp": (1, 1, 1, -1)
}


def main(run_name: str, 
         model_id: str,
         datasets: str,
         batch_size: int = 1, 
         lr: float = 2e-5,  
         sharding: str = "fsdp", 
         save_tokens_in_billion: float = 5,
         total_train_tokens_in_billion: float = 50,
         max_length: int = 2048,
         push_to_hub_id: Optional[str] = None,
         packing: bool = True,
         streaming: bool = True,
         keep_in_memory: bool = False,
         ):
    sharding_axis_dims = SHARDING_AXIES[sharding]
    pretrained_model_name_or_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )
    data_args = DatasetArguments(
        dataset=datasets,
        load_eval=False,
        limit=None,
        eval_dataset=None,
        packing=packing,
        max_length=max_length,
        streaming=streaming,
        keep_in_memory=keep_in_memory,
        interleaving_strategy="all_exhausted"
    )
    dataset = DatasetLoader(data_args, tokenizer)
    assert dataset.train_dataset is not None, "No training dataset found"
    
    batch_total_tokens = 1 * 1024 * 1024
    total_batch_size = batch_total_tokens // max_length
    # save every n-billion tokens
    save_steps = int(save_tokens_in_billion * 1024 ** 3 // max_length // batch_size)
    
    max_training_steps = int(total_train_tokens_in_billion * 1024 ** 3 // max_length // batch_size)
    # 10B 토큰까지 LR 감소 이후 최소치 유지 
    max_schedule_steps = int(min(10, total_train_tokens_in_billion) * 1024 ** 3 // max_length // batch_size)
    print(f"save every {save_steps} steps ({save_tokens_in_billion}B tokens)")

    if push_to_hub_id is None:
        push_to_hub_id = "heegyu/" + f"{model_id}-{run_name}".replace("/", "__")

    model, params, hf_model_cls = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=sharding_axis_dims,
        return_hf_model_class=True,
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
        use_scan_mlp=True,
    )

    configs_to_initialize_model_class = {
        'config': config,
        'dtype': dtype,
        'param_dtype': dtype,
        'input_shape': (1, max_length)
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

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
        warmup_steps=0,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0,
        total_batch_size=batch_size,
        gradient_accumulation_steps=total_batch_size // batch_size,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=sharding_axis_dims,
        use_pjit_attention_force=False,

        init_input_shape=(1, max_length),
        save_temp_dir=True,
        save_steps=save_steps,

        dtype=dtype,
        param_dtype=dtype,

        step_start_point=0,

        wandb_entity=None,
        shuffle_train_dataset=not streaming,
        push_to_hub=True,
        push_to_hub_id=push_to_hub_id,
        push_to_hub_hf_pt_model_cls=hf_model_cls,
    )

    trainer = CausalLanguageModelTrainer(
        train_args,
        dataset.train_dataset,
        checkpoint_path=None
    )

    output = trainer.train(
        model_parameters=model_parameters,
        state=None
    )


if __name__ == "__main__":
    # tyro.cli(main)
    fire.Fire(main)
