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
import jax
import fire


SHARDING_AXIES = {
    "dp": (-1, 1, 1, 1),
    "fsdp": (1, -1, 1, 1),
    "mp": (1, 1, 1, -1)
}


def main(run_name: str, 
         model_id: str,
         datasets: str,
         step_batch_size: int = 1, 
         lr: float = 2e-5,  
         total_batch_size: int = 128,
         sharding: str = "fsdp", 
         epoch: int = 1,
         max_steps: Optional[int] = None,
         save_epochs: Optional[int] = None,
         save_steps: Optional[int] = None,
         max_length: int = 2048,
         push_to_hub_id: Optional[str] = None,
         chat_template: Optional[str] = None,
         packing: bool = True,
         streaming: bool = False,
         distributed: bool = False,
         keep_in_memory: bool = False,
         ):
    sharding_axis_dims = SHARDING_AXIES[sharding]
    pretrained_model_name_or_path = model_id
    assert total_batch_size % step_batch_size == 0, "total_batch_size must be divisible by step_batch_size"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )
    data_args = DatasetArguments(
        dataset=datasets,
        limit=None,
        eval_dataset=None,
        packing=packing,
        max_length=max_length,
        streaming=streaming,
        keep_in_memory=keep_in_memory,
        chat_template=chat_template or model_id
    )
    dataset = DatasetLoader(data_args, tokenizer)
    if not streaming:
        save_steps = int(len(dataset.train_dataset) // step_batch_size * save_epochs) if save_epochs else None
        save_epochs = None
        print(f"{len(dataset.train_dataset)} items in dataset, save every {save_steps} steps.")
        assert len(dataset.train_dataset) > 0, "No items in dataset"
    else:
        print("Cannot estimate dataset size in streaming.")
        save_steps = None
        

    if push_to_hub_id is None:
        push_to_hub_id = "heegyu/" + f"{model_id}-{run_name}".replace("/", "__")

    model, params, hf_model_cls = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=sharding_axis_dims,
        return_hf_model_class=True
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

        num_train_epochs=epoch,
        max_training_steps=max_steps,

        learning_rate=lr, # 5e-5,
        learning_rate_end=0.1 * lr,
        warmup_steps=0,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0,
        total_batch_size=step_batch_size,
        gradient_accumulation_steps=total_batch_size // step_batch_size,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=sharding_axis_dims,
        use_pjit_attention_force=False,
        jax_distributed_config=dict(initialize_jax_distributed=distributed),

        init_input_shape=(1, max_length),
        save_temp_dir=True,
        save_steps=save_steps,
        save_epochs=save_epochs,

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

    # output.state = output.state.replace(params=FrozenDict(params))
    # output.state.save_state("Jupyter-State.easy")
    # with jax.default_device(jax.devices("cpu")[0]):
    #     model = easystate_to_huggingface_model(
    #         state=EasyDelState.load_state(
    #             "Jupyter-State.easy"
    #         ),
    #         base_huggingface_module=hf_model_cls,
    #         config=config
    #     )

    # model = model.half()
    # model.push_to_hub(push2hub)
    # tokenizer.push_to_hub(push2hub)


if __name__ == "__main__":
    # tyro.cli(main)
    fire.Fire(main)
