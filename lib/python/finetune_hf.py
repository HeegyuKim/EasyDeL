import jax
jax.distributed.initialize()  # Should not produce any error

from EasyDel import (
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    HfCausalLanguageModelTrainer,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    EasyDelState,
    EasyDeLXRapTureConfig,
    get_modules_by_type,
    easystate_to_huggingface_model
)
from EasyDel.partitioning import get_partition_rules

from dataset import ChatDatasetLoader, DatasetArguments
from typing import Optional, Sequence
import logging
from tqdm.auto import tqdm

import flax
from flax.core import FrozenDict, unfreeze
from fjformer import make_shard_and_gather_fns, match_partition_rules, with_sharding_constraint
from fjformer.checkpoint import get_dtype

import torch
import transformers
from transformers import AutoTokenizer
from jax import numpy as jnp
import fire
import EasyDel.modules.tf_flax


SHARDING_AXIES = {
    "dp": (-1, 1, 1, 1),
    "fsdp": (1, -1, 1, 1),
    "mp": (1, 1, 1, -1)
}


@torch.no_grad()
def load_from_huggingface(
        pretrained_model_name_or_path: str,
        device=jax.devices('cpu')[0],
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        input_shape: Sequence[int] = (1, 1),
        add_params_field: bool = False,
        do_memory_log: bool = False,
        verbose: bool = True,
):  
    with jax.default_device(device):
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
        flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
            config,
            _do_init=True,
            dtype=dtype,
            # param_dtype=param_dtype,
            # precision=precision,
            input_shape=input_shape
            )

        pt_model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        pt_state_dict = pt_model.state_dict()

        print("Converting Pytorch parameters to Flax")
        params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

        # if pt_model.config.tie_word_embeddings and "lm_head" not in pt_state_dict:
        #     print("No lm_head in pytorch model!")
        #     params["lm_head"] = jnp.array(
        #         pt_model.get_input_embeddings().weight.T.numpy(),
        #         dtype=dtype
        #     )
        if pt_model.config.tie_word_embeddings:
            print("Tie word embeddings, delete lm_head from flax model!")
            params.pop("lm_head")

        del pt_state_dict
        del pt_model
        import gc
        gc.collect()

    return flax_model, params
    
def partition_model(mesh, dtype, config, params, fully_sharded_data_parallel):

    with mesh:
        logging.info(
            "matching partition rules"
        )
        partition_specs = match_partition_rules(params=params, rules=get_partition_rules(config, fully_sharded_data_parallel))
        shard_fns, _ = make_shard_and_gather_fns(partition_specs, get_dtype(dtype))
        logging.info(
            "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
        )
        params = flax.traverse_util.flatten_dict(params)
        shard_fns = flax.traverse_util.flatten_dict(shard_fns)
        pbar = tqdm(params.keys())
        for key in pbar:
            key = tuple(key)
            params[key] = shard_fns[key](params[key])
            pbar.set_description("Sharding Params")
        params = flax.traverse_util.unflatten_dict(params)

    return FrozenDict(params)

def main(run_name: str, 
         model_id: str,
         datasets: str,
         step_batch_size: int = 1, 
         lr: float = 2e-5,  
         total_batch_size: int = 128,
         sharding: str = "fsdp", 
         epoch: int = 1,
         warmup_steps: Optional[int] = None,
         warmup_ratio: Optional[float] = 0.1,
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
    run_name = run_name.replace("/", "__")
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
        load_eval=False,
        packing=packing,
        max_length=max_length,
        train_only_response=True,
        streaming=streaming,
        keep_in_memory=keep_in_memory,
        chat_template=chat_template or model_id
    )
    dataset = ChatDatasetLoader(data_args, tokenizer)
    if save_steps is None:
        if not streaming:
            save_steps = int(len(dataset.train_dataset) // step_batch_size * save_epochs) if save_epochs else None
            save_epochs = None
            print(f"{len(dataset.train_dataset)} items in dataset, save every {save_steps} steps.")
            assert len(dataset.train_dataset) > 0, "No items in dataset"
        else:
            print("Cannot estimate dataset size in streaming.")
            save_steps = None
        
    if warmup_ratio is not None:
        if not streaming:
            warmup_steps = int(epoch * len(dataset.train_dataset) // step_batch_size * warmup_ratio)
            print(f"set warmup_steps to {warmup_steps} (ratio: {warmup_ratio})")
        else:
            print("Warmup_ratio is ignored in streaming")


    if push_to_hub_id is None:
        push_to_hub_id = "heegyu/" + f"{model_id}-{run_name}".replace("/", "__")

    model, params = load_from_huggingface(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(1, max_length),
    )

    config = model.config

    model_parameters = FrozenDict({"params": params})

    dtype = jnp.bfloat16
    configs_to_initialize_model_class = {
        'config': config,
        'dtype': dtype,
        # 'param_dtype': dtype,
        'input_shape': (1, max_length)
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    train_args = TrainArguments(
        model_class=model.module_class,
        configs_to_initialize_model_class=configs_to_initialize_model_class,

        model_name=run_name,
        run_name=run_name,

        num_train_epochs=epoch,
        max_training_steps=max_steps,

        learning_rate=lr, # 5e-5,
        learning_rate_end=0.1 * lr,
        warmup_steps=warmup_steps,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.WARM_UP_LINEAR,
        weight_decay=0,
        total_batch_size=step_batch_size,
        gradient_accumulation_steps=total_batch_size // step_batch_size,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        custom_rule=get_partition_rules(model.config, False),
        fully_sharded_data_parallel=False,
        do_shard_fns=False,
        sharding_array=sharding_axis_dims,
        use_pjit_attention_force=False,
        train_on_inputs=False,

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
        # push_to_hub_hf_pt_model_cls=hf_model_cls,

    )

    trainer = HfCausalLanguageModelTrainer(
        train_args,
        dataset.train_dataset,
        checkpoint_path=None,
        model=model,
        tokenizer=tokenizer
    )
    model_parameters = partition_model(
        trainer.mesh,
        dtype,
        model.config,
        model_parameters,
        False
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
