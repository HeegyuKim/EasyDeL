from EasyDel import (
    TrainArguments,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    DPOTrainer,
    EasyDelState,
    easystate_to_huggingface_model
)

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, LlamaForCausalLM as module_pt
from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec
from fjformer import GenerateRNG
from typing import Optional, Dict
from datasets import Dataset

rng_g = GenerateRNG()
api = HfApi()

max_length = 512  # Overall maximum length
max_target_length = 1024  # Maximum Length for target column in Dataset
max_prompt_length = 1024  # Maximum Length for prompt column in Dataset

model_name_or_path = "erfanzar/LinguaMatic-Tiny"
ref_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dtype = jnp.bfloat16

sharding_axis_dims = (1, 1, 1, -1)
sharding_axis_names = ("dp", "fsdp", "tp", "sp")
query_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Query Partition Spec for Model
key_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Key Partition Spec for Model
value_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Value Partition Spec for Model
bias_partition_spec = PartitionSpec(
    ("dp", "fsdp"), None, None, None
)  # Attention Mask / Bias Partition Spec for Model
attention_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Attention Score / Weight Partition Spec for Model

ref_model_query_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Query Partition Spec for Ref Model
ref_model_key_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Key Partition Spec for Ref Model
ref_model_value_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Value Partition Spec for Ref Model
ref_model_bias_partition_spec = PartitionSpec(
    ("dp", "fsdp"), None, None, None
)  # Attention Mask / Bias Partition Spec for Ref Model
ref_model_attention_partition_spec = PartitionSpec(
    ("dp", "fsdp"), "sp", "tp", None
)  # Attention Score / Weight Partition Spec for Ref Model


arguments = TrainArguments(
    model_name="EasyDeL-DPO",
    num_train_epochs=5,
    learning_rate=1e-4,
    learning_rate_end=3e-5,
    warmup_steps=200,
    optimizer=EasyDelOptimizers.ADAMW,
    scheduler=EasyDelSchedulers.LINEAR,
    weight_decay=0.02,
    total_batch_size=128,
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=sharding_axis_dims,
    fully_sharded_data_parallel=True,
    gradient_accumulation_steps=2,
    dtype=dtype,
    param_dtype=dtype,
    step_start_point=0,
    training_time="7H",
    do_train=True,
    do_eval=True,
    track_memory=False  # Performance boost.
    # You can set other options too or play with them but for now I just stick with these arguments.
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def extract_anthropic_prompt(prompt_and_response):
    """
    Extract the anthropic prompt from a prompt and response pair.
    """
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        context = sample["chosen"][:-1]
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=True)
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][-1]["content"],
            "rejected": sample["rejected"][-1]["content"],
        }

    return dataset.map(split_prompt_and_responses)

train_dataset = get_hh("train", sanity_check=True)
eval_dataset = get_hh("test", sanity_check=True)

state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    query_partition_spec=query_partition_spec,
    key_partition_spec=key_partition_spec,
    value_partition_spec=value_partition_spec,
    bias_partition_spec=bias_partition_spec,
    attention_partition_spec=attention_partition_spec,
)

ref_state = EasyDelState.from_pretrained(
    pretrained_model_name_or_path=ref_model_name_or_path,
    dtype=dtype,
    param_dtype=dtype,
    init_optimizer_state=False,
    free_optimizer_state=True,
    sharding_axis_dims=sharding_axis_dims,
    sharding_axis_names=sharding_axis_names,
    query_partition_spec=ref_model_query_partition_spec,
    key_partition_spec=ref_model_key_partition_spec,
    value_partition_spec=ref_model_value_partition_spec,
    bias_partition_spec=ref_model_bias_partition_spec,
    attention_partition_spec=ref_model_attention_partition_spec,
)

dpo_trainer = DPOTrainer(
    model_state=state,
    ref_model_state=ref_state,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    arguments=arguments,
    max_length=max_length,
    max_target_length=max_target_length,
    max_prompt_length=max_prompt_length,
    ref_model_init_kwargs=None,  # In case that you pass the ref_model_state a string you have to pass this one too
    model_init_kwargs=None,  # In case that you pass the model_state a string you have to pass this one too
    dataset_map_arguments={
        "num_proc": 8,
        "batched": True,
        "batch_size": 100,
    },
    auto_shard_model_state=True,
    auto_shard_ref_model_state=True,
    loss_type="sigmoid",
    data_collator=None,  # Pass None in order to use default data_collector (you can create your own)
)

output = dpo_trainer.train()

easydel_jax_model = output.state  # Here's you EasyDeL Model

with jax.default_device(jax.devices("cpu")[0]):
    model = easystate_to_huggingface_model(
        state=EasyDelState.load_state(
            output.checkpoint_path
        ),
        base_huggingface_module=module_pt,
        config=dpo_trainer.model_state.module.config
    )  # Here's you PyTorch Model

model.push_to_hub("<REPO_ID>", private=False)  # Hope you love open-source too :)
tokenizer.push_to_hub("<REPO_ID>", private=False)  # Hope you love open-source too :)