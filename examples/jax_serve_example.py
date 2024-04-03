from typing import List, Union, Optional

from absl.app import run
from absl import flags
from lib.python.EasyDel import JAXServer, JAXServerConfig
import jax
from fjformer import get_dtype
from lib.python.EasyDel.serve.prompters import GemmaPrompter, Llama2Prompter, OpenChatPrompter, PrometheusPrompter, ChatMLPrompter, ZephyrPrompter, HD42DotPrompter
from lib.python.EasyDel.serve.prompters.base_prompter import BasePrompter

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "prompter_type",
    enum_values=("gemma", "llama", "openchat", "chatml", "zephyr", "qwen2", "prometheus", "42dot"),
    help="Prompter to be used to prompt the model",
    default="gemma"
)
flags.DEFINE_string(
    "pretrained_model_name_or_path",
    default="google/gemma-7b-it",
    help="The pretrained model path in huggingface.co/models"
)
flags.DEFINE_string(
    "tokenizer",
    default="",
    help="The pretrained model path in huggingface.co/models"
)

flags.DEFINE_integer(
    "max_compile_tokens",
    default=256,
    help="Maximum number of compiled tokens"
)

flags.DEFINE_integer(
    "max_sequence_length",
    default=4096,
    help="max sequence length to be used in the model"
)
flags.DEFINE_integer(
    "prompt_length",
    default=2048,
    help="max sequence length to be used in the model"
)


flags.DEFINE_enum(
    "dtype",
    enum_values=(
        "bf16",
        "fp16",
        "fp32"
    ),
    default="bf16",
    help="The data type of the model"
)

flags.DEFINE_list(
    "sharding_axis_dims",
    default=[1, 1, 1, -1],
    help="Sharding Axis dimensions for the model"
)

flags.DEFINE_bool(
    "use_sharded_kv_caching",
    default=False,
    help="whether to use sharded kv for Large Sequence model up to 1M"
)

flags.DEFINE_bool(
    "scan_ring_attention",
    default=True,
    help="whether to scan ring attention for Large Sequence model up to 1M (works with attn_mechanism='ring')"
)

flags.DEFINE_bool(
    "use_scan_mlp",
    default=False,
    help="whether to scan MLP or FFN Layers for Large Sequence model up to 1M"
)

flags.DEFINE_enum(
    "attn_mechanism",
    enum_values=["normal", "flash", "ring", "splash"],
    default="normal",
    help="The attention mechanism to be used in the model"
)

flags.DEFINE_integer(
    "block_k",
    default=128,
    help="the number of chunks for key block in attention (Works with flash, splash, ring Attention mechanism)"
)

flags.DEFINE_integer(
    "block_q",
    default=128,
    help="the number of chunks for query block in attention (Works with flash, splash, ring Attention mechanism)"
)

flags.DEFINE_bool(
    "share_gradio",
    default=True,
    help="whether to share gradio app"
)
flags.DEFINE_integer(
    "eos_token_id",
    default=-1,
    help="eos token id"
)


def main(argv):
    title = FLAGS.pretrained_model_name_or_path

    if FLAGS.pretrained_model_name_or_path.startswith("peft:"):
        peft_model_id = FLAGS.pretrained_model_name_or_path.split(":", 1)[1]
        
        if "@" in peft_model_id:
            peft_model_id, revision = peft_model_id.split("@", 1)
        else:
            peft_model_id, revision = peft_model_id, None


        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(peft_model_id, revision=revision)
        FLAGS.pretrained_model_name_or_path = peft_config.base_model_name_or_path
        # peft:heegyu/hi@epoch-3

        adapter_kwargs = dict(
            model_id=peft_model_id,
            revision=revision,
        )
        print(f"Original Model: {FLAGS.pretrained_model_name_or_path}, Peft Model: {peft_model_id}, Peft Revision: {revision}")

    else:
        adapter_kwargs = None

    server_config = JAXServerConfig(
        prompt_length=FLAGS.prompt_length,
        max_sequence_length=FLAGS.max_sequence_length,
        max_compile_tokens=FLAGS.max_sequence_length - FLAGS.prompt_length, # FLAGS.max_compile_tokens,
        # max_new_tokens=FLAGS.max_sequence_length,
        eos_token_id=FLAGS.eos_token_id if FLAGS.eos_token_id != -1 else None,
        dtype=FLAGS.dtype,
        host="0.0.0.0",
        port=35020,
        title=title
    )
    prompters = {
        "gemma": GemmaPrompter(),
        "llama": Llama2Prompter(),
        "openchat": OpenChatPrompter(),
        "chatml": ChatMLPrompter(),
        "zephyr": ZephyrPrompter(),
        "qwen2": ChatMLPrompter(),
        "prometheus": PrometheusPrompter(),
        "42dot": HD42DotPrompter(),
    }
    prompter: BasePrompter = prompters[FLAGS.prompter_type]

    FLAGS.sharding_axis_dims = tuple([int(s) for s in FLAGS.sharding_axis_dims])

    class JAXServerC(JAXServer):
        @staticmethod
        def format_chat(history: List[List[str]], prompt: str, system: Union[str, None], response_prefix: Optional[str] = None) -> str:
            return prompter.format_message(
                history=history,
                prompt=prompt,
                system_message=system,
                prefix=response_prefix
            )

        @staticmethod
        def format_instruct(system: str, instruction: str) -> str:
            return prompter.format_message(
                prefix=None,
                system_message=system,
                prompt=instruction,
                history=[]
            )
    print(FLAGS.sharding_axis_dims)

    server = JAXServerC.from_torch_pretrained(
        server_config=server_config,
        pretrained_model_name_or_path=FLAGS.pretrained_model_name_or_path,
        tokenizer_path=FLAGS.tokenizer,
        device=jax.devices('cpu')[0],
        dtype=get_dtype(dtype=FLAGS.dtype),
        param_dtype=get_dtype(dtype=FLAGS.dtype),
        precision=jax.lax.Precision("fastest"),
        sharding_axis_dims=FLAGS.sharding_axis_dims,
        sharding_axis_names=("dp", "fsdp", "tp", "sp"),
        input_shape=(1, server_config.max_sequence_length),
        adapter_kwargs=adapter_kwargs,
        model_config_kwargs=dict(
            fully_sharded_data_parallel=False,
            attn_mechanism=FLAGS.attn_mechanism,
            scan_mlp_chunk_size=FLAGS.max_sequence_length,
            use_scan_mlp=FLAGS.use_scan_mlp,
            scan_ring_attention=FLAGS.scan_ring_attention,
            block_k=FLAGS.block_k,
            block_q=FLAGS.block_q,
            use_sharded_kv_caching=FLAGS.use_sharded_kv_caching
        )
    )

    # server.gradio_inference().launch(
    #     server_name="0.0.0.0",
    #     server_port=7680,
    #     show_api=True,
    #     share=FLAGS.share_gradio
    # )
    server.fire()


if __name__ == "__main__":
    run(main)
