from .mistral import FlaxMistralForCausalLM

import transformers as tf

tf.FlaxAutoModelForCausalLM.register(
    tf.MistralConfig,
    FlaxMistralForCausalLM,
    exist_ok=True
    )