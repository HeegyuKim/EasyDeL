from .mistral import FlaxMistralForCausalLM
from .gemma import FlaxGemmaForCausalLM
from .llama import FlaxLlamaForCausalLM

import transformers as tf

tf.FlaxAutoModelForCausalLM.register(
    tf.MistralConfig,
    FlaxMistralForCausalLM,
    exist_ok=True
    )

tf.FlaxAutoModelForCausalLM.register(
    tf.GemmaConfig,
    FlaxGemmaForCausalLM,
    exist_ok=True
    )
    
tf.FlaxAutoModelForCausalLM.register(
    tf.LlamaConfig,
    FlaxLlamaForCausalLM,
    exist_ok=True
    )