
from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


@datasources("heegyu/ultrafeedback_binarized_feedback:user-feedback")
class UltraFeedbackUserFeedback(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/ultrafeedback_binarized_feedback", split=split)
        return ds

    def map_conversations(self, item):
        convs = deepcopy(item["rejected"])
        for conv in convs:
            conv["trainable"] = False

        feedback = item["feedback"]
        instruction = item["chosen"][-2]["content"]
        convs.append({
            "role": "user",
            "content": f"Your feedback and score for your response are as follows.\n[Feedback]\n{feedback}\n[Instruction]\nFollow the feedback and provide your response again:{instruction}"
        })
        convs.append(item["chosen"][-1])

        return {
            "conversations": convs
        }
    
@datasources("heegyu/ultrafeedback_binarized_feedback:self-feedback")
class UltraFeedbackSelfFeedback(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/ultrafeedback_binarized_feedback", split=split)
        return ds

    def map_conversations(self, item):
        convs = deepcopy(item["rejected"])
        for conv in convs:
            conv["trainable"] = False

        feedback = item["feedback"]
        instruction = item["chosen"][-2]["content"]
        convs.append({
            "role": "user",
            "content": "Score your previous response [1-10] and give feedback"
        })
        convs.append({
            "role": "assistant",
            "content": feedback
        })
        convs.append({
            "role": "user",
            "content": f"Follow the feedback and provide your response again:{instruction}"
        })
        convs.append(item["chosen"][-1])

        return {
            "conversations": convs
        }