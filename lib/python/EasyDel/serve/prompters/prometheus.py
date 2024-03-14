from abc import ABC
from typing import Optional

from .base_prompter import BasePrompter


class PrometheusPrompter(BasePrompter, ABC):
    def __init__(
            self,
    ):
        user_prefix = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.
"""
        assistant_prefix = "###Feedback: "
        super().__init__(
            user_message_token=user_prefix,
            assistant_message_token=assistant_prefix,
            prompter_type="prometheus",
            end_of_turn_token="</s>",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = ""
        for user, assistant in history:
            prompt += f"{self.user_message_token}{user} "
            prompt += f"{self.assistant_message_token}{assistant} "

        return prompt

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:

        dialogs = system_message if system_message is not None else ""

        for user, assistant in history:
            dialogs += f"{self.user_message_token}{user}"
            dialogs += f"{self.assistant_message_token}{assistant}"

        dialogs += f"{self.user_message_token}{prompt}"
        dialogs += self.assistant_message_token
        if prefix:
            dialogs += prefix
        return dialogs
