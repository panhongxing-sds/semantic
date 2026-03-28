import os
from typing import Type, Dict, Any
import json

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .llm import LLM

from .config import CFG

load_dotenv()


class OAILLM(LLM):
    def __init__(self, model: str = CFG["OAI"]["oai_llm_large"]):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OAI_KEY"))

    def generate(
        self,
        prompt: str,
        temperature: float = CFG["general"]["temperature"],
        n_generations: int = 1,
        length_normalize: bool = CFG["general"]["length_normalize"],
        **kwargs,
    ):
        """
        Generate text in response to a prompt

        Parameters
        ----------
        prompt : str
            The prompt you want to send to the model.
        temperature : float
            Softmax temperature for sampling.
            0 is fully greedy. 1 is 'most random.'
        n_generations : int
            The number of samples you want to draw from the LLM.
        length_normalize : bool
            Whether you want to length-normalize the log-probs.
            You almost certainly want this for consistency.

        *SHOULD* Return
        ---------------
        tuple : (text, log_prob)
            The response text(s) and corresponding sequence(s)' log-probability(ies).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            n=n_generations,
            logprobs=True,
        )

        if n_generations == 1:
            choice_obj = response.choices[0]
            text = choice_obj.message.content
            log_prob = OAILLM.get_logprobs(
                choice_obj=choice_obj, length_normalize=length_normalize
            )
        else:
            choice_objs = response.choices
            text = [choice_obj.message.content for choice_obj in choice_objs]
            log_prob = [
                OAILLM.get_logprobs(
                    choice_obj=choice_obj, length_normalize=length_normalize
                )
                for choice_obj in choice_objs
            ]
        return text, log_prob

    def get_logprobs(choice_obj, length_normalize: bool = True):
        """
        Given an OpenAI response, extract the log-probability.

        Parameters
        ----------
        choice_obj : openai.types.chat.ChatCompletionResponseChoice
            OpenAI API response choice object.
        length_normalize : bool
            Whether you want to length-normalize the log-probs.
            You almost certainly want this for consistency.
        """
        token_log_probs = [_.logprob for _ in choice_obj.logprobs.content]
        log_prob = np.sum(token_log_probs)
        if length_normalize:
            log_prob /= len(token_log_probs)
        return log_prob

    def generate_pydantic(
        self,
        prompt: str,
        pydantic_object: Type[BaseModel],
        pydantic_object_info: Dict[str, Any],
        temperature: float = 0.7,
        seed: float = None,
        **kwargs,
    ):
        """
        OpenAI generation using a Pydantic object.

        Parameters
        ----------
        prompt : str
            The prompt you want to send to the model.
        pydantic_object : Type[BaseModel]
            The Pydantic object class to coerce structured response from the LLM.
        pydantic_object_info : Dict[str, Any]
            The JSON schema for the Pydantic object.
            This should be produced via utils.pydantic.create_function_info_from_pydantic.
        temperature : float
            Softmax temperature for sampling. 0 is fully greedy.
        seed : float
            Random seed.
        **kwargs
            Args for the chat completion method.
        """
        messages = [{"role": "user", "content": prompt}]
        tools = [pydantic_object_info]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature,
            seed=seed,
            **kwargs,
        )

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", [])

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)

            try:
                obj = pydantic_object(**function_arguments)
                return obj
            except ValidationError as e:
                raise ValidationError(f"Validation failed: {e}")
        else:
            raise RuntimeError("Failed to call function.")
