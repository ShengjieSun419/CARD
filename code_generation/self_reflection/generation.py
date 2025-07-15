import re
import time
from typing import Any, List, Mapping, Optional, Tuple
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer, pipeline
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from code_generation.post_process.post_process import RewardFunctionConverter
from code_generation.self_reflection.utils import print_with_color


class ZeroShotGenerator:
    def __init__(self, log_path: str, model_name="gpt-4", temperature=0.7, benchmark_name="metaworld",
                 validation_function=None,
                 max_try_num: int = 3,
                 args=None,
                 **kwargs) -> None:
        self.temperature = temperature
        if model_name in ["gpt-3.5-turbo", "gpt-35-turbo-1106", "gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview", "gemini-1.5-pro-preview-0409"]:
            self.llm = AzureChatOpenAI(
                deployment_name=model_name,
                openai_api_type="azure",
                temperature=self.temperature,
                **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported!")
        self.benchmark_name = benchmark_name
        self.log_path = log_path
        self.validation_function = validation_function
        self.max_try_num = max_try_num
        self.args = args
        self.model_name = model_name
        self.message_history = []

    def generate_code(self, chat_message, map_dict: dict, step, generate_type):
        try_num = 0
        converter = RewardFunctionConverter(map_dict)

        while try_num + 1 <= self.max_try_num:
            try_num += 1
            success_flag = False
            # AIMessage
            response = self.llm.invoke(chat_message)
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            # get code content
            for pattern in patterns:
                match = re.search(pattern, response.content, re.DOTALL)
                if match:
                    break
            if match:
                general_code = match.group(1)
                # Post-processing, replace the general terms with specific terms
                specific_code = converter.general_to_specific(general_code)

                is_valid, check_msg = self.validation_function(specific_code, self.args)
                if is_valid:
                    success_flag = True
                    break
                else:
                    print(check_msg)
                    continue
            else:
                time.sleep(10)
                continue

        if success_flag:
            return response, general_code, specific_code
        else:
            print(f"Maximum number of tries ({self.max_try_num}) reached. To prevent API token waste, terminate the program.")
            exit()

    def save_code(self, general_code, specific_code, step):
        code_dir = os.path.join(self.log_path, f"code_{step}")
        os.makedirs(code_dir, exist_ok=True)
        general_code_path = os.path.join(code_dir, f"general_{step}.py")
        specific_code_path = os.path.join(code_dir, f"specific_{step}.py")
        with open(general_code_path, "w") as f:
            f.write(general_code)
        with open(specific_code_path, "w") as f:
            f.write(specific_code)

        print_with_color(content=f"{step}_general code save to {general_code_path}", color="GREEN")
        print_with_color(content=f"{step}_specific code save to {specific_code_path}", color="GREEN")
        return general_code_path, specific_code_path

    def first_generation(self, system_prompt: str, first_generation_prompt: str, instruction: str, map_dict: dict, step=0) -> Tuple[str, str]:
        # add prompt to message history
        system_message = SystemMessagePromptTemplate.from_template(system_prompt).format()
        first_generation_message = HumanMessagePromptTemplate.from_template(first_generation_prompt).format(instruction=instruction)
        self.message_history.extend([system_message, first_generation_message])

        chat_message = ChatPromptTemplate.from_messages(self.message_history).format_prompt().to_messages()
        response, general_code, specific_code = self.generate_code(chat_message=chat_message, map_dict=map_dict,
                                                         step=step, generate_type="first_generation")
        # add response to message history
        self.message_history.append(response)
        # save reward function
        general_code_path, specific_code_path = self.save_code(general_code, specific_code, step)
        return general_code, specific_code, general_code_path, specific_code_path

    def self_reflection(self, self_reflection_prompt: str, instruction: str, map_dict: dict, inference_results: str, step: int):
        # add prompt to message history
        self_reflection_message = HumanMessagePromptTemplate.from_template(self_reflection_prompt).format(inference_results=inference_results)
        self.message_history.append(self_reflection_message)
        chat_message = ChatPromptTemplate.from_messages(self.message_history).format_prompt().to_messages()
        response, general_code, specific_code = self.generate_code(chat_message=chat_message, map_dict=map_dict,
                                                    step=step, generate_type="self_reflection")

        # add response to message history
        self.message_history.append(response)
        # save reward function
        general_code_path, specific_code_path = self.save_code(general_code, specific_code, step)
        return general_code, specific_code, general_code_path, specific_code_path