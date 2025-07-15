import argparse
from generation import ZeroShotGenerator
import pickle
import os, sys
import json
from code_generation.self_reflection.benchmark_prompt.metaworld_prompt import system_prompt, first_generation_prompt
from code_generation.self_reflection.benchmark_prompt.common_prompt import self_reflection_prompt
from code_generation.self_reflection.metaworld_exp_one_step import LLM_RESPONSE, env_is_valid
from code_generation.self_reflection.metaworld_exp import instruction_mapping, mapping_dicts
from utils import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_question_file_path', type=str,)
    args = parser.parse_args()

    # need to change
    with open(args.rl_question_file_path, 'rb') as file:
        loaded_objects_dict = pickle.load(file)
        rl_args = loaded_objects_dict["rl_args"]
        step = loaded_objects_dict["step"]
        message_history = loaded_objects_dict["message_history"]
        trajectories_for_check_list = loaded_objects_dict["trajectories_for_check_list"]
        feedback_str = loaded_objects_dict["feedback_str"]
    real_log_path = os.path.dirname(args.rl_question_file_path)
    std_file = os.path.join(real_log_path, f"{LLM_RESPONSE}_{step}.log")
    logger = Logger(filename=std_file)
    sys.stdout = logger
    sys.stderr = logger
    print("args is:")
    print(json.dumps(vars(args), indent=4))

    code_generator = ZeroShotGenerator(log_path=real_log_path,
                            model_name=rl_args.llm_model_name,
                            temperature=rl_args.temperature,
                            benchmark_name=rl_args.benchmark_name,
                            validation_function=env_is_valid,
                            max_try_num=rl_args.max_try_num,
                            args=rl_args)
    code_generator.message_history = message_history

    if step == 0:
        # first generation
        general_code, specific_code, general_code_path, specific_code_path = code_generator.first_generation(system_prompt=system_prompt,
                                first_generation_prompt=first_generation_prompt,
                                instruction=instruction_mapping[rl_args.task_name],
                                map_dict=mapping_dicts)
    else:
        general_code, specific_code, general_code_path, specific_code_path = code_generator.self_reflection(self_reflection_prompt=self_reflection_prompt,
                                instruction=instruction_mapping[rl_args.task_name],
                                map_dict=mapping_dicts,
                                inference_results=feedback_str,
                                step=step)

    # save
    with open(os.path.join(real_log_path, f"{LLM_RESPONSE}_{step}.pkl"), 'wb') as file:
        pickle.dump({
            "trajectories_for_check_list": trajectories_for_check_list,
            "message_history": code_generator.message_history
        }, file)
    
if __name__ == "__main__":
    main()