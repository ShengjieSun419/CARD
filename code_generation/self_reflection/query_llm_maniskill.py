import argparse
from generation import ZeroShotGenerator
import pickle
import os, sys
import json
from code_generation.self_reflection.benchmark_prompt.panda_prompt import panda_system_prompt, panda_first_generation_prompt
from code_generation.self_reflection.benchmark_prompt.mobile_dual_arm_prompt import mobile_dual_arm_system_prompt, mobile_dual_arm_first_generation_prompt
from code_generation.self_reflection.benchmark_prompt.mobile_panda_prompt import mobile_panda_system_prompt, mobile_panda_first_generation_prompt
from code_generation.self_reflection.benchmark_prompt.common_prompt import self_reflection_prompt
from code_generation.self_reflection.maniskill_exp_one_step import LLM_RESPONSE, franka_list, mobile_list, instruction_mapping, mapping_dicts_mapping, env_is_valid
from utils import Logger

task_list = franka_list + mobile_list

LiftCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_height = 0.2 # in meters, indicate the z-axis height of our target
""".strip()

PickCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of our target position
""".strip()

StackCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
""".strip()

TurnFaucet_Env = """
    self.faucet : ArticulateObject # faucet in the environment
    self.faucet.handle : LinkObject # the handle of the faucet in the environment
    self.robot : PandaRobot # a Franka Panda robot
""".strip()


system_prompt_mapping = {
    "LiftCube-v0": panda_system_prompt.replace("<environment_description>", LiftCube_Env),
    "PickCube-v0": panda_system_prompt.replace("<environment_description>", PickCube_Env),
    "StackCube-v0": panda_system_prompt.replace("<environment_description>", StackCube_Env),
    "TurnFaucet-v0": panda_system_prompt.replace("<environment_description>", TurnFaucet_Env),
    "OpenCabinetDoor-v1": mobile_panda_system_prompt,
    "OpenCabinetDrawer-v1": mobile_panda_system_prompt,
    "PushChair-v1": mobile_dual_arm_system_prompt,
}

first_generattion_prompt_mapping = {
    "LiftCube-v0": panda_first_generation_prompt,
    "PickCube-v0": panda_first_generation_prompt,
    "StackCube-v0": panda_first_generation_prompt,
    "TurnFaucet-v0": panda_first_generation_prompt,
    "OpenCabinetDoor-v1": mobile_panda_first_generation_prompt,
    "OpenCabinetDrawer-v1": mobile_panda_first_generation_prompt,
    "PushChair-v1": mobile_dual_arm_first_generation_prompt,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_question_file_path', type=str,)
    args = parser.parse_args()

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
        general_code, specific_code, general_code_path, specific_code_path = code_generator.first_generation(
                                system_prompt=system_prompt_mapping[rl_args.task_name],
                                first_generation_prompt=first_generattion_prompt_mapping[rl_args.task_name],
                                instruction=instruction_mapping[rl_args.task_name],
                                map_dict=mapping_dicts_mapping[rl_args.task_name])
    else:
        general_code, specific_code, general_code_path, specific_code_path = code_generator.self_reflection(
                                self_reflection_prompt=self_reflection_prompt,
                                instruction=instruction_mapping[rl_args.task_name],
                                map_dict=mapping_dicts_mapping[rl_args.task_name],
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