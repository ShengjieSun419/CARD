import sys, os
from stable_baselines3.common.callbacks import BaseCallback
import random
import numpy as np
import torch
import re
import colorama
colorama.init()
SUB_REWARD_PREFIX="sub_reward/"
MEAN_SUFFIX="_mean"
STD_SUFFIX="_std"
INFO_PREFIX="convert_info/"
COLOR_ENABLED=False


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if using_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

class SubRewardLog(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path, log_file_name, n_eval_episodes, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.log_file_name = log_file_name
        self.n_eval_episodes = n_eval_episodes
        self.learning_result = {}
        # init learning_result
        obs = self.eval_env.reset()
        next_obs, reward, done, info = self.eval_env.step(self.eval_env.action_space.sample())
        for k, v in info.items():
            if k.startswith(SUB_REWARD_PREFIX):
                self.learning_result[k] = []

    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            for k in self.learning_result.keys():
                self.learning_result[k].append([])
            for i in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_data = {}
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, done, info = self.eval_env.step(action)
                    for k, v in info.items():
                        if k.startswith(SUB_REWARD_PREFIX):
                            if k not in episode_data.keys():
                                episode_data[k] = 0
                            episode_data[k] += v
                    obs = next_obs
                for k, v in episode_data.items():
                    self.learning_result[k][-1].append(v)
            for k, v in self.learning_result.items():
                self.logger.record(f"ep_mean_{k}", np.mean(v[-1]))
        return super()._on_step()

    def on_training_end(self) -> None:
        os.makedirs(self.log_path, exist_ok=True)
        file_path = os.path.join(self.log_path, self.log_file_name)
        np.savez(file_path, **self.learning_result)
        if self.verbose > 0:
            print(f"Sub rewards saved to {file_path}")

def get_learning_result_mean(learning_result_list: list, save_path: str):
    data_all_dict = {}
    for learning_result in learning_result_list:
        for key in learning_result.keys():
            if key not in data_all_dict:
                data_all_dict[key] = []
            if key == "timesteps":
                data_all_dict[key].append(learning_result[key])
            else:
                data_all_dict[key].append(np.mean(learning_result[key], axis=1, keepdims=True))
    learning_result_mean_dict = {}
    for key in data_all_dict:
        if key == "timesteps":
            learning_result_mean_dict[key] = data_all_dict[key][0]
        else:
            learning_result_mean_dict[key] = np.concatenate(data_all_dict[key], axis=1)
            learning_result_mean_dict[key+MEAN_SUFFIX] = np.mean(learning_result_mean_dict[key], axis=1)
            learning_result_mean_dict[key+STD_SUFFIX] = np.std(learning_result_mean_dict[key], axis=1)
    np.savez(save_path, **learning_result_mean_dict)
    return {key: value for key, value in learning_result_mean_dict.items() 
            if key == "timesteps" or MEAN_SUFFIX in key or STD_SUFFIX in key}

def get_learning_result_feedback_simple(learning_result: dict, log_point_num):
    timesteps = learning_result['timesteps']
    results_mean = np.mean(learning_result['results'], axis=1)
    results_std = np.std(learning_result['results'], axis=1)
    ep_lengths_mean = np.mean(learning_result['ep_lengths'], axis=1)
    ep_lengths_std = np.std(learning_result['ep_lengths'], axis=1)
    successes_mean = np.mean(learning_result['successes'], axis=1)
    successes_std = np.std(learning_result['successes'], axis=1)
    learning_result_feedback = "Evaluation is conducted at specific training intervals. The average results of the episodes are as follows:\n"
    if len(timesteps) < log_point_num:
        log_interval = 1
    else:
        log_interval = len(timesteps) // log_point_num
    step_list = list(range(len(timesteps)-1, -1, -log_interval))
    if len(step_list) > log_point_num:
        step_list = step_list[:log_point_num]
    step_list.reverse()
    for i in step_list:
        learning_result_feedback += (f"When step is {timesteps[i]}, "
                    f"return is {results_mean[i]:.1f} (±{results_std[i]:.1f}), "
                    f"episode length is {ep_lengths_mean[i]:.1f} (±{ep_lengths_std[i]:.1f}), "
                    f"success rate is {successes_mean[i]*100:.1f}% (±{successes_std[i]*100:.1f}%), ")
        for key in learning_result.keys():
            if SUB_REWARD_PREFIX in key:
                core_key = key.replace(SUB_REWARD_PREFIX, "")
                value_mean = np.mean(learning_result[key], axis=1)
                value_std = np.std(learning_result[key], axis=1)
                learning_result_feedback += (
                    f"cumulative {core_key} for the trajectory is {value_mean[i]:.1f} (±{value_std[i]:.1f}), "
                )
        learning_result_feedback += "\n"
    if learning_result_feedback[-3:] == ", \n":
        learning_result_feedback = learning_result_feedback[:-3] + ".\n"
    print("learning result (simple) feedback is:")
    print(learning_result_feedback)
    return learning_result_feedback


def collect_trajectories(model, env, mapping_dicts, trajectory_num=10, benchmark_name="metaworld"):
    trajectories = []
    reward_sums = []
    for episode in range(trajectory_num):
        obs = env.reset()
        done = False
        episode_data = {'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'infos': []}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            
            episode_data['obs'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            episode_data['infos'].append(info)
            if benchmark_name == "metaworld":
                for key, value in mapping_dicts.items():
                    if key not in episode_data.keys():
                        episode_data[key] = []
                    if key == "self.goal_position":
                        episode_data[key].append(info[value])
                    else:
                        episode_data[key].append(eval(value))
            elif benchmark_name == "maniskill":
                for key, value in info.items():
                    if key.startswith(INFO_PREFIX):
                        if key not in episode_data.keys():
                            episode_data[key] = []
                        episode_data[key].append(value)
            else:
                raise RuntimeError("benchmark name error!")
            for key, value in info.items():
                if key.startswith(SUB_REWARD_PREFIX):
                    if key not in episode_data.keys():
                        episode_data[key] = []
                    episode_data[key].append(value)
            obs = next_obs
        
        for key in episode_data.keys():
            episode_data[key] = np.array(episode_data[key])
        reward_sums.append(np.sum(episode_data['rewards']))
        trajectories.append(episode_data)
    return trajectories

def convert_traj_to_str(trajectory_list: list, mapping_dicts, log_interval=100, benchmark_name="metaworld"):
    feedback_list = []
    for traj in trajectory_list:
        ep_len = len(traj['obs'])
        feedback_str = ""
        sequence = [0] + list(range(log_interval-1, ep_len, log_interval))
        if sequence[-1] != ep_len-1:
            sequence.append(ep_len-1)
        for i in sequence:
            feedback_str += f"When step is {i+1}, "
            for key, value in traj.items():
                if benchmark_name == "metaworld":
                    if key in mapping_dicts or "reward" in key:
                        if isinstance(value[i], np.ndarray) and value[i].ndim == 1:
                            formatted_value = ", ".join(f"{v:.1f}" for v in value[i])
                            formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                            feedback_str += f"{formatted_key} is [{formatted_value}], "
                        else:
                            formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                            if key == "rewards":
                                formatted_key = "reward"
                            feedback_str += f"{formatted_key} is {value[i]:.1f}, "
                elif benchmark_name == "maniskill":
                    if key.startswith(INFO_PREFIX) or "reward" in key:
                        if isinstance(value[i], np.ndarray) and value[i].ndim == 1:
                            formatted_value = ", ".join(f"{v:.1f}" for v in value[i])
                            formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                            feedback_str += f"{formatted_key} is [{formatted_value}], "
                        else:
                            formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                            if key == "rewards":
                                formatted_key = "reward"
                            feedback_str += f"{formatted_key} is {value[i]:.1f}, "
            feedback_str += "\n"
        if traj["infos"][-1]["success"] == True:
            feedback_str += f"At step {sequence[-1]+1}, the agent solves the task, so the trajectory ends.\n"
        else:
            feedback_str += f"At step {sequence[-1]+1}, the agent still has not solved the task and the maximum trajectory length of {sequence[-1]+1} is reached, so the trajectory ends.\n"
        if feedback_str[-3:] == ", \n":
            feedback_str = feedback_str[:-3] + ". \n"
        feedback_list.append(feedback_str)
    return feedback_list

def get_eval_feedback(trajectory_list: list, mapping_dicts, log_interval=100):
    sums = [np.sum(traj['rewards']) for traj in trajectory_list]
    max_index = np.argmax(sums)
    min_index = np.argmin(sums)
    special_trajectory_list = [trajectory_list[max_index], trajectory_list[min_index]]
    feedback_list = []
    eval_feedback = "After training, we evaluated the model and the two trajectories with the highest and lowest return are as follows:\n"
    feedback_list = convert_traj_to_str(trajectory_list=special_trajectory_list, mapping_dicts=mapping_dicts, log_interval=log_interval)
    eval_feedback += f"trajectories with the highest return {sums[max_index]:.1f} is:\n" + feedback_list[0] + "\n"
    eval_feedback += f"trajectories with the lowest return {sums[min_index]:.1f} is:\n" + feedback_list[1]
    print("eval feedback is:")
    print(eval_feedback)
    return eval_feedback

def convert_function_external(reward_function_code: str, benchmark_name="metaworld"):
    if benchmark_name == "metaworld":
        reward_function_code = re.sub(r'def compute_dense_reward\(self, ', 'def compute_dense_reward(', reward_function_code)
        reward_function_code = re.sub(r"action\s*:\s*np\.ndarray\s*,\s*obs\s*:\s*np\.ndarray\s*\)\s*->\s*Tuple\s*\[\s*float\s*,\s*Dict\s*\[\s*str\s*,\s*float\s*\]\s*\]\s*:", 'action: np.ndarray, obs: np.ndarray, goal_position) -> Tuple[float, Dict[str, float]]:', reward_function_code)
        reward_function_code = re.sub(r'self.env._get_pos_goal\(\)', 'goal_position', reward_function_code)
        return reward_function_code
    elif benchmark_name == "maniskill":
        reward_function_code = re.sub(r'def compute_dense_reward\(self, ', 'def compute_dense_reward(', reward_function_code)
        return reward_function_code
    else:
        raise RuntimeError("benchmark name error!")


def trajectories_return_check(reward_function, trajectories, mapping_dicts, threshold, log_interval=100, benchmark_name="metaworld", env=None):
    success_trajs = {
        "pos": [],
        "traj": [],
        "return": [],
        "reward_per_step": [],
    }
    failure_trajs = {
        "pos": [],
        "traj": [],
        "return": [],
        "reward_per_step": [],
    }
    for traj_pos, traj in enumerate(trajectories):
        rewards = []
        for step in range(len(traj['obs'])):
            if benchmark_name == "metaworld":
                result = reward_function(action=traj['actions'][step], obs=traj['obs'][step], goal_position=traj['self.goal_position'][step])
            elif benchmark_name == "maniskill":
                ac = traj['actions'][step]
                ob, result, done, info = env.step(ac)
            else:
                raise RuntimeError("benchmark name error!")
            if isinstance(result, tuple):
                reward, reward_dict = result
            else:
                reward = result
            rewards.append(reward)
        total_reward = np.sum(rewards)
        if traj['infos'][-1]['success']:
            success_trajs["pos"].append(traj_pos)
            success_trajs["traj"].append(traj)
            success_trajs["return"].append(total_reward)
            success_trajs["reward_per_step"].append(total_reward / len(traj['obs']))
        else:
            failure_trajs["pos"].append(traj_pos)
            failure_trajs["traj"].append(traj)
            failure_trajs["return"].append(total_reward)
            failure_trajs["reward_per_step"].append(total_reward / len(traj['obs']))

    if len(failure_trajs["pos"]) == 0 or len(success_trajs["pos"]) == 0:
        correct_count = len(success_trajs["pos"])
        correct_rate = 1.0
        pass_flag = correct_rate >= threshold
        feedback = f"all success or fail! success traj num={len(success_trajs['pos'])}, fail traj num={len(failure_trajs['pos'])}"
    else:
        failure_max_r = max(failure_trajs["reward_per_step"])
        failure_max_index = failure_trajs["reward_per_step"].index(failure_max_r)
        correct_count = 0
        for r in success_trajs["reward_per_step"]:
            if r > failure_max_r:
                correct_count += 1
        
        correct_rate = correct_count / len(success_trajs["pos"])
        error_count = len(success_trajs["pos"]) - correct_count
        pass_flag = correct_rate >= threshold
        if pass_flag:
            feedback = f"check pass! success traj num={len(success_trajs['pos'])}, fail traj num={len(failure_trajs['pos'])}, correct_count={correct_count}"
        else:
            feedback = (
                "We collected several trajectories where some agents successfully solved the task, and others failed.\n"
                f"The average reward per step on a successful trajectory is expected to be higher than that on a failed trajectory.\n"
                "Using the reward function you provided to calculate the average reward per step for each trajectory, "
                f"We found {len(success_trajs['pos'])} successful trajectories, {error_count} of which have a smaller average reward per step than some of the failed trajectories.\n"
                f"This means that the reward function you provided is {correct_rate*100:.2f}% accurate in distinguishing successful trajectories. "
                f"This indicates a flaw in the design of the reward function. You need to revise the reward function to ensure that the average reward per step for most successful trajectories is higher than that of the failed trajectories.\n"
                )
            success_min_r = min(success_trajs["reward_per_step"])
            success_min_index = success_trajs["reward_per_step"].index(success_min_r)
            feedback_list = convert_traj_to_str(trajectory_list=[success_trajs["traj"][success_min_index],
                                                                 failure_trajs["traj"][failure_max_index]],
                                                mapping_dicts=mapping_dicts,
                                                log_interval=log_interval
                                                )
            feedback += f"For example, this is a trajectory where the agent successfully solved the task, with a return of {success_trajs['return'][success_min_index]:.1f}, a length of {len(success_trajs['traj'][success_min_index]['obs'])}, and an average reward per step of {success_min_r:.1f}:\n"
            feedback = feedback + feedback_list[0] + "\n"
            feedback += f"However, the following shows a trajectory where the agent failed to solve the task, with a return of {failure_trajs['return'][failure_max_index]:.1f}, a length of {len(failure_trajs['traj'][failure_max_index]['obs'])}, and an average reward per step of {failure_max_r:.1f}:\n"
            feedback = feedback + feedback_list[1] + "\n"

    return correct_count, len(success_trajs["pos"]), correct_rate, pass_flag, feedback

class Logger(object):
    def __init__(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.stderr = sys.stderr

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def write_error(self, message):
        self.stderr.write(message)
        self.log.write(message)

    def write_warning(self, message):
        warning_message = f"WARNING: {message}\n"
        self.terminal.write(warning_message)
        self.log.write(warning_message)