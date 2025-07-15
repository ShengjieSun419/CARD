import os, argparse, glob
import sys
import json
import pickle
import traceback
from datetime import datetime
from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import gym
import mani_skill2.envs
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.sapien_utils import (check_actor_static, get_entity_by_name)
import sapien.core as sapien
import trimesh
from scipy.spatial.distance import cdist
import scipy
import scipy.spatial
import scipy.spatial.distance
from scipy.spatial import distance as sdist
from code_generation.self_reflection.utils import (set_seed_everywhere,
                                                   get_learning_result_feedback_simple,
                                                   collect_trajectories,
                                                   get_learning_result_mean,
                                                   trajectories_return_check,
                                                   get_eval_feedback,
                                                   SubRewardLog,
                                                   print_with_color,
                                                   convert_function_external,
                                                   Logger)
import torch
import numpy as np
from wandb.integration.sb3 import WandbCallback
import wandb

from code_generation.self_reflection.utils import INFO_PREFIX

LLM_RESPONSE="llm_response_code"
RL_QUERY="rl_query_code"
CHECKPOINT="checkpoint"

# environment list
franka_list = ["LiftCube-v0", "PickCube-v0", "StackCube-v0", "TurnFaucet-v0"]
mobile_list = ["OpenCabinetDoor-v1", "OpenCabinetDrawer-v1", "PushChair-v1"]


instruction_mapping = {
    "LiftCube-v0": "Pick up cube A and lift it up by 0.2 meter.",
    "PickCube-v0": "Pick up cube A and move it to the 3D goal position.",
    "TurnFaucet-v0": "Turn on a faucet by rotating its handle. The task is finished when qpos of faucet handle is larger than target qpos.",
    "OpenCabinetDoor-v1": "A single-arm mobile robot needs to open a cabinet door. The task is finished when qpos of cabinet door is larger than target qpos.",
    "OpenCabinetDrawer-v1": "A single-arm mobile robot needs to open a cabinet drawer. The task is finished when qpos of cabinet drawer is larger than target qpos.",
    "PushChair-v1": "A dual-arm mobile robot needs to push a swivel chair to a target location on the ground and prevent it from falling over.",
}

mapping_dicts_mapping = {
    "LiftCube-v0": {
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose" : "self.tcp.pose",
        "self.robot.check_grasp" : "self.agent.check_grasp",
        "self.cube_half_size" : "0.02",
        "self.robot.qpos" : "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel" : "self.agent.robot.get_qvel()[:-2]",
    },
    "PickCube-v0": {
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose" : "self.tcp.pose",
        "self.robot.check_grasp" : "self.agent.check_grasp",
        "self.goal_position" : "self.goal_pos",
        "self.cube_half_size" : "0.02",
        "self.robot.qpos" : "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel" : "self.agent.robot.get_qvel()[:-2]",
    },
    "TurnFaucet-v0": {
        "self.faucet.handle.target_qpos": "self.target_angle",
        "self.faucet.handle.qpos": "self.current_angle",
        "self.faucet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)",
        "self.robot.lfinger.get_world_pcd()": "transform_points(self.lfinger.pose.to_transformation_matrix(), self.lfinger_pcd)",
        "self.robot.rfinger.get_world_pcd()": "transform_points(self.rfinger.pose.to_transformation_matrix(), self.rfinger_pcd)",
        "self.faucet.handle.get_local_pcd()": "self.target_link_pcd",
        "self.robot.lfinger.get_local_pcd()": "self.lfinger_pcd",
        "self.robot.rfinger.get_local_pcd()": "self.rfinger_pcd",
        "self.robot.lfinger": "self.lfinger",
        "self.robot.rfinger": "self.rfinger",
        "self.faucet.handle": "self.target_link",
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
    },
    "OpenCabinetDoor-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
    },
    "OpenCabinetDrawer-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
    },
    "PushChair-v1": {
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.chair.get_pcd()": "self.env.env._get_chair_pcd()",
        "self.chair.check_static()": "self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2)",
        "self.chair": "self.root_link",
    },
}

class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, task_name, env, max_episode_steps: int, is_eval: bool, use_ground_truth_reward: bool, exp_name) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps
        self.is_eval = is_eval
        self.use_ground_truth_reward = use_ground_truth_reward
        self.task_name = task_name
        self.exp_name = exp_name
        # get package
        transform_points
        check_actor_static
        get_entity_by_name
        sapien
        # over

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs
    
    def compute_dense_reward(self, action):
        assert (0)

    def step(self, action):
        ob, rew, done, info = super().step(action)
        task_map = mapping_dicts_mapping[self.task_name]
        for key, value in task_map.items():
            if isinstance(eval(value), (int, float, bool, str, np.integer, np.floating, np.bool_, np.ndarray)):
                if isinstance(eval(value), (trimesh.caching.TrackedArray)) or key == "self.robot.get_ee_coords()" or key == "self.chair.get_pcd()":
                    pass
                else:
                    info[INFO_PREFIX+key] = eval(value)
            elif key == "self.robot.ee_pose":
                info[INFO_PREFIX+key+".p"] = eval(value+".p")
                info[INFO_PREFIX+key+".q"] = eval(value+".q")
            else:
                pass
        if not self.use_ground_truth_reward:
            result = self.compute_dense_reward(action)
            if isinstance(result, tuple):
                rew, rew_dict = result
                for key, value in rew_dict.items():
                    info["sub_reward/"+key] = value
            else:
                rew = result
        info["ground_truth_reward"] = rew
        self._elapsed_steps += 1
        if not self.is_eval:
            if self._elapsed_steps >= self._max_episode_steps:
                done = True
                info["TimeLimit.truncated"] = True
            else:
                done = False
                info["TimeLimit.truncated"] = False
        return ob, rew, done, info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        if info["success"]:
            done = True
        return ob, rew, done, info


def make_env(env_id, seed, max_episode_steps: int = None, record_dir: str = None, is_eval: bool = False,
             use_ground_truth_reward: bool = True, control_mode=None, exp_name=None):
    def _init() -> gym.Env:
        import mani_skill2.envs
        if env_id == "OpenCabinetDoor-v1":
            env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode, model_ids=['1000'],
                           fixed_target_link_idx=1)
        elif env_id == "OpenCabinetDrawer-v1":
            env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode, model_ids=['1000'],
                           fixed_target_link_idx=1)
        elif env_id == "PushChair-v1":
            env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode, model_ids=['3001'])
        elif env_id == "TurnFaucet-v0":
            env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode, model_ids=['5029'])
        else:
            env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode, )
        if not is_eval and max_episode_steps is not None:
            env = ContinuousTaskWrapper(env_id, env, max_episode_steps, is_eval=is_eval, use_ground_truth_reward=use_ground_truth_reward, exp_name=exp_name)
        elif is_eval and record_dir is not None:
            env = ContinuousTaskWrapper(env_id, env, None, is_eval=is_eval, use_ground_truth_reward=use_ground_truth_reward, exp_name=exp_name)
            env = SuccessInfoWrapper(env)
        else:
            raise RuntimeError("Environmental parameter conflicts!")
        return env

    return _init

def load_reward_function(code: str):
    namespace = {
        **globals(),
        "np": np,
        "Dict": Dict,
        "Tuple": Tuple,
        "transform_points": transform_points,
        "check_actor_static": check_actor_static,
        "get_entity_by_name": get_entity_by_name,
        "sapien": sapien,
        "scipy": scipy,
        "cdist": cdist,
        "sdist": sdist,
        "distance": scipy.spatial.distance,
    }
    exec(code, namespace)
    new_function = namespace['compute_dense_reward']
    ContinuousTaskWrapper.compute_dense_reward = new_function

def env_is_valid(reward_code_str, args):
    try:
        train_env = make_env(args.task_name, seed=args.seed, max_episode_steps=args.max_episode_steps, is_eval=False, use_ground_truth_reward=False, control_mode=args.control_mode, exp_name=args.exp_name)()
        eval_env = make_env(args.task_name, seed=args.seed, record_dir="logs/videos", is_eval=True, use_ground_truth_reward=False, control_mode=args.control_mode, exp_name=args.exp_name)()
        external_reward_code = convert_function_external(reward_code_str, benchmark_name=args.benchmark_name)
        for env in (train_env, eval_env):
            obs = env.reset()
            action = env.action_space.sample()
            # check internal call
            load_reward_function(code=reward_code_str)
            obs, reward, done, info = env.step(action)
            assert isinstance(reward, float), "reward is not float type"
        train_env.close()
        eval_env.close()
    except Exception as e:
        return False, traceback.format_exc()
    else:
        return True, ""

def train_and_eval(args, seed, step, use_ground_truth_reward: bool, specific_code, specific_code_path: str, load_model):
    log_path_root = os.path.join(os.path.dirname(specific_code_path), f"seed_{seed}")
    os.makedirs(log_path_root, exist_ok=True)
    if not use_ground_truth_reward:
        load_reward_function(code=specific_code)
    else:
        pass


    if args.use_wandb:
        # initialize wandb
        run = wandb.init(project=args.benchmark_name, entity=args.entity,
                        dir=log_path_root,
                        group=args.task_name,
                        config=vars(args),
                        name="-".join([args.llm_model_name, "zeroshot", args.task_name, args.exp_name, f"step_{step}", f"seed_{seed}"]),
                        mode='offline' if args.offline else 'online',
                        sync_tensorboard=True,
                        save_code=True)
        
        # create a dir on wandb to store the codes, copy these to wandb
        if specific_code_path is not None and glob.glob(os.path.join(os.path.dirname(specific_code_path), "*.py")):
            os.makedirs(f"{wandb.run.dir}/reward_codes/{run.id}", exist_ok=True)
            os.system(f"cp -r {os.path.dirname(specific_code_path)}/*.py {wandb.run.dir}/reward_codes/{run.id}")

    # set up eval environment
    eval_env = SubprocVecEnv([make_env(args.task_name, seed=seed, record_dir="logs/videos", is_eval=True,
                                       use_ground_truth_reward=use_ground_truth_reward, control_mode=args.control_mode, exp_name=args.exp_name) for i in range(args.eval_num)])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(seed)
    eval_env.reset()

    # set up extra info environment
    extra_eval_env = make_env(args.task_name, seed=seed, record_dir="logs/videos", is_eval=True,
                                       use_ground_truth_reward=use_ground_truth_reward, control_mode=args.control_mode, exp_name=args.exp_name)()
    extra_eval_env.seed(seed)
    extra_eval_env.reset()
    # set up train environment
    env = SubprocVecEnv([make_env(args.task_name, seed=seed, max_episode_steps=args.max_episode_steps, is_eval=False,
                                  use_ground_truth_reward=use_ground_truth_reward, control_mode=args.control_mode, exp_name=args.exp_name) for i in range(args.train_num)])
    env = VecMonitor(env)
    env.seed(seed)
    env.reset()

    # set up callback
    eval_log_path = os.path.join(log_path_root, "tensorboard_log/eval")
    os.makedirs(eval_log_path, exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_path, log_path=eval_log_path,
                                 eval_freq=args.eval_freq // args.train_num, deterministic=True, render=False,
                                 n_eval_episodes=args.n_eval_episodes)
    log_file_name="extra_info.npz"
    extra_info_callback = SubRewardLog(eval_env=extra_eval_env,
                                       eval_freq=args.eval_freq // args.train_num,
                                       log_path=eval_log_path,
                                       log_file_name=log_file_name,
                                       n_eval_episodes=args.n_eval_episodes,
                                       verbose=1)
    using_cuda = True if torch.cuda.is_available() else False
    set_seed_everywhere(seed, using_cuda)

    if load_model:
        resume_path = os.path.join(log_path_root, "tensorboard_log", "latest_model.zip")
        model = SAC.load(resume_path, env=env)
    else:
        # set up sac algorithm
        if args.algorithm == "sac":
            policy_kwargs = dict(net_arch=[256, 256])
            model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, batch_size=1024, gamma=0.95,
                        learning_starts=4000, ent_coef='auto_0.2', train_freq=8, gradient_steps=4,
                        seed=seed)
        elif args.algorithm == "ppo":
            policy_kwargs = dict(net_arch=[256, 256])
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=args.rollout_steps // args.train_num, batch_size=400, n_epochs=15, gamma=0.85, target_kl=0.05)
        else:
            model = None

        train_log_path = os.path.join(log_path_root, "tensorboard_log")
        os.makedirs(train_log_path, exist_ok=True)
        model.set_logger(configure(train_log_path, ["stdout", "csv", "tensorboard"]))
        callback_list = [eval_callback, extra_info_callback]
        if args.use_wandb:
            callback_list.append(WandbCallback(verbose=2))
        model.learn(args.train_max_steps, callback=callback_list)
        model.save(os.path.join(train_log_path, "latest_model"))
    # load evaluations.npz and add extra info
    learning_result = np.load(os.path.join(eval_log_path, "evaluations.npz"))
    learning_result = {key: learning_result[key] for key in learning_result.files}
    extra_info = np.load(os.path.join(eval_log_path, log_file_name))
    extra_info = {key: extra_info[key] for key in extra_info.files}
    learning_result.update(extra_info)
    # get trajectories, train env for same episode length
    check_traj_env = make_env(args.task_name, seed=seed, record_dir="logs/videos", is_eval=True,
                                       use_ground_truth_reward=use_ground_truth_reward, control_mode=args.control_mode, exp_name=args.exp_name)()
    check_traj_env.seed(seed)
    check_traj_env.reset()
    diff_len_env = make_env(args.task_name, seed=seed, record_dir="logs/videos", is_eval=True,
                                       use_ground_truth_reward=use_ground_truth_reward, control_mode=args.control_mode, exp_name=args.exp_name)()
    diff_len_env.seed(seed)
    diff_len_env.reset()
    trajectories_for_check = collect_trajectories(model=model, env=check_traj_env, mapping_dicts=mapping_dicts_mapping, trajectory_num=args.trajectory_num, benchmark_name=args.benchmark_name)
    diff_len_trajectories = collect_trajectories(model=model, env=diff_len_env, mapping_dicts=mapping_dicts_mapping, trajectory_num=args.n_eval_episodes, benchmark_name=args.benchmark_name)
    if args.use_wandb:
        run.finish()
        wandb.finish()
    env.close()
    eval_env.close()
    extra_eval_env.close()
    check_traj_env.close()
    diff_len_env.close()
    return learning_result, trajectories_for_check, diff_len_trajectories


def load_llm_response(args, llm_response_file_name):
    llm_ans_file = os.path.join(args.log_path, llm_response_file_name)
    try:
        with open(llm_ans_file, 'rb') as file:
            loaded_objects_dict = pickle.load(file)
    except Exception as e:
        print(f"{e}")
        return False, {}
    else:
        return True, loaded_objects_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="card")

    # to generate reward function code
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--llm_model_name', type=str, default="gpt-4-1106-preview")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--reflection_num', type=int, default=10)
    parser.add_argument('--algorithm', type=str, default=None)

    # checkpoint
    parser.add_argument('--checkpoint', action="store_true", default=False)
    parser.add_argument('--load_model', action="store_true", default=False)

    # to eval reward function code
    parser.add_argument('--train_num', type=int, default=8)
    parser.add_argument('--eval_num', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=16_000)
    parser.add_argument('--max_episode_steps', type=int, default=200)
    parser.add_argument('--rollout_steps', type=int, default=3200)
    parser.add_argument('--train_max_steps', type=int, default=8_000_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--benchmark_name', type=str, default="maniskill")
    parser.add_argument('--max_try_num', type=int, default=10)
    parser.add_argument('--baseline_reward_path', type=str, default=None)
    parser.add_argument('--trajectory_num', type=int, default=100)
    parser.add_argument('--train_seed_num', type=int, default=2)
    parser.add_argument('--n_eval_episodes', type=int, default=3)

    # trajectories return check
    parser.add_argument('--use_return_check', action="store_true")
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_point_num', type=int, default=5)
    parser.add_argument('--control_mode', type=str)

    # wandb
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--entity', type=str, default="Automated-Reward-Generation-with-Introspection")
    parser.add_argument('--offline', action="store_true")

    # change step
    parser.add_argument('--step', type=int)

    # ablation
    parser.add_argument('--exclude', type=str, default=None)
    parser.add_argument('--query_restart', type=bool, default=True)

    args = parser.parse_args()

    # log path to save result
    if args.log_path == None:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        args.log_path = f"logs/{args.benchmark_name}/{args.llm_model_name}/t={args.temperature}/zero-shot/SAC_{args.seed}/{args.task_name}/{args.exp_name}/exclude={args.exclude}/{formatted_time}"
    os.makedirs(args.log_path, exist_ok=True)

    args.control_mode = ""
    if args.task_name in franka_list + mobile_list:
        if args.task_name in franka_list:
            args.control_mode = "pd_ee_delta_pose"
        elif args.task_name in mobile_list:
            args.control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose"
        else:
            assert (0)
    else:
        print("Please specify a valid environment!")
        assert (0)
        
    if args.exp_name == "card":
        std_file = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.log")
        logger = Logger(filename=std_file)
        sys.stdout = logger
        sys.stderr = logger
        print("args is:")
        print(json.dumps(vars(args), indent=4))
        if args.step == 0:
            # 1.fist generation
            message_history = []
            trajectories_for_check_list = []
            query_file_path = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.pkl")
            with open(query_file_path, 'wb') as file:
                pickle.dump({
                    "rl_args": args,
                    "step": 0,
                    "message_history": message_history,
                    "trajectories_for_check_list": trajectories_for_check_list,
                    "feedback_str": "",
                }, file)
        else:
            # 2.step>0 introspection
            code_file = f"{LLM_RESPONSE}_{args.step-1}.pkl"
            load_success, loaded_objects_dict = load_llm_response(args=args, llm_response_file_name=code_file)
            if load_success:
                general_code_path = os.path.join(args.log_path, f"code_{args.step-1}", f"general_{args.step-1}.py")
                specific_code_path = os.path.join(args.log_path, f"code_{args.step-1}", f"specific_{args.step-1}.py")
                with open(general_code_path, "r") as f:
                    general_code = f.read()
                with open(specific_code_path, "r") as f:
                    specific_code = f.read()
                message_history = loaded_objects_dict["message_history"]
                trajectories_for_check_list = loaded_objects_dict["trajectories_for_check_list"]
                print(specific_code_path)
                print(specific_code)
            else:
                exit(0)

            feedback_str = ""
            # eval introspection
            correct_count = 0
            correct_rate = 1
            pass_flag = True
            check_feedback = ""
            if args.checkpoint:
                # load from checkpoint
                checkpoint_file = os.path.join(args.log_path, f"{CHECKPOINT}_{args.step}.pkl")
                with open(checkpoint_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                correct_count, correct_all, correct_rate, pass_flag, check_feedback = loaded_data["return_check_info"]
                learning_result, trajectories_for_check_need_update, diff_len_trajectories = loaded_data["train_info"]
                print("="*20 + f"{args.exp_name} load checkpoint successfully!" + "="*20)
                print(f"checkpoint path is: {checkpoint_file}" + "="*20)
            else:
                # save checkpoint
                if args.use_return_check:
                    # 2.1 trajectories return check
                    load_reward_function(code=specific_code)
                    reward_function = ContinuousTaskWrapper.compute_dense_reward
                    return_check_env = make_env(args.task_name, seed=args.seed, record_dir="logs/videos", is_eval=True, use_ground_truth_reward=False, control_mode=args.control_mode, exp_name=args.exp_name)()
                    return_check_env.seed(args.seed)
                    return_check_env.reset()
                    correct_count, correct_all, correct_rate, pass_flag, check_feedback = trajectories_return_check(
                                            reward_function=reward_function,
                                            trajectories=trajectories_for_check_list,
                                            mapping_dicts=mapping_dicts_mapping,
                                            threshold=args.threshold,
                                            log_interval=args.log_interval,
                                            benchmark_name=args.benchmark_name,
                                            env=return_check_env)
                    return_check_env.close()
                # 2.2 eval no matter pass_flag is True or False
                # train one seed in default
                learning_result, trajectories_for_check_need_update, diff_len_trajectories = train_and_eval(args=args,
                                                                seed=args.seed, step=args.step,
                                                                use_ground_truth_reward=False,
                                                                specific_code=specific_code,
                                                                specific_code_path=specific_code_path,
                                                                load_model=args.load_model)

                checkpoint_file = os.path.join(args.log_path, f"{CHECKPOINT}_{args.step}.pkl")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        "return_check_info": [correct_count, correct_all, correct_rate, pass_flag, check_feedback],
                        "train_info": [learning_result, trajectories_for_check_need_update, diff_len_trajectories]
                    }, f)
                    print("="*20 + f"{args.exp_name} checkpoint saved!" + "="*20)
                    print(f"load path is: {checkpoint_file}" + "="*20)

                # only for eval performance, not for feedback
                if args.train_seed_num > 1:
                    learning_result_list = [learning_result]
                    for seed in range(args.seed+1, args.seed+args.train_seed_num):
                        learning_result_more, _, _ = train_and_eval(args=args,
                                                                        seed=seed, step=args.step,
                                                                        use_ground_truth_reward=False,
                                                                        specific_code=specific_code,
                                                                        specific_code_path=specific_code_path,
                                                                        load_model=args.load_model)
                        learning_result_list.append(learning_result_more)
                    learning_result_mean = get_learning_result_mean(learning_result_list=learning_result_list,
                                                                    save_path=os.path.join(os.path.dirname(specific_code_path),
                                                                                    "evaluations_mean.npz"))
            # after load checkpoint or train
            learning_result_feedback = get_learning_result_feedback_simple(learning_result=learning_result, log_point_num=args.log_point_num)
            eval_feedback = get_eval_feedback(trajectory_list=diff_len_trajectories,
                                    mapping_dicts=mapping_dicts_mapping,
                                    log_interval=args.log_interval)
            if not args.use_return_check or pass_flag:
                # dont check or check pass
                if args.exclude == "process":
                    feedback_str += eval_feedback
                elif args.exclude == "trajectory":
                    feedback_str += learning_result_feedback
                else:
                    feedback_str += "1." + learning_result_feedback + "\n"
                    feedback_str += "2." + eval_feedback

                if not args.use_return_check:
                    with open(os.path.join(os.path.dirname(specific_code_path), "no check.txt"), "w") as file:
                        file.write("no check")
                if args.use_return_check and pass_flag:
                    with open(os.path.join(os.path.dirname(specific_code_path), "check pass.txt"), "w") as file:
                        file.write("check pass")
                # update traj only check pass
                trajectories_for_check_list.extend(trajectories_for_check_need_update)

            else:
                if args.exclude == "preference":
                    feedback_str += "1." + learning_result_feedback + "\n"
                    feedback_str += "2." + eval_feedback
                else:
                    feedback_str += check_feedback
                with open(os.path.join(os.path.dirname(specific_code_path), "check fail.txt"), "w") as file:
                    file.write("check fail")

            # save data
            query_file_path = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.pkl")
            with open(query_file_path, 'wb') as file:
                pickle.dump({
                    "rl_args": args,
                    "step": args.step,
                    "message_history": message_history,
                    "trajectories_for_check_list": trajectories_for_check_list,
                    "feedback_str": feedback_str,
                }, file)

    print(args.exp_name, "finished!")