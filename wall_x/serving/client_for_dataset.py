#!/usr/bin/env python3
"""
Example client for Wall-X model server with sync support.

This script demonstrates how to connect to a Wall-X server and request
action predictions from observations in both sync and async contexts.
"""
import sys
sys.path.insert(0, "/home/bozhi/Desktop/wall-x")

import re
import asyncio
import logging
from typing import Dict, List
import numpy as np
import threading
import yaml
import torch
import matplotlib.pyplot as plt
import os

from wall_x.model.action_head import Normalizer
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from wall_x.utils.constant import action_statistic_dof

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    print("Please install msgpack-numpy: pip install msgpack-numpy")
    exit(1)

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)


from wall_x.data.utils import (
    process_grounding_points,
    get_wallx_normal_text,
    replace_action_token,
    preprocesser_call,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallXClient:
    """Client for connecting to Wall-X model server."""

    def __init__(self, config_path: str, uri: str = "ws://localhost:8000"):
        """Initialize client.

        Args:
            uri: WebSocket URI of the server (e.g., ws://localhost:8000)
        """
        self.uri = uri
        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None

        with open(config_path, "r") as f:
            self.train_config = yaml.load(f, Loader=yaml.FullLoader)

        self.init_normalizer(self.train_config)

    async def connect(self):
        """Connect to the server and receive metadata."""
        logger.info(f"Connecting to {self.uri}...")
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )

        self.metadata = msgpack.unpackb(await self.websocket.recv())
        logger.info(f"Connected! Server metadata: {self.metadata}")

    async def predict(self, obs: Dict) -> Dict:
        """Get action prediction from observation.

        Args:
            obs: Observation dictionary containing:
                - 'image': Image array (H, W, C)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state

        Returns:
            Dictionary with:
                - 'action': Predicted action array
                - 'server_timing': Timing information
        """
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        await self.websocket.send(msgpack.packb(obs))
        response = msgpack.unpackb(await self.websocket.recv())
        return response

    async def close(self):
        """Close the connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")

    async def reset(self):
        """Reset the policy (if supported)."""
        pass

    # ============ Synchronous methods (using independent thread event loop) ============

    def _start_background_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        """Ensure background event loop is running."""
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(
                target=self._start_background_loop, daemon=True
            )
            self._thread.start()
            # Wait for loop to start
            import time

            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        """Run coroutine in background event loop."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def connect_sync(self):
        """Synchronously connect to server."""
        return self._run_async(self.connect())

    def norm_state(
        self,
        state: np.ndarray,
        dataset_names: List[str],
        state_mask: torch.Tensor = None,
    ) -> np.ndarray:
        """Normalize state."""
        return self.normalizer_propri.normalize_data(state, dataset_names, state_mask)

    def predict_sync(self, obs: Dict) -> Dict:
        """Synchronous prediction method.

        Args:
            obs: Observation dictionary

        Returns:
            Prediction result dictionary
        """
        return self._run_async(self.predict(obs))

    def close_sync(self):
        """Synchronously close connection."""
        result = self._run_async(self.close())
        # Stop event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return result

    def init_normalizer(self, train_config):
        customized_dof_config = train_config["customized_robot_config"][
            "customized_dof_config"
        ]
        customized_agent_pos_config = train_config["customized_robot_config"][
            "customized_agent_pos_config"
        ]
        Qwen2_5_VLMoEForAction._set_customized_config(train_config)

        self.normalizer_action = Normalizer(
            action_statistic_dof, customized_dof_config
        ).to("cuda")
        self.normalizer_propri = Normalizer(
            action_statistic_dof, customized_agent_pos_config
        ).to("cuda")

        print("Normalizer initialized")

def parse_task(task_str):
    # 匹配英文格式：Catch: xxx. Put: yyy
    match = re.search(r"(.*)Catch:\s*(.*)\.\s*Put:\s*(.*)", task_str)
    if match:
        instruction = match.group(1).strip()
        catch_target = match.group(2).strip()
        put_target = match.group(3).strip()
        return instruction, catch_target, put_target
    else:
        # 格式不匹配时返回空字符串
        return "", "", ""
def deal_instruction(instruction: str) -> str:
    """Deal with the instruction.

    Args:
        instruction: The instruction string.

    Returns:
        The processed instruction string.
    """
    grasp = False
    grasp_state = (
        "now you have already grasped the object"
        if grasp
        else "now you have not grasped the object yet")
    if not grasp:
        task, catch_target, put_target = parse_task(instruction)
        target = catch_target
        instruction_info = {"instruction": task}
        text_prompt = f"\nYou are performing a robotic manipulation task, {grasp_state}. If you believe the robot can now **grasp or place** the object, identify the {target} in the **front view** and output its bounding box in the format **[x1, y1, x2, y2]**. If you believe the robot still needs to **move closer to the target**, then **predict the robot's next actions**. \n"

    
    return text_prompt



def prepare_batch_sync(data, normalizer_action, normalizer_propri, dataset_names):
    """Synchronous version of prepare_batch."""
    # print("data_keys:", data.keys())
    # print("image1 shape:", data["video.front"].shape)
    image1 = (data["video.front"].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    # print("10")
    image2 = (
        (data["video.front_first"].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    )
    # print("9")
    # 添加调试信息
    # print(f"[DEBUG] image1 shape: {image1.shape}, dtype: {image1.dtype}")
    # print(f"[DEBUG] image2 shape: {image2.shape}, dtype: {image2.dtype}")
    # prompt = deal_instruction(data["task"])
    prompt = data["task"]
    state = data["state"].to("cuda")
    print("prompt:", prompt,"  state:", state)
    if state.dim() == 1:
        state = state.unsqueeze(0)

    state_mask = torch.zeros([1, 5, 20]).to("cuda")  # 改为 pred_horizon=5, state_dim=6
    state_mask[:, :, :6] = 1  # 将前 6 维设置为 1

    state = normalizer_propri.normalize_data(state, dataset_names, state_mask)
    state = state.cpu().numpy().astype(np.float32)
    # print("8")
    obs = {
        "face_view": image1,  # 改为与 server camera_key 匹配
        "first_face_view": image2,
        "prompt": prompt,
        "state": state,
        "dataset_names": dataset_names,
    }
    return obs

def init_serving_sample_dataset(train_config):
    repo_id = train_config["data"]["lerobot_config"]["repo_id"]

    meta_info = LeRobotDatasetMetadata(repo_id)
    dataset_fps = meta_info.fps
    delta_timestamps = {
        "action": [t / dataset_fps for t in range(5)],
    }
    dataset = LeRobotDataset(
        repo_id,
        episodes=range(18),
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    return dataset, repo_id


# ============ Synchronous version of main function ============


def main_sync(args):
    """Synchronous version of main function."""

    # Create client and connect
    client = WallXClient(args.config_path, uri=args.uri)
    client.connect_sync()

    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))
    import torch

    dof_mask = torch.ones([1, 5, 20]).to("cuda")
    dof_mask[:, :, args.action_dim :] = 0

    step = 0
    pred_horizon = args.pred_horizon
    pic = -1

    # Synchronous processing
    for idx, data in enumerate(dataset):
        # print("data[\"action\"]:", data["action"][0], "data[\"task\"]:", data["task"],"  idx:",idx)
        if (torch.equal(data['action'][0], data['action'][1]) and step!=0) or (idx == total_frames-1):
            # 只可视化前两个维度的 XY 轨迹并显示关键点编号
            timesteps = gt_traj.shape[0]

            import matplotlib.pyplot as plt
            import os


            fig = plt.figure(figsize=(10, 10))
            plt.title("XY Trajectory Comparison for lerobot", fontsize=16)

            save_dir = r"client_results"
            pic += 1
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"lerobot_xy_trajectory_points_{pic}.png")

            
            start_idx = 0
            plt.plot(
                gt_traj[start_idx:step, 0],
                gt_traj[start_idx:step, 1],
                "-o",
                label="Ground Truth",
                alpha=0.7,
                color="blue"
            )

            # Predicted trajectory
            plt.plot(
                pred_traj[start_idx:step, 0],
                pred_traj[start_idx:step, 1],
                "-o",
                label="Prediction",
                alpha=0.7,
                color="orange"
            )

            # 标出编号（step index）
            for i in range(start_idx, step):
                plt.text(
                    gt_traj[i, 0],
                    gt_traj[i, 1],
                    str(i),
                    fontsize=8,
                    color="blue",
                )
                plt.text(
                    pred_traj[i, 0],
                    pred_traj[i, 1],
                    str(i),
                    fontsize=8,
                    color="orange",
                )


            plt.xlabel("Action Dim 1 (X)")
            plt.ylabel("Action Dim 2 (Y)")
            plt.legend()
            plt.axis("equal")
            plt.tight_layout()
            instruction_text = data["task"]
            # 在图的整体坐标系上加说明文字
            if instruction_text:
                # 使用 fig.text，而不是 plt.text
                fig.text(
                    0.02,
                    0.02,
                    instruction_text,
                    fontsize=6,
                    color="black",
                    wrap=True,
                    ha="left",
                    va="bottom",
                )

            # 高分辨率保存
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Saved clearer XY trajectory plot to {save_path}")
            plt.close()
            step = 0
            gt_traj = torch.zeros((total_frames, args.action_dim)).cpu().numpy()
            pred_traj = torch.zeros((total_frames, args.action_dim)).cpu().numpy()
            continue
        if step % pred_horizon == 0 and step + pred_horizon < total_frames and step >=0:
            if step == 0 and abs(data['action'][0][0])+abs(data['action'][0][1])>0.05:
                continue
            # print("推理")
            obs = prepare_batch_sync(
                data,
                client.normalizer_action,
                client.normalizer_propri,
                dataset_names=[repo_id],
            )
            response = client.predict_sync(obs)
            pred_action = response["action"]
            if pred_action is None:
                # print("pred_action:",pred_action)
                continue
            pred_traj[step : step + args.pred_horizon] = pred_action
            gt_traj[step : step + args.pred_horizon] = data["action"]
            # print("gt_action:",data["action"])
        step +=1


    




# ============ Asynchronous version of main function (keep original functionality) ============


async def main(args):
    client = WallXClient(args.config_path, uri=args.uri)
    await client.connect()
    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    step = 0
    pred_horizon = args.pred_horizon
    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))

    for idx, data in enumerate(dataset):
        if (torch.equal(data['action'][0], data['action'][1]) and step!=0) or (idx == total_frames-1):
            timesteps = gt_traj.shape[0]

            fig, axs = plt.subplots(
                args.action_dim, 1, figsize=(15, 5 * args.action_dim), sharex=True
            )
            fig.suptitle("Action Comparison for lerobot", fontsize=16)

            for i in range(args.action_dim):
                axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth")
                axs[i].plot(range(timesteps), pred_traj[:, i], label="Prediction")
                axs[i].set_ylabel(f"Action Dim {i+1}")
                axs[i].legend()
                axs[i].grid(True)

            axs[-1].set_xlabel("Timestep")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "lerobot_comparison_serving.png")
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close()
            step = 0
            continue
        if step % pred_horizon == 0 and step + pred_horizon < total_frames and step >=0:

            obs = prepare_batch_sync(
                data,
                client.normalizer_action,
                client.normalizer_propri,
                dataset_names=[repo_id],
            )
            response = await client.predict(obs)
            pred_action = response["action"]
            # print(pred_action.shape)
            pred_traj[idx : idx + args.pred_horizon] = pred_action
            gt_traj[idx : idx + args.pred_horizon] = data["action"]
        step +=1
    


if __name__ == "__main__":
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Wall-X client for server testing")
    parser.add_argument(
        "--uri",
        default="ws://localhost:8000",
        help="Server WebSocket URI",
    )

    parser.add_argument(
        "--pred_horizon", 
        type=int, 
        default=5,  # 改为 5
        help="Prediction horizon",
    )
    parser.add_argument(
        "--action_dim", 
        type=int, 
        default=6,  # 改为 6
        help="Action dimension",
    )
    parser.add_argument(
        "--config_path",
        default="workspace/lerobot_example/UAV_test/wall-oss_fast-withMOE/config_qact.yml",
        help="Train config path",
    )
    parser.add_argument(
        "--save_dir",
        default="client_results",
        help="Save directory for results",
    )
    args = parser.parse_args()

    # Synchronous mode (推荐用这个)
    main_sync(args)

    # Asynchronous mode (可选)
    # asyncio.run(main(args))