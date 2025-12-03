#!/usr/bin/env python3
"""
Example client for Wall-X model server with sync support.

This script demonstrates how to connect to a Wall-X server and request
action predictions from observations in both sync and async contexts.
"""
import sys
sys.path.insert(0, "/home/bozhi/Desktop/wall-x")

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


import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import threading
from quadrotor_msgs.msg import PositionCommand, GoalSet
import subprocess


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


               # ROS节点初始化
        rospy.init_node('uav_policy_node', anonymous=True)
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 存储最新的状态和图像
        self.current_state = None
        self.current_image = None
        self.state_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.first_image = None
        self.first_image_received = False
        self.count = 0
        self.save_count = 0  # 添加保存计数器
        self.image_save_count = 0  # 添加图像保存计数器
        self.last_state = [0, 0, 1, 0, 0, 0]
        self.last_inference_time = None  # 添加上次推理时间记录
        self.inference_timeout = 10.0  # 设置推理超时时间（秒），可以根据需要调整
        
        # 创建图像保存目录
        self.image_save_dir = 'simlulate/saved_images'
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        # 创建模型输入图像保存目录
        self.model_input_dir = 'simlulate/model_input_images'
        os.makedirs(self.model_input_dir, exist_ok=True)
        
        # 订阅无人机状态和图像话题
        self.odom_sub = rospy.Subscriber('/drone_0_odom_visualization/pose',    #/drone_0_odom_visualization/pose是无人机在unity里的世界坐标。需要交换YZ，四元数转欧拉角
                                        PoseStamped, 
                                        self.odom_callback,
                                        queue_size=1)
        
        self.image_sub = rospy.Subscriber('/camera0/color/image/compressed', #/camera0/color/image/compressed是前置摄像头
                                         CompressedImage, 
                                         self.image_callback,
                                         queue_size=1)
        
        # 发布无人机动作
        self.action_pub = rospy.Publisher('/uav_actions', 
                                         PoseStamped, 
                                         queue_size=1)

        self.initial_state = None
        # 初始化WebSocket客户端
        self.client = None
        self.prompt = None
        
        # 轨迹存储
        self.original_trajectory = []
        self.inferred_trajectory = []

        # 从参数服务器获取配置
        self.host = rospy.get_param('~host', '127.0.0.1')  # 修改默认host
        self.port = rospy.get_param('~port', 8000)
        self.replan_steps = rospy.get_param('~replan_steps', 10)
        self.prompt = rospy.get_param('~prompt', 'Pick up the cup on the bookshelf and put it on the table at the back left. Catch: cup. Put: table.')

        if self.prompt is None:
            self.prompt = "无人机实时轨迹控制"  # 默认提示
            logging.warning("无指令")
        print(f"当前的指令为: {self.prompt}")

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




    # ============ ros部分 ============

    def odom_callback(self, msg):
        """处理无人机状态回调"""
        # print("odom_callback")  # 先确保终端能看到这行

        with self.state_lock:
            # PoseStamped: 直接 msg.pose
            position = msg.pose.position
            orientation = msg.pose.orientation

            roll, pitch, yaw = self.quaternion_to_euler(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            self.current_state = np.array([
                position.x, position.y, position.z,
                roll, pitch, yaw
            ])
            if self.initial_state is None:
                self.initial_state = np.array([
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    roll, pitch, yaw
                ])
                print("initial_state:", self.initial_state)
            self.current_state = self.current_state - self.initial_state

    
    def image_callback(self, msg):
        """处理图像回调"""
        try:
            with self.image_lock:
                # 将压缩图像转换为OpenCV格式
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # 保存接收到的图像
                timestamp = rospy.Time.now()
                # image_filename = f"camera_image_{self.image_save_count:06d}_{timestamp.secs}_{timestamp.nsecs}.jpg"
                # image_path = os.path.join(self.image_save_dir, image_filename)
                # cv2.imwrite(image_path, cv_image)
                
                # 如果是第一帧图像，保存为固定图像
                if not self.first_image_received:
                    # 将图像resize到256x256用于推理
                    self.first_image = cv_image.copy()
                    self.count = self.count + 1
                    if self.count == 3:
                        self.first_image_received = True
                        # 保存第一帧图像作为固定主图像
                        first_image_path = os.path.join(self.image_save_dir, "first_main_image.jpg")
                        cv2.imwrite(first_image_path, self.first_image)
                        logging.info(f"固定主图像已保存: {first_image_path} (256x256)")
                

                self.current_image = cv_image.copy()
                
                # 增加图像保存计数器
                self.image_save_count += 1
                
                # 定期输出保存信息（每100张图像输出一次）
                if self.image_save_count % 100 == 0:
                    logging.info(f"已保存 {self.image_save_count} 张图像")
                
        except Exception as e:
            logging.error(f"图像处理错误: {e}")

    def quaternion_to_euler(self, x, y, z, w):
        """将四元数转换为欧拉角 (roll, pitch, yaw)"""
        # 这里实现了四元数到欧拉角的转换
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        return roll, pitch, yaw
    
    def prepare_image_for_inference(self, image):
        """准备推理用的图像，确保尺寸为256x256"""
        if image is None:
            return None
        
        # 确保图像是256x256
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        return image
    
    def publish_action(self, action):
        """发布动作到ROS话题"""

        # print("action_pub:", action)
        if len(action) < 5:
            logging.error("动作步数不足5")
            return
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        # print("pose_msg:", pose_msg)

        step = 3
        # 设置位置
        self.last_state = action[step]
        pose_msg.pose.position.x = action[step][0]+self.initial_state[0]
        pose_msg.pose.position.y = action[step][1]+self.initial_state[1]
        pose_msg.pose.position.z = action[step][2]+self.initial_state[2]

        # print("pose_msg:", pose_msg)

        # 将欧拉角转换为四元数
        qx, qy, qz, qw = self.euler_to_quaternion(action[step][3]+self.initial_state[3], action[step][4]+self.initial_state[4], action[step][5]+self.initial_state[5])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.action_pub.publish(pose_msg)

        rospy.sleep(5.0)

        yaw = action[step][5] + self.initial_state[5]

        subprocess.call(["/home/bozhi/Desktop/DataCollect/pub_yaw.sh", str(yaw)])

        # self.rotate_yaw_smoothly(
        #     x=pose_msg.pose.position.x,
        #     y=pose_msg.pose.position.y,
        #     z=pose_msg.pose.position.z,
        #     target_yaw=yaw,
        #     max_step_angle=0.3,   # 和原 shell 脚本一样
        #     publish_frequency=10.0
        # )
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """将欧拉角转换为四元数"""
        # 这里实现了欧拉角到四元数的转换
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw

    def run_inference(self):
        """执行推理循环"""
        rate = rospy.Rate(20)  # 10Hz
        # print("1")
        while not rospy.is_shutdown():
            # 检查是否有有效的状态和图像
            with self.state_lock, self.image_lock:
                # print("2")
                # print("self.current_state:",self.current_state)
                # print("self.current_image",self.current_image)
                # print("self.first_image_received:", self.first_image_received)
                if (self.current_state is None or 
                    self.current_image is None or 
                    not self.first_image_received):
                    # print("4")
                    rate.sleep()
                    continue
                
                state = self.current_state.copy()
                image = self.first_image.copy()  # 使用第一帧作为固定图像 (已经是256x256)
                current_image = self.current_image.copy()  # 使用实时图像 (已经是256x256)
                
                # 确保图像尺寸为256x256
                # image = self.prepare_image_for_inference(image)
                # wrist_image = self.prepare_image_for_inference(wrist_image)
                
                if image is None or current_image is None:
                    rate.sleep()
                    # print("3")
                    continue
            # print("state")
            # print(state)
            # print("Last_state")
            # print(self.last_state)
            # print("distance:")
            distance = np.linalg.norm(state[0:2] - self.last_state[0:2], axis=0)
            # print(distance)
            # print(f"Image shape: {image.shape}, Wrist image shape: {wrist_image.shape}")
            
            # 检查是否满足推理条件：位置移动距离 OR 时间超时
            current_time = rospy.Time.now()
            should_infer = False
            
            # 条件1：位置移动距离小于阈值（原逻辑，表示已到达目标点附近）
            if distance < 0.01:
                should_infer = True
                inference_reason = f"位置距离满足条件: {distance:.4f} < 0.01"
            
            # 条件2：时间超时（新增逻辑）
            elif self.last_inference_time is not None:
                time_since_last_inference = (current_time - self.last_inference_time).to_sec()
                if time_since_last_inference > self.inference_timeout:
                    should_infer = True
                    inference_reason = f"时间超时: {time_since_last_inference:.2f}s > {self.inference_timeout}s"
                else:
                    print(f"等待中... 距离: {distance:.4f}, 时间间隔: {time_since_last_inference:.2f}s")
            else:
                # 第一次推理
                should_infer = True
                inference_reason = "首次推理"
            
            # print("should_infer:", should_infer)
            if should_infer:
                try:
                    print(f"开始推理 - {inference_reason}")
                    
                    # 保存输入到模型的图像
                    timestamp = rospy.Time.now()
                    # 保存固定主图像 (256x256)
                    main_image_filename = f"inference_main_{self.save_count:04d}_{timestamp.secs}.jpg"
                    main_image_path = os.path.join(self.model_input_dir, main_image_filename)
                    cv2.imwrite(main_image_path, image)
                    
                    # 保存current图像 (256x256)
                    current_image_filename = f"inference_current_{self.save_count:04d}_{timestamp.secs}.jpg"
                    current_image_path = os.path.join(self.model_input_dir, current_image_filename)
                    cv2.imwrite(current_image_path, current_image)
                    # 保存状态信息到文本文件
                    state_filename = f"inference_state_{self.save_count:04d}_{timestamp.secs}.txt"
                    state_path = os.path.join(self.model_input_dir, state_filename)
                    with open(state_path, 'w') as f:
                        f.write(f"推理步骤: {self.save_count}\n")
                        f.write(f"时间戳: {timestamp.secs}.{timestamp.nsecs}\n")
                        f.write(f"推理原因: {inference_reason}\n")
                        f.write(f"位置距离: {distance:.6f}\n")
                        f.write(f"状态向量: {state.tolist()}\n")
                        f.write(f"提示: {self.prompt}\n")
                        f.write(f"主图像文件: {main_image_filename}\n")
                        f.write(f"手腕图像文件: {current_image_filename}\n")
                        f.write(f"图像尺寸: {image.shape}\n")
                    logging.info(f"模型输入图像已保存: {main_image_filename}, {current_image_filename}")

                    # state: (6,) -> (1, 6) tensor
                    state_tensor = torch.from_numpy(state).float()      # [6]
                    # 让 prepare_batch_sync 里自己加 batch 维，这里可以不 unsqueeze

                    # 图像：BGR (OpenCV) -> RGB, HWC -> CHW, 归一化到 0~1
                    front_tensor = torch.from_numpy(
                        cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                    ).permute(2, 0, 1).float() / 255.0      # [3, H, W]

                    first_front_tensor = torch.from_numpy(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ).permute(2, 0, 1).float() / 255.0      # [3, H, W]
                    
                    state_tensor = torch.from_numpy(state).float()

                    data = {
                        "state": state_tensor,
                        "video.front": front_tensor,
                        "video.front_first": first_front_tensor,
                        "task": self.prompt,
                    }

                    repo_id = "dzb/our_data_test"

                    obs = prepare_batch_sync(
                        data,
                        self.normalizer_action,
                        self.normalizer_propri,
                        dataset_names=[repo_id],
                    )


                    # pred_action shape: [pred_horizon, action_dim]

                    # 准备输入数据
                    # element = {
                    #     "observation/image": image,          # 固定使用第一帧图像
                    #     "observation/wrist_image": current_image,  # 使用实时图像
                    #     "observation/state": state,
                    #     "prompt": self.prompt
                    # }
                    # print(self.prompt)
                    # # 保存图像（创建输出目录）
                    # trail_dir = '/home/yang/VLA/Openpi/test/infer/trail'
                    # os.makedirs(trail_dir, exist_ok=True)
                    
                    # # 保存当前图像
                    # plt.figure()
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # plt.savefig(f'{trail_dir}/image{self.save_count}_main.png')
                    # plt.close()
                    
                    # plt.figure()
                    # plt.imshow(cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB))
                    # plt.savefig(f'{trail_dir}/image{self.save_count}_wrist.png')
                    # plt.close()
                    
                    # self.save_count += 1
                    
                    # 调用服务器进行推理
                    response = self.predict_sync(obs)
                    action_chunk = response["action"]
                    # print("action_chunk:", action_chunk)
                    # result = self.client.infer(element)
                    # action_chunk = result["actions"]
                    
                    # 更新推理时间记录
                    self.last_inference_time = current_time

                    state_filename = f"action_{self.save_count:04d}_{timestamp.secs}.txt"
                    state_path = os.path.join(self.model_input_dir, state_filename)
                    with open(state_path, 'w') as f:
                        f.write(f"action: {action_chunk}\n")
                    # print("action_chunk:", action_chunk)
                    # 发布第一个动作
                    # for i in range(5):
                    #     self.publish_action(action_chunk[i])
                        
                    #     self.inferred_trajectory.append(action_chunk[i])
                    #     if len(self.original_trajectory) == 0:
                    #         self.original_trajectory.append(state)
                    #     else:
                    #         self.original_trajectory.append(action_chunk[i])

                    #     rate.sleep()
                    print("action_chunk.shape:", action_chunk.shape)
                    if len(action_chunk) > 0:
                        self.publish_action(action_chunk[0])

                        # 存储轨迹用于可视化
                        # self.inferred_trajectory.append(action_chunk[0])
                        # if len(self.original_trajectory) == 0:
                        #     self.original_trajectory.append(state)
                        # else:
                        #     # 在实际应用中，这里应该获取真实的下一状态
                        #     # 由于我们是在线运行，无法获取真实下一状态，所以使用推理结果
                        #     self.original_trajectory.append(action_chunk[0])
                    
                except Exception as e:
                    logging.error(f"推理错误: {e}")
            rate.sleep()
    
    def visualize_trajectories(self):
        """可视化轨迹"""
        if len(self.original_trajectory) == 0 or len(self.inferred_trajectory) == 0:
            logging.warning("没有足够的轨迹数据来可视化")
            return
        
        original_traj = np.array(self.original_trajectory)
        inferred_traj = np.array(self.inferred_trajectory)
        
        # 确保两条轨迹长度一致
        min_len = min(len(original_traj), len(inferred_traj))
        original_traj = original_traj[:min_len]
        inferred_traj = inferred_traj[:min_len]
        
        # 计算偏差
        pos_mae, rot_mae, pos_errors = self.calculate_trajectory_deviation(original_traj, inferred_traj)
        
        print("\n--- 轨迹偏差分析 ---")
        print(f"平均绝对位置误差 (Positional MAE): {pos_mae:.4f}")
        print(f"平均绝对姿态误差 (Rotational MAE): {rot_mae:.4f}")
        print("\n每个点的具体位置误差:")
        for i, err in enumerate(pos_errors):
            print(f"  点 {i}: {err:.4f}")
        print("------------------------\n")
        
        # 可视化
        self.visualize_trajectories_3d(original_traj, inferred_traj)
        self.visualize_trajectories_2d_topdown(original_traj, inferred_traj)
    
    def calculate_trajectory_deviation(self, traj1, traj2):
        """计算两条轨迹的位置和姿态偏差"""
        pos1, rot1 = traj1[:, :3], traj1[:, 3:]
        pos2, rot2 = traj2[:, :3], traj2[:, 3:]

        positional_errors = np.linalg.norm(pos1 - pos2, axis=1)
        positional_mae = np.mean(positional_errors)

        rotational_errors = np.abs(rot1 - rot2)
        rotational_mae = np.mean(rotational_errors)

        return positional_mae, rotational_mae, positional_errors
    
    def visualize_trajectories_3d(self, traj1, traj2):
        """3D轨迹可视化"""
        pos1 = traj1[:, :3]
        pos2 = traj2[:, :3]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(pos1[:, 1], pos1[:, 0], pos1[:, 2], 'o-', label='Original Trajectory', color='blue')
        ax.plot(pos2[:, 1], pos2[:, 0], pos2[:, 2], 'o-', label='Inferred Trajectory', color='red')

        for i in range(len(pos1)):
            ax.plot([pos1[i, 1], pos2[i, 1]], [pos1[i, 0], pos2[i, 0]], [pos1[i, 2], pos2[i, 2]],
                    '--', color='gray', linewidth=0.8)

        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        # plt.show()
        plt.savefig('/home/yang/VLA/Openpi/test/infer/trail/traj3D_output.png')
        plt.close()
    
    def visualize_trajectories_2d_topdown(self, traj1, traj2):
        """2D俯视图轨迹可视化"""
        pos1 = traj1[:, :2]
        pos2 = traj2[:, :2]

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.plot(pos1[:, 1], pos1[:, 0], 'o-', label='Original Trajectory', color='blue')
        ax.plot(pos2[:, 1], pos2[:, 0], 'o-', label='Inferred Trajectory', color='red')

        for i in range(len(pos1)):
            ax.plot([pos1[i, 1], pos2[i, 1]], [pos1[i, 0], pos2[i, 0]],
                    '--', color='gray', linewidth=0.8)

        ax.set_xlabel('Y (m, Left/Right)')
        ax.set_ylabel('X (m, Forward/Backward)')
        ax.set_title('Trajectory Comparison (Top-down View)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        # plt.show()
        plt.savefig('/home/yang/VLA/Openpi/test/infer/trail/traj2D_output.png')
        plt.close()


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
    prompt = data["task"]

    state = data["state"].to("cuda")
    
    # print("prompt:", prompt,"  state:", state)
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
    """Load local dataset to get observations for inference."""
    repo_id = train_config["data"]["lerobot_config"]["repo_id"]

    meta_info = LeRobotDatasetMetadata(repo_id)
    dataset_fps = meta_info.fps
    # 改为 pred_horizon=5
    delta_timestamps = {
        "action": [t / dataset_fps for t in range(5)],
    }
    dataset = LeRobotDataset(
        repo_id,
        episodes=[0],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    return dataset, repo_id


def main_sync(args):
    """Synchronous version of main function."""

    # Create client and connect
    client = WallXClient(args.config_path, uri=args.uri)
    client.connect_sync()

    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))

    # 在一个单独的线程中运行推理循环
    inference_thread = threading.Thread(target=client.run_inference)
    inference_thread.start()    
        
    # 主线程运行ROS spin
    rospy.spin()
    
    # 等待推理线程结束
    inference_thread.join()
    
    # 可视化轨迹
    client.visualize_trajectories()


    # # Synchronous processing
    # for idx, data in enumerate(dataset):
    #     if idx % args.pred_horizon == 0 and idx + args.pred_horizon < total_frames:
            # print(f"Processing frame {idx}/{total_frames}")
            # obs = prepare_batch_sync(
            #     data,
            #     client.normalizer_action,
            #     client.normalizer_propri,
            #     dataset_names=[repo_id],
            # )
            # response = client.predict_sync(obs)
            # pred_action = response["action"]
            # # pred_action shape: [pred_horizon, action_dim]
            # pred_traj[idx : idx + args.pred_horizon] = pred_action
            # gt_traj[idx : idx + args.pred_horizon] = data["action"].cpu().numpy()


            # gt_traj[idx : idx + args.pred_horizon] = data["actions"].cpu().numpy()

    # # Draw plot
    # timesteps = gt_traj.shape[0]
    # fig, axs = plt.subplots(
    #     args.action_dim, 1, figsize=(15, 5 * args.action_dim), sharex=True
    # )
    # fig.suptitle("Action Comparison: Server Prediction vs Ground Truth", fontsize=16)

    # for i in range(args.action_dim):
    #     axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth", marker='o', markersize=2)
    #     axs[i].plot(range(timesteps), pred_traj[:, i], label="Server Prediction", marker='s', markersize=2)
    #     axs[i].set_ylabel(f"Action Dim {i+1}")
    #     axs[i].legend()
    #     axs[i].grid(True, alpha=0.3)

    # axs[-1].set_xlabel("Timestep")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # os.makedirs(args.save_dir, exist_ok=True)
    # save_path = os.path.join(args.save_dir, "server_prediction_comparison.png")
    # plt.savefig(save_path, dpi=150)
    # print(f"Saved plot to {save_path}")
    # plt.close()

    # # Close connection
    client.close_sync()


async def main(args):
    """Asynchronous version of main function."""
    client = WallXClient(args.config_path, uri=args.uri)
    await client.connect()
    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))

    for idx, data in enumerate(dataset):
        if idx % args.pred_horizon == 0 and idx + args.pred_horizon < total_frames:
            print(f"Processing frame {idx}/{total_frames}")
            obs = prepare_batch_sync(
                data,
                client.normalizer_action,
                client.normalizer_propri,
                dataset_names=[repo_id],
            )
            response = await client.predict(obs)
            pred_action = response["action"]
            pred_traj[idx : idx + args.pred_horizon] = pred_action
            # gt_traj[idx : idx + args.pred_horizon] = data["            data["actions"] = data["action"]"].cpu().numpy()
            gt_traj[idx : idx + args.pred_horizon] = data["action"].cpu().numpy()


    timesteps = gt_traj.shape[0]

    fig, axs = plt.subplots(
        args.action_dim, 1, figsize=(15, 5 * args.action_dim), sharex=True
    )
    fig.suptitle("Action Comparison: Server Prediction vs Ground Truth", fontsize=16)

    for i in range(args.action_dim):
        axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth", marker='o', markersize=2)
        axs[i].plot(range(timesteps), pred_traj[:, i], label="Server Prediction", marker='s', markersize=2)
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "server_prediction_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()

    await client.close()


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
        default="/home/bozhi/Desktop/wall-x/workspace/lerobot_example/UAV_test/wall-oss_fast-withMOE/config_qact.yml",
        help="Train config path",
    )
    parser.add_argument(
        "--save_dir",
        default="/home/bozhi/Desktop/wall-x/client_results",
        help="Save directory for results",
    )
    args = parser.parse_args()

    # Synchronous mode (推荐用这个)
    main_sync(args)

    # Asynchronous mode (可选)
    # asyncio.run(main(args))