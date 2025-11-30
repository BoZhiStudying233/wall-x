#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import sys
import os
import threading
import asyncio

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge

# ==== 新增：Wall-X WebSocket 客户端相关 ====
try:
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("Please install msgpack-numpy: pip install msgpack msgpack-numpy")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


class WallXWebsocketClient:
    """
    精简版 Wall-X WebSocket 客户端，用于与 server.py 交互。
    协议和你的 client.py 一致：
      - 连接后先收一条 metadata
      - 每次发送 obs(dict，经 msgpack 编码)，接收 response(dict)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"

        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None

        logging.info(f"WallXWebsocketClient connecting to {self.uri} ...")
        self.connect_sync()
        logging.info("WallXWebsocketClient connected.")

    # ---------- 异步原始接口 ----------

    async def _connect(self):
        """真正的异步连接方法，连接后接收一条 metadata。"""
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )
        # 第一次消息为 metadata
        self.metadata = msgpack.unpackb(await self.websocket.recv())
        logging.info(f"Connected! Server metadata: {self.metadata}")

    async def _predict(self, obs: dict) -> dict:
        """异步预测接口：发送 obs，接收 response。"""
        if self.websocket is None:
            raise RuntimeError("WebSocket is not connected.")

        await self.websocket.send(msgpack.packb(obs))
        response = msgpack.unpackb(await self.websocket.recv())
        return response

    async def _close(self):
        if self.websocket is not None:
            await self.websocket.close()
            logging.info("Wall-X websocket closed.")

    # ---------- 背景 event loop 封装成同步接口 ----------

    def _start_background_loop(self):
        """在独立线程中启动 event loop。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        """确保后台 event loop 在运行。"""
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(
                target=self._start_background_loop, daemon=True
            )
            self._thread.start()
            import time
            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        """在后台 event loop 中同步执行协程。"""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ---------- 对外同步接口 ----------

    def connect_sync(self):
        return self._run_async(self._connect())

    def predict_sync(self, obs: dict) -> dict:
        return self._run_async(self._predict(obs))

    def close_sync(self):
        result = self._run_async(self._close())
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return result


# ============= 原 ROS UAV 节点（改成用 Wall-X client） =============

class UAVPolicyNode:
    def __init__(self):
        # ROS节点初始化
        rospy.init_node('uav_policy_node', anonymous=True)

        # 创建CV桥接器
        self.bridge = CvBridge()

        # 存储最新的状态和图像
        self.current_state = None
        self.current_image = None
        self.current_wrist_image = None
        self.state_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.first_image = None
        self.first_image_received = False
        self.count = 0
        self.save_count = 0
        self.image_save_count = 0
        self.last_state = np.array([0, 0, 1, 0, 0, 0], dtype=float)
        self.last_inference_time = None
        self.inference_timeout = 10.0  # 推理超时时间（秒）

        # 创建图像保存目录
        self.image_save_dir = rospy.get_param(
            "~raw_image_save_dir",
            "/home/yang/VLA/Openpi/test/infer/trail/saved_images",
        )
        os.makedirs(self.image_save_dir, exist_ok=True)

        # 模型输入图像保存目录
        self.model_input_dir = rospy.get_param(
            "~model_input_dir",
            "/home/yang/VLA/Openpi/test/infer/trail/model_input_images",
        )
        os.makedirs(self.model_input_dir, exist_ok=True)

        # 订阅无人机状态和图像话题
        self.odom_sub = rospy.Subscriber(
            '/drone_0_visual_slam/odom',
            Odometry,
            self.odom_callback,
            queue_size=1,
        )

        self.image_sub = rospy.Subscriber(
            '/camera/color/image/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
        )

        # 发布无人机动作
        self.action_pub = rospy.Publisher(
            '/uav_actions',
            PoseStamped,
            queue_size=1,
        )

        # 初始化WebSocket客户端（Wall-X）
        self.client = None
        self.prompt = None

        # 轨迹存储
        self.original_trajectory = []
        self.inferred_trajectory = []

        self.tasks_jsonl_path = rospy.get_param(
            "~tasks_jsonl_path",
            "/home/yang/VLA/Openpi/test/infer/test_data/meta/tasks.jsonl",
        )
        self.task_index = rospy.get_param("~task_index", 2)

        # 从参数服务器获取配置
        self.host = rospy.get_param('~host', '127.0.0.1')
        self.port = rospy.get_param('~port', 8000)
        self.replan_steps = rospy.get_param('~replan_steps', 10)

        # Wall-X 的 prompt（可以直接沿用原来的逻辑）
        self.prompt = self.get_prompt_from_task_index()
        if self.prompt is None:
            self.prompt = "无人机实时轨迹控制"
            logging.warning("使用默认提示")
        print(f"使用提示: {self.prompt}")

        # 连接 Wall-X WebSocket 服务器
        self.connect_websocket()

    # ---------- 与任务文件相关 ----------

    def get_prompt_from_task_index(self):
        """
        从 tasks.jsonl 文件中根据 task_index 查找并返回对应的 prompt。
        """
        try:
            if not os.path.exists(self.tasks_jsonl_path):
                logging.error(f"tasks.jsonl 文件不存在: {self.tasks_jsonl_path}")
                return None

            with open(self.tasks_jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('task_index') == self.task_index:
                        return data.get('task')
        except FileNotFoundError:
            logging.error(f"tasks.jsonl 文件未找到: {self.tasks_jsonl_path}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"解析 tasks.jsonl 文件时出错: {self.tasks_jsonl_path}, 错误: {e}")
            return None
        except Exception as e:
            logging.error(f"读取 tasks.jsonl 文件时出错: {e}")
            return None

        logging.warning(
            f"在 {self.tasks_jsonl_path} 中未找到 task_index {self.task_index} 对应的任务。"
        )
        return None

    # ---------- WebSocket 连接 ----------

    def connect_websocket(self):
        """连接 Wall-X WebSocket 服务器"""
        try:
            logging.info(f"正在连接到服务器 ws://{self.host}:{self.port}")
            self.client = WallXWebsocketClient(self.host, self.port)
        except Exception as e:
            logging.error(f"连接WebSocket服务器失败: {e}")
            rospy.signal_shutdown("WebSocket连接失败")

    # ---------- ROS 回调 ----------

    def odom_callback(self, msg):
        """处理无人机状态回调"""
        with self.state_lock:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation

            roll, pitch, yaw = self.quaternion_to_euler(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )

            self.current_state = np.array(
                [position.x, position.y, position.z, roll, pitch, yaw],
                dtype=float,
            )

    def image_callback(self, msg):
        """处理图像回调"""
        try:
            with self.image_lock:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # 可选：保存原始图像
                # timestamp = rospy.Time.now()
                # image_filename = f"camera_image_{self.image_save_count:06d}_{timestamp.secs}_{timestamp.nsecs}.jpg"
                # image_path = os.path.join(self.image_save_dir, image_filename)
                # cv2.imwrite(image_path, cv_image)

                # 首帧作为主视角图像（256x256）
                if not self.first_image_received:
                    resized_image = cv2.resize(cv_image, (256, 256))
                    self.first_image = resized_image.copy()
                    self.count += 1
                    if self.count == 3:
                        self.first_image_received = True
                        first_image_path = os.path.join(
                            self.image_save_dir, "first_main_image.jpg"
                        )
                        cv2.imwrite(first_image_path, self.first_image)
                        logging.info(
                            f"固定主图像已保存: {first_image_path} (256x256)"
                        )

                # 实时图像作为第二视角（wrist）
                resized_wrist_image = cv2.resize(cv_image, (256, 256))
                self.current_wrist_image = resized_wrist_image.copy()

                self.image_save_count += 1
                if self.image_save_count % 100 == 0:
                    logging.info(f"已接收 {self.image_save_count} 张图像")

        except Exception as e:
            logging.error(f"图像处理错误: {e}")

    # ---------- 坐标转换 ----------

    def quaternion_to_euler(self, x, y, z, w):
        """四元数 -> 欧拉角 (roll, pitch, yaw)"""
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

    def euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角 -> 四元数"""
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

    # ---------- 推理相关 ----------

    def prepare_image_for_inference(self, image):
        """确保图像为 256x256"""
        if image is None:
            return None
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256))
        return image

    def publish_action(self, action):
        """发布动作到 ROS 话题"""
        if len(action) < 6:
            logging.error("动作数据不足6个元素")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(action[0])
        pose_msg.pose.position.y = float(action[1])
        pose_msg.pose.position.z = float(action[2])

        qx, qy, qz, qw = self.euler_to_quaternion(
            float(action[3]), float(action[4]), float(action[5])
        )
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.action_pub.publish(pose_msg)

    def run_inference(self):
        """主推理循环"""
        rate = rospy.Rate(20)  # 20Hz

        while not rospy.is_shutdown():
            # 1. 先检查状态与图像是否就绪
            with self.state_lock, self.image_lock:
                if (
                    self.current_state is None
                    or self.current_wrist_image is None
                    or not self.first_image_received
                ):
                    rate.sleep()
                    continue

                state = self.current_state.copy()
                image = self.first_image.copy()
                wrist_image = self.current_wrist_image.copy()

                image = self.prepare_image_for_inference(image)
                wrist_image = self.prepare_image_for_inference(wrist_image)

                if image is None or wrist_image is None:
                    rate.sleep()
                    continue

            # 2. 判断是否需要重新规划 / 推理
            distance = np.linalg.norm(state[0:2] - self.last_state[0:2], axis=0)
            current_time = rospy.Time.now()
            should_infer = False

            if distance < 0.01:
                should_infer = True
                inference_reason = f"位置距离满足条件: {distance:.4f} < 0.01"
            elif self.last_inference_time is not None:
                time_since_last = (current_time - self.last_inference_time).to_sec()
                if time_since_last > self.inference_timeout:
                    should_infer = True
                    inference_reason = (
                        f"时间超时: {time_since_last:.2f}s > {self.inference_timeout}s"
                    )
                else:
                    print(
                        f"等待中... 距离: {distance:.4f}, "
                        f"时间间隔: {time_since_last:.2f}s"
                    )
            else:
                should_infer = True
                inference_reason = "首次推理"

            if should_infer:
                try:
                    print(f"开始推理 - {inference_reason}")

                    timestamp = rospy.Time.now()

                    # 保存输入图像
                    main_image_filename = (
                        f"inference_main_{self.save_count:04d}_{timestamp.secs}.jpg"
                    )
                    main_image_path = os.path.join(
                        self.model_input_dir, main_image_filename
                    )
                    cv2.imwrite(main_image_path, image)

                    wrist_image_filename = (
                        f"inference_wrist_{self.save_count:04d}_{timestamp.secs}.jpg"
                    )
                    wrist_image_path = os.path.join(
                        self.model_input_dir, wrist_image_filename
                    )
                    cv2.imwrite(wrist_image_path, wrist_image)

                    # 保存状态信息
                    state_filename = (
                        f"inference_state_{self.save_count:04d}_{timestamp.secs}.txt"
                    )
                    state_path = os.path.join(self.model_input_dir, state_filename)
                    with open(state_path, 'w') as f:
                        f.write(f"推理步骤: {self.save_count}\n")
                        f.write(
                            f"时间戳: {timestamp.secs}.{timestamp.nsecs}\n"
                        )
                        f.write(f"推理原因: {inference_reason}\n")
                        f.write(f"位置距离: {distance:.6f}\n")
                        f.write(f"状态向量: {state.tolist()}\n")
                        f.write(f"提示: {self.prompt}\n")
                        f.write(f"主图像文件: {main_image_filename}\n")
                        f.write(f"手腕图像文件: {wrist_image_filename}\n")
                        f.write(f"图像尺寸: {image.shape}\n")

                    logging.info(
                        f"模型输入图像已保存: {main_image_filename}, "
                        f"{wrist_image_filename}"
                    )

                    # ===== 关键：组装 Wall-X server 期望的 obs =====
                    # 注意：模型训练一般用 RGB，这里从 BGR -> RGB
                    rgb_main = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb_wrist = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)

                    obs = {
                        # 与你 client.py 的 prepare_batch_sync 保持风格一致
                        "face_view": rgb_main.astype(np.uint8),
                        "first_face_view": rgb_wrist.astype(np.uint8),
                        "prompt": self.prompt,
                        # 状态简单打包成 (1, 6) float32；如需归一化可在此补充
                        "state": state.astype(np.float32)[None, :],
                        # dataset_names 随便给个名字，和 Normalizer 强绑定时可改成 repo_id
                        "dataset_names": ["uav_ros"],
                    }

                    # 调用 Wall-X server 进行推理
                    result = self.client.predict_sync(obs)
                    # server.py + client.py 约定的字段名是 "action"
                    action_chunk = result["action"]  # 形状一般是 [pred_horizon, action_dim]

                    self.last_inference_time = current_time

                    # 发布前若干步动作
                    action_chunk = np.asarray(action_chunk)
                    num_steps = min(self.replan_steps, action_chunk.shape[0])

                    for i in range(num_steps):
                        action = action_chunk[i]
                        self.publish_action(action)

                        self.inferred_trajectory.append(action)
                        if len(self.original_trajectory) == 0:
                            self.original_trajectory.append(state)
                        else:
                            self.original_trajectory.append(action)

                        rate.sleep()

                    self.last_state = action_chunk[num_steps - 1]
                    self.save_count += 1

                except Exception as e:
                    logging.error(f"推理错误: {e}")

            rate.sleep()

    # ---------- 轨迹分析与可视化（保持不变） ----------

    def visualize_trajectories(self):
        """可视化轨迹"""
        if len(self.original_trajectory) == 0 or len(self.inferred_trajectory) == 0:
            logging.warning("没有足够的轨迹数据来可视化")
            return

        original_traj = np.array(self.original_trajectory)
        inferred_traj = np.array(self.inferred_trajectory)

        min_len = min(len(original_traj), len(inferred_traj))
        original_traj = original_traj[:min_len]
        inferred_traj = inferred_traj[:min_len]

        pos_mae, rot_mae, pos_errors = self.calculate_trajectory_deviation(
            original_traj, inferred_traj
        )

        print("\n--- 轨迹偏差分析 ---")
        print(f"平均绝对位置误差 (Positional MAE): {pos_mae:.4f}")
        print(f"平均绝对姿态误差 (Rotational MAE): {rot_mae:.4f}")
        print("\n每个点的具体位置误差:")
        for i, err in enumerate(pos_errors):
            print(f"  点 {i}: {err:.4f}")
        print("------------------------\n")

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

        ax.plot(
            pos1[:, 1], pos1[:, 0], pos1[:, 2],
            'o-', label='Original Trajectory', color='blue',
        )
        ax.plot(
            pos2[:, 1], pos2[:, 0], pos2[:, 2],
            'o-', label='Inferred Trajectory', color='red',
        )

        for i in range(len(pos1)):
            ax.plot(
                [pos1[i, 1], pos2[i, 1]],
                [pos1[i, 0], pos2[i, 0]],
                [pos1[i, 2], pos2[i, 2]],
                '--', color='gray', linewidth=0.8,
            )

        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        plt.savefig('/home/yang/VLA/Openpi/test/infer/trail/traj3D_output.png')
        plt.close()

    def visualize_trajectories_2d_topdown(self, traj1, traj2):
        """2D俯视图轨迹可视化"""
        pos1 = traj1[:, :2]
        pos2 = traj2[:, :2]

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.plot(
            pos1[:, 1], pos1[:, 0],
            'o-', label='Original Trajectory', color='blue',
        )
        ax.plot(
            pos2[:, 1], pos2[:, 0],
            'o-', label='Inferred Trajectory', color='red',
        )

        for i in range(len(pos1)):
            ax.plot(
                [pos1[i, 1], pos2[i, 1]],
                [pos1[i, 0], pos2[i, 0]],
                '--', color='gray', linewidth=0.8,
            )

        ax.set_xlabel('Y (m, Left/Right)')
        ax.set_ylabel('X (m, Forward/Backward)')
        ax.set_title('Trajectory Comparison (Top-down View)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.savefig('/home/yang/VLA/Openpi/test/infer/trail/traj2D_output.png')
        plt.close()


def main():
    logging.basicConfig(level=logging.INFO)

    try:
        print("UAV Wall-X policy node starting...")
        node = UAVPolicyNode()
        print("UAV Wall-X policy node started.")
        # 推理线程
        inference_thread = threading.Thread(target=node.run_inference)
        inference_thread.start()

        rospy.spin()

        inference_thread.join()

        node.visualize_trajectories()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logging.error(f"主程序错误: {e}")


if __name__ == "__main__":
    main()
