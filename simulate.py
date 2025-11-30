#!/usr/bin/env python3

import json
import logging
import sys
import os
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import threading

from openpi_client import websocket_client_policy as _websocket_client_policy

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
        self.save_count = 0  # 添加保存计数器
        self.image_save_count = 0  # 添加图像保存计数器
        self.last_state = [0, 0, 1, 0, 0, 0]
        self.last_inference_time = None  # 添加上次推理时间记录
        self.inference_timeout = 10.0  # 设置推理超时时间（秒），可以根据需要调整
        
        # 创建图像保存目录
        self.image_save_dir = '/home/yang/VLA/Openpi/test/infer/trail/saved_images'
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        # 创建模型输入图像保存目录
        self.model_input_dir = '/home/yang/VLA/Openpi/test/infer/trail/model_input_images'
        os.makedirs(self.model_input_dir, exist_ok=True)
        
        # 订阅无人机状态和图像话题
        self.odom_sub = rospy.Subscriber('/drone_0_visual_slam/odom',    #/drone_0_odom_visualization/pose是无人机在unity里的世界坐标。需要交换YZ，四元数转欧拉角
                                        Odometry, 
                                        self.odom_callback,
                                        queue_size=1)
        
        self.image_sub = rospy.Subscriber('/camera/color/image/compressed', #/camera0/color/image/compressed是前置摄像头
                                         CompressedImage, 
                                         self.image_callback,
                                         queue_size=1)
        
        # 发布无人机动作
        self.action_pub = rospy.Publisher('/uav_actions', 
                                         PoseStamped, 
                                         queue_size=1)

        # 初始化WebSocket客户端
        self.client = None
        self.prompt = None
        
        # 轨迹存储
        self.original_trajectory = []
        self.inferred_trajectory = []

        self.tasks_jsonl_path = "/home/yang/VLA/Openpi/test/infer/test_data/meta/tasks.jsonl"
        self.task_index = 2
        
        # 从参数服务器获取配置
        self.host = rospy.get_param('~host', '127.0.0.1')  # 修改默认host
        self.port = rospy.get_param('~port', 8000)
        self.replan_steps = rospy.get_param('~replan_steps', 10)
        # self.prompt = rospy.get_param('~prompt', 'Navigate to the target location')
        self.prompt = self.get_prompt_from_task_index()
        if self.prompt is None:
            self.prompt = "无人机实时轨迹控制"  # 默认提示
            logging.warning("使用默认提示")
        print(f"使用提示: {self.prompt}")
        
        # 连接WebSocket服务器
        self.connect_websocket()

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
        
        logging.warning(f"在 {self.tasks_jsonl_path} 中未找到 task_index {self.task_index} 对应的任务。")
        return None
        
    def connect_websocket(self):
        """连接WebSocket服务器"""
        try:
            logging.info(f"正在连接到服务器 ws://{self.host}:{self.port}")
            self.client = _websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
        except Exception as e:
            logging.error(f"连接WebSocket服务器失败: {e}")
            rospy.signal_shutdown("WebSocket连接失败")
    
    def odom_callback(self, msg):
        """处理无人机状态回调"""
        with self.state_lock:
            # 从Odometry消息中提取位置和姿态
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            
            # 将四元数转换为欧拉角
            roll, pitch, yaw = self.quaternion_to_euler(orientation.x, 
                                                      orientation.y, 
                                                      orientation.z, 
                                                      orientation.w)
            
            # 更新当前状态: [x, y, z, roll, pitch, yaw]
            self.current_state = np.array([
                position.x, position.y, position.z,
                roll, pitch, yaw
            ])
    
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
                    resized_image = cv2.resize(cv_image, (256, 256))
                    self.first_image = resized_image.copy()
                    self.count = self.count + 1
                    if self.count == 3:
                        self.first_image_received = True
                        # 保存第一帧图像作为固定主图像
                        first_image_path = os.path.join(self.image_save_dir, "first_main_image.jpg")
                        cv2.imwrite(first_image_path, self.first_image)
                        logging.info(f"固定主图像已保存: {first_image_path} (256x256)")
                
                # 将实时图像resize到256x256作为wrist图像
                resized_wrist_image = cv2.resize(cv_image, (256, 256))
                self.current_wrist_image = resized_wrist_image.copy()
                
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
        if len(action) < 6:
            logging.error("动作数据不足6个元素")
            return
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        # 设置位置
        pose_msg.pose.position.x = action[0]
        pose_msg.pose.position.y = action[1]
        pose_msg.pose.position.z = action[2]
        
        # 将欧拉角转换为四元数
        qx, qy, qz, qw = self.euler_to_quaternion(action[3], action[4], action[5])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self.action_pub.publish(pose_msg)
    
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
        
        while not rospy.is_shutdown():
            # 检查是否有有效的状态和图像
            with self.state_lock, self.image_lock:
                if (self.current_state is None or 
                    self.current_wrist_image is None or 
                    not self.first_image_received):
                    rate.sleep()
                    continue
                
                state = self.current_state.copy()
                image = self.first_image.copy()  # 使用第一帧作为固定图像 (已经是256x256)
                wrist_image = self.current_wrist_image.copy()  # 使用实时图像 (已经是256x256)
                
                # 确保图像尺寸为256x256
                image = self.prepare_image_for_inference(image)
                wrist_image = self.prepare_image_for_inference(wrist_image)
                
                if image is None or wrist_image is None:
                    rate.sleep()
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
            
            if should_infer:
                try:
                    print(f"开始推理 - {inference_reason}")
                    
                    # 保存输入到模型的图像
                    timestamp = rospy.Time.now()
                    
                    # 保存固定主图像 (256x256)
                    main_image_filename = f"inference_main_{self.save_count:04d}_{timestamp.secs}.jpg"
                    main_image_path = os.path.join(self.model_input_dir, main_image_filename)
                    cv2.imwrite(main_image_path, image)
                    
                    # 保存wrist图像 (256x256)
                    wrist_image_filename = f"inference_wrist_{self.save_count:04d}_{timestamp.secs}.jpg"
                    wrist_image_path = os.path.join(self.model_input_dir, wrist_image_filename)
                    cv2.imwrite(wrist_image_path, wrist_image)
                    
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
                        f.write(f"手腕图像文件: {wrist_image_filename}\n")
                        f.write(f"图像尺寸: {image.shape}\n")
                    
                    logging.info(f"模型输入图像已保存: {main_image_filename}, {wrist_image_filename}")
                    
                    # 准备输入数据
                    element = {
                        "observation/image": image,          # 固定使用第一帧图像
                        "observation/wrist_image": wrist_image,  # 使用实时图像
                        "observation/state": state,
                        "prompt": self.prompt
                    }
                    print(self.prompt)
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
                    result = self.client.infer(element)
                    action_chunk = result["actions"]
                    
                    # 更新推理时间记录
                    self.last_inference_time = current_time

                    # state_filename = f"action_{self.save_count:04d}_{timestamp.secs}.txt"
                    # state_path = os.path.join(self.model_input_dir, state_filename)
                    # with open(state_path, 'w') as f:
                    #     f.write(f"action: {action_chunk}\n")
                    
                    # 发布第一个动作
                    for i in range(10):
                        self.publish_action(action_chunk[i])
                        
                        self.inferred_trajectory.append(action_chunk[i])
                        if len(self.original_trajectory) == 0:
                            self.original_trajectory.append(state)
                        else:
                            self.original_trajectory.append(action_chunk[i])

                        rate.sleep()
                    self.last_state = action_chunk[9]
                        
                    # if len(action_chunk) > 0:
                    #     self.publish_action(action_chunk[0])
                        
                    #     # 存储轨迹用于可视化
                    #     self.inferred_trajectory.append(action_chunk[0])
                    #     if len(self.original_trajectory) == 0:
                    #         self.original_trajectory.append(state)
                    #     else:
                    #         # 在实际应用中，这里应该获取真实的下一状态
                    #         # 由于我们是在线运行，无法获取真实下一状态，所以使用推理结果
                    #         self.original_trajectory.append(action_chunk[0])
                    
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

def main():
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("program starting...")
        node = UAVPolicyNode()
        
        # 在一个单独的线程中运行推理循环
        inference_thread = threading.Thread(target=node.run_inference)
        inference_thread.start()
        
        # 主线程运行ROS spin
        rospy.spin()
        
        # 等待推理线程结束
        inference_thread.join()
        
        # 可视化轨迹
        node.visualize_trajectories()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logging.error(f"主程序错误: {e}")

if __name__ == "__main__":
    main()