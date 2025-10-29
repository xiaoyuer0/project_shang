# python_scripts/PPO/RobotRun1.py

import numpy as np
import math
import torch
from python_scripts.Webots_interfaces import Darwin # 我们需要达尔文控制器
from python_scripts.Project_config import Darwin_config, gps_goal, path_list

class RobotRun:
    """
    机器人执行一步逻辑的控制器。
    它接收一个 Darwin 控制器实例，并利用它来执行动作、观测状态和计算奖励。
    """
    def __init__(self, darwin_instance, state, action_shouder, action_arm, step, catch_flag, gps1, gps2, gps3, gps4, img_name):
        """
        初始化 RobotRun 控制器。
        
        注意：第一个参数 darwin_instance 是一个 Darwin 控制器实例，
        不是一个 Webots Robot 对象。
        """
        print("--- RobotRun 初始化开始 ---")
        
        # 【核心】darwin_instance 就是我们封装好的硬件控制器
        self.darwin = darwin_instance
        self.robot_state = state
        self.step_num = step
        self.catch_flag = catch_flag
        self.gps = [gps1, gps2, gps3, gps4]
        self.img_name = img_name
        
        # 从 Darwin 配置中读取常量
        self.gps_x_goal, self.gps_y_goal = gps_goal[0], gps_goal[1]
        self.standard_angle = Darwin_config.standard_angle
        self.touch_T = np.array(Darwin_config.touch_T)
        self.touch_F = np.array(Darwin_config.touch_F)
        self.acc_low = np.array(Darwin_config.acc_low)
        self.acc_high = np.array(Darwin_config.acc_high)
        self.gyro_low = np.array(Darwin_config.gyro_low)
        self.gyro_high = np.array(Darwin_config.gyro_high)
        self.joint_limits = Darwin_config.limit

        # --- 动作计算和目标状态 ---
        current_left_arm = self.robot_state[5]
        current_left_shoulder = self.robot_state[1]
        
        # 将 torch 张量/标量转换为 Python float
        left_arm_target = 1.25 * float(action_arm) + 0.25
        left_shoulder_target = 0.2995 * float(action_shouder) - 0.145
        
        self.ArmLower = left_arm_target - current_left_arm
        self.ArmLower = max(-0.3, min(0.3, self.ArmLower))
        self.Shoulder = left_shoulder_target - current_left_shoulder
        self.Shoulder = max(-0.3, min(0.3, self.Shoulder))

        # 定义需要操作的电机索引 (根据 Darwin.motorName 元组)
        # ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL', ...)
        self.shoulder_l_idx = 1
        self.arm_l_idx = 3
        self.shoulder_r_idx = 0
        self.arm_r_idx = 2
        self.grasp_l_idx = 20
        self.grasp_r_idx = 21

        # 初始化快捷传感器列表
        self.touch_sensors = self.darwin.touch_sensors
        self.touch_grab_sensors = [self.touch_sensors['grasp_L1_1'], self.touch_sensors['grasp_R1_2']]
        self.touch_body_sensors = [self.touch_sensors['arm_L1'], self.touch_sensors['arm_R1'], 
                                   self.touch_sensors['leg_L1'], self.touch_sensors['leg_L2'], 
                                   self.touch_sensors['leg_R1'], self.touch_sensors['leg_R2']]
        
        # 用于返回的中间状态
        self.next_state_buffer = [0] * 20
        self.return_flags = {'reward': 0, 'done': 0, 'good': 0, 'goal': 0, 'count': 1}

        print("--- RobotRun 初始化完成 ---")

    def run(self):
        """
        执行预定义的动作序列，并返回结果。
        """
        print(f"--- RobotRun: 执行第 {self.step_num} 步动作 ---")
        
        # 1. 执行动作
        if self.catch_flag == 0.0: # 如果还没有抓取
            # 设置目标关节位置
            self.darwin.motors[self.shoulder_l_idx].setPosition(self.robot_state[1] + self.Shoulder)
            self.darwin.motors[self.shoulder_r_idx].setPosition(self.robot_state[0] - self.Shoulder)
            self.darwin.motors[self.arm_l_idx].setPosition(self.robot_state[5] + self.ArmLower)
            self.darwin.motors[self.arm_r_idx].setPosition(self.robot_state[4] - self.ArmLower)

            # 推进仿真以使动作生效
            for _ in range(10):
                self.darwin.robot.step(self.darwin.get_timestep())

            # 2. 观测新状态和计算奖励
            next_robot_state = self.darwin.get_robot_state() # 关节角度
            acc = self.darwin.accelerometer.getValues()
            gyro = self.darwin.gyro.getValues()
            
            # 高级GPS奖励
            gps_val, _, _, _, _ = self.darwin.get_gps_values()
            dx = self.gps_x_goal - gps_val[1]
            dy = self.gps_y_goal - gps_val[2]
            distance_to_goal = math.sqrt(dx**2 + dy**2)
            self.return_flags['reward'] = 20 - distance_to_goal * 200 # 基础距离奖励
            if distance_to_goal < 0.03:
                self.return_flags['reward'] += 100 # 到达目标奖励
                self.return_flags['goal'] = 1 # 设置成功标志
                print("距离目标很近，给予额外奖励!")
            
            # 3. 多次结束判断
            # 3a: 检查关节限制
            if not self.darwin.check_joint_limits(next_robot_state):
                self.return_flags.update({'done': 1, 'good': 1, 'count': 0, 'reward': 0})
                print("关节超出限制，回合结束!")
                return next_robot_state, self.return_flags['reward'], self.return_flags['done'], self.return_flags['good'], self.return_flags['goal'], self.return_flags['count']

            # 3b: 检查加速度/陀螺仪
            if not (self.acc_low < acc < self.acc_high).all() or not (self.gyro_low < gyro < self.gyro_high).all():
                self.return_flags.update({'done': 1, 'good': 0, 'count': 0, 'reward': 0})
                print("身体传感器超出限制，回合结束!")
                return next_robot_state, self.return_flags['reward'], self.return_flags['done'], self.return_flags['good'], self.return_flags['goal'], self.return_flags['count']

            # 3c: 检查碰撞
            if any(sensor.getValue() == 1.0 for sensor in self.touch_body_sensors):
                self.return_flags.update({'done': 1, 'good': 1, 'count': 0})
                print("身体发生碰撞，回合结束!")
                return next_robot_state, self.return_flags['reward'], self.return_flags['done'], self.return_flags['good'], self.return_flags['goal'], self.return_flags['count']

            # 3d: 检查是否接触物体 (启动抓取序列)
            if any(sensor.getValue() == 1.0 for sensor in self.darwin.touch_sensors.values() if 'grasp' in sensor.name): # 对所有抓取传感器
                print("检测到与物体接触，尝试抓取!")
                return self._process_grasp_attempt(next_robot_state)
            
            # 如果一切正常，返回中间状态
            return next_robot_state, self.return_flags['reward'], self.return_flags['done'], self.return_flags['good'], self.return_flags['goal'], self.return_flags['count']

        else: # 如果已经抓取了 (catch_flag != 0.0)
            return self._process_grasp_attempt(self.robot_state) # 用当前状态继续处理

    def _process_grasp_attempt(self, current_state):
        """
        辅助方法：处理抓取逻辑。
        """
        print("--- 抓取处理中 ---")
        self.darwin.motors[self.grasp_l_idx].setPosition(-0.5)
        self.darwin.motors[self.grasp_r_idx].setPosition(-0.5)
        
        # 等待抓取动作完成
        for _ in range(20):
            self.darwin.robot.step(self.darwin.get_timestep())

        # 读取抓取传感器
        touch_values = [s.getValue() for s in self.touch_grab_sensors]
        if touch_values == [0.0, 1.0]: touch_values = [1.0, 1.0] # 修复一个已知的传感器组合问题

        # 判断抓取结果
        success = np.array_equal(touch_values, self.touch_T)
        failure = np.array_equal(touch_values, self.touch_F)
        
        gps_val, _, _, _, _ = self.darwin.get_gps_values()

        if success:
            print("抓取成功!")
            self.return_flags.update({'done': 1, 'good': 1, 'goal': 1, 'count': 0})
            # 写入GPS数据
            with open(path_list['gps_path_DQN'], 'a') as f:
                f.write(f"{[gps_val[1], gps_val[2]]},") # 只写入关键x,y坐标
        elif failure and self.step_num <= 5:
            print("抓取失败（过早）!")
            self.return_flags.update({'done': 1, 'good': 1, 'count': 1})
            with open(path_list['shu_ju_path_DQN'], 'a') as f:
                f.write('0,')
        elif failure:
            print("抓取失败（太晚）!")
            self.return_flags.update({'done': 1, 'good': 1, 'count': 1})
        
        # 即使失败，也返回当前状态和累积的奖励
        return current_state, self.return_flags['reward'], self.return_flags['done'], self.return_flags['good'], self.return_flags['goal'], self.return_flags['count']

