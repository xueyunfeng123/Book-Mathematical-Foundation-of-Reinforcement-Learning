#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : Dyna-Q.py
@Author  : Xue Yunfeng
@Date    : 2025/10/6
@Description : 简要描述文件功能
"""
import gymnasium as gym
import numpy as np
import random
import time


class DynaQ:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=5):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.n_planning = n_planning  # 规划次数

        # 初始化Q表
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

        # 初始化模型 (状态, 动作) -> (奖励, 下一个状态)
        self.model = {}

        # 记录访问过的状态-动作对
        self.visited_sa = set()

    def choose_action(self, state):
        # ε-贪婪策略选择动作
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # 随机探索
        else:
            return np.argmax(self.Q[state, :])  # 利用

    def update(self, state, action, reward, next_state):
        # 更新Q值
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        # 更新模型
        self.model[(state, action)] = (reward, next_state)
        self.visited_sa.add((state, action))

    def planning(self):
        # 规划步骤：从访问过的状态-动作对中随机选择并进行模拟更新
        if len(self.visited_sa) == 0:
            return

        for _ in range(self.n_planning):
            # 随机选择一个曾经访问过的状态-动作对
            state, action = random.choice(list(self.visited_sa))

            # 从模型中获取预测的奖励和下一个状态
            reward, next_state = self.model.get((state, action), (0, state))

            # 使用模拟经验更新Q值
            best_next_action = np.argmax(self.Q[next_state, :])
            td_target = reward + self.gamma * self.Q[next_state, best_next_action]
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

    def train(self, episodes=1000):
        rewards = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # 选择动作
                action = self.choose_action(state)

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 更新Q值和模型
                self.update(state, action, reward, next_state)

                # 规划步骤
                self.planning()

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

            # 每100轮打印一次进度
            if (episode + 1) % 100 == 0:
                print(
                    f"Episode: {episode + 1}, Total Reward: {total_reward}, Average Reward: {np.mean(rewards[-100:])}")

        return rewards


# 创建环境
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)

# 创建Dyna-Q代理
dyna_q = DynaQ(env, n_planning=10)  # 设置规划次数为10

# 训练代理
rewards = dyna_q.train(episodes=1000)


# 测试训练后的策略
def test_policy(agent, env, n_tests=5):
    for test in range(n_tests):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"Test {test + 1}")

        while not done:
            action = np.argmax(agent.Q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 可视化
            env.render()
            time.sleep(0.5)
            state = next_state

        print(f"Total reward: {total_reward}\n")


# 测试训练后的策略
test_policy(dyna_q, env)

# 关闭环境
env.close()