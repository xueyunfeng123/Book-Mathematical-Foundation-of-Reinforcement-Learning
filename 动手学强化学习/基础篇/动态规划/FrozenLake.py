import gymnasium as gym
import numpy as np


class PolicyIteration:
    def __init__(self, env, theta=1e-5, gamma=0.9):
        self.env = env
        self.theta = theta  # 收敛阈值
        self.gamma = gamma  # 折扣因子

        # 初始化值函数和策略
        self.V = np.zeros(env.observation_space.n)  # 值函数
        self.policy = np.zeros(env.observation_space.n, dtype=int)  # 策略

        # 动作意义映射
        self.action_meaning = ['<', 'v', '>', '^']  # 左, 下, 右, 上

    def policy_evaluation(self):
        """策略评估：计算当前策略下的值函数"""
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                # 跳过终止状态（目标或冰洞）
                if s in ends or s in holes:
                    continue

                v = self.V[s]
                # 获取当前状态下采取策略动作的转移信息
                action = self.policy[s]
                transitions = self.env.P[s][action]

                # 计算新的值函数
                new_v = 0
                for prob, next_state, reward, _ in transitions:
                    new_v += prob * (reward + self.gamma * self.V[next_state])

                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))

            # 检查是否收敛
            if delta < self.theta:
                break

    def policy_improvement(self):
        """策略改进：基于当前值函数改进策略"""
        policy_stable = True
        for s in range(self.env.observation_space.n):
            # 跳过终止状态（目标或冰洞）
            if s in ends or s in holes:
                continue

            old_action = self.policy[s]

            # 计算每个动作的期望价值
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                transitions = self.env.P[s][a]
                for prob, next_state, reward, _ in transitions:
                    action_values[a] += prob * (reward + self.gamma * self.V[next_state])

            # 选择价值最大的动作
            self.policy[s] = np.argmax(action_values)

            # 检查策略是否稳定
            if old_action != self.policy[s]:
                policy_stable = False

        return policy_stable

    def policy_iteration(self):
        """策略迭代主循环"""
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}:")

            # 策略评估
            self.policy_evaluation()

            # 策略改进
            policy_stable = self.policy_improvement()

            # 如果策略稳定，则停止迭代
            if policy_stable:
                print("Policy stable after", iteration, "iterations")
                break


def print_agent(agent, action_meaning, holes, ends):
    """打印最终策略和值函数"""
    print("\nValue function:")
    print(agent.V.reshape(4, 4))  # 假设是4x4网格

    print("\nPolicy:")
    policy = np.array([action_meaning[a] for a in agent.policy])
    print(policy.reshape(4, 4))  # 假设是4x4网格

    # 标记冰洞和目标
    grid = np.full((4, 4), ' ')
    for hole in holes:
        row, col = hole // 4, hole % 4
        grid[row, col] = 'H'
    for end in ends:
        row, col = end // 4, end % 4
        grid[row, col] = 'G'

    print("\nGrid (H: hole, G: goal):")
    print(grid)


# 主程序
if __name__ == "__main__":
    # 创建FrozenLake环境
    env = gym.make("FrozenLake-v1", render_mode='human')
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P

    # 分析环境中的冰洞和目标位置
    holes = set()
    ends = set()
    for s in env.P:  # 遍历所有状态
        for a in env.P[s]:  # 遍历每个状态下所有可能的动作
            for s_ in env.P[s][a]:  # 遍历每个动作可能的结果
                if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                    ends.add(s_[1])
                if s_[3] == True:  # 如果episode终止且不是目标，则是冰洞
                    holes.add(s_[1])

    # 确保冰洞集合中不包含目标位置
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)

    # 查看状态14（目标左边一格）的状态转移信息
    print("\n状态14的转移信息:")
    for a in env.P[14]:
        print(env.P[14][a])

    # 动作意义映射
    action_meaning = ['<', 'v', '>', '^']  # 左, 下, 右, 上

    # 算法参数
    theta = 1e-5
    gamma = 0.9

    # 创建并运行策略迭代算法
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()

    # 打印结果
    print_agent(agent, action_meaning, list(holes), list(ends))

    # 演示最优策略
    print("\n演示最优策略:")
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.policy[observation]  # 使用学习到的最优策略
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()  # 显示环境

    print(f"总奖励: {total_reward}")
    env.close()