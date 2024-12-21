# requirements
# - Python >= 3.7
# - torch >= 1.7
# - gym == 0.23
# - (Optional) tensorboard, wandb

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import sys


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        # inputsize "hidden_size" and outputsize "output_size"
        self.fc2 = nn.Linear(128, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        self.buffer.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        # network for evaluate
        self.eval_net = QNet(input_size, hidden_size, output_size)
        # target network
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        #self.scheduler = lr_scheduler.StepLR(self.optim, step_size=50, gamma=0.99)  # 每10个epoch学习率乘以0.1
        self.eps = args.eps
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0

    def choose_action(self, obs):
        # print(self.eps)
        if np.random.uniform() < self.eps:  # 以一定的概率随机选择动作
            action = np.random.randint(0, self.env.action_space.n)
        else:  # 以1-eps的概率选择最优动作
            obs_tmp = torch.FloatTensor(obs)
            with torch.no_grad():
                vals = self.eval_net(obs_tmp)  # 输出值
            action = vals.argmax().item()  # 选取神经网络输出值大的动作
        return action

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def learn(self):
        # [Epsilon Decay]
        if self.eps > args.eps_min:
            self.eps *= args.eps_decay
        #self.scheduler.step()
        # [Update Target Network Periodically]
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        # [Sample Data From Experience Replay Buffer]
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)  # to use 'gather' latter
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)

        # For example:
        # 1. calculate q_eval with eval_net and q_target with target_net
        # 2. td_target = r + gamma * (1-dones) * q_target
        # 3. calculate loss between "q_eval" and "td_target" with loss_fn
        # 4. optimize the network with self.optim

        q_eval = self.eval_net(np.array(obs)).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = self.eval_net(np.array(next_obs)).argmax(dim=1)
        q_next = self.target_net(np.array(next_obs)).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        q_target = rewards + args.gamma * (1 - dones) * q_next  # 目标值
        loss = self.loss_fn(q_eval, q_target)  # 计算损失
        self.optim.zero_grad()  # 清空上一轮梯度
        loss.backward()  # 反向传播
        self.optim.step()  # 优化反向传播


def plot_rewards(rewards, avg_rewards, i):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per episode', color='blue')
    plt.plot(range(100, len(rewards)), avg_rewards, label='Average reward per 100 episodes', color='orange')

    # Find the maximum average reward and its index
    max_avg_reward = max(avg_rewards)
    max_index = avg_rewards.index(max_avg_reward) + 100  # +100 because avg_rewards starts from 100th episode

    # Plot a red dot at the maximum average reward
    plt.plot(max_index, max_avg_reward, 'ro')
    # Annotate the maximum average reward on the plot
    plt.annotate(f'Max Avg Reward = {max_avg_reward}', (max_index, max_avg_reward), textcoords="offset points",
                 xytext=(-10, -10), ha='center')

    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'../DQNpics_epsdecay9999/rewards_{args.capacity}_{args.lr}_{args.update_target}_{args.eps_min}_{i}.png')
    # plt.show()


def main(i):
    avg_reward = 0.0
    rewards = []
    avg_rewards = []
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)  # 初始化DQN智能体
    stop_learning = False
    for i_episode in range(args.n_episodes):  # 开始玩游戏
        obs = env.reset()  # 重置环境
        episode_reward = 0  # 用于记录整局游戏能获得的reward总和
        done = False
        step_cnt = 0
        while not done and step_cnt < 500:
            step_cnt += 1
            action = agent.choose_action(obs)  # 根据当前观测选择动作
            next_obs, reward, done, info = env.step(action)  # 与环境交互
            agent.store_transition(obs, action, reward, next_obs, done)  # 存储转移
            # reward -= abs(next_obs[0])/5  # 为了让小车尽快到达目标，设置reward为负的x坐标
            episode_reward += reward  # 记录当前动作获得的reward
            obs = next_obs
            if agent.buffer.len() >= 256 and not stop_learning:
                agent.learn()
        rewards.append(episode_reward)
        if episode_reward == 500 and not stop_learning:
            stop_learning = True
        if episode_reward < 470:
            stop_learning = False
        if i_episode >= 100:
            avg_reward = np.mean(rewards[-100:])
            avg_rewards.append(avg_reward)
        if i_episode >= 100:
            print(f"Episode: {i_episode}, Reward: {episode_reward}, Avg Reward per 100 Episodes: {avg_reward}")
        else:
            print(f"Episode: {i_episode}, Reward: {episode_reward}")
    plot_rewards(rewards, avg_rewards, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment name")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--hidden", default=64, type=int, help="dimension of hidden layer")
    parser.add_argument("--n_episodes", default=500, type=int, help="number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity", default=10000, type=int, help="capacity of replay buffer")
    parser.add_argument("--eps", default=0.1, type=float, help="epsilon of ε-greedy")
    parser.add_argument("--eps_min", default=0.05, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eps_decay", default=0.9999, type=float)
    parser.add_argument("--update_target", default=100, type=int, help="frequency to update target network")
    args = parser.parse_args()
    for i in range(10):
        main(i)
        # 清屏

