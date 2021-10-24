# -*- coding: UTF-8 -*-
import numpy as np
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
from gym.envs.registration import EnvSpec
import logging
from matplotlib import pyplot as plt
from IPython import display

logging.basicConfig(level=logging.WARNING)


# plt.figure()
# plt.ion()


def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r ** 2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]


class MEC_MARL_ENV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world, alpha=0.5, beta=0.2, aggregate_reward=False, discrete=True,
                 reset_callback=None, info_callback=None, done_callback=None):
        # system initialize
        self.world = world
        self.obs_r = world.obs_r  # 观察半径
        self.move_r = world.move_r  # 移动半径
        self.collect_r = world.collect_r  # 收集半径
        self.max_buffer_size = self.world.max_buffer_size
        self.agents = self.world.agents
        self.agent_num = self.world.agent_count
        self.sensor_num = self.world.sensor_count
        self.sensors = self.world.sensors
        self.DS_map = self.world.DS_map
        self.map_size = self.world.map_size
        self.DS_state = self.world.DS_state
        self.alpha = alpha  # TODO
        self.beta = beta  # TODO
        # TODO
        self.reset_callback = reset_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # game mode
        self.aggregate_reward = aggregate_reward  # share same rewards
        self.discrete_flag = discrete
        self.state = None
        self.time = 0
        self.images = []

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if self.discrete_flag:
                act_space = spaces.Tuple((circle_space.Discrete_Circle(agent.move_r),
                                          # 移动范围      # gym.spaces.Discrete() 创建一个离散的n维空间，n为整数,从space进行抽样
                                          onehot_space.OneHot(self.max_buffer_size),  # 执行空间
                                          sum_space.SumOne(self.agent_num),  # 分配带宽
                                          onehot_space.OneHot(self.max_buffer_size)))  # 卸载空间
                # move, offloading(boolxn), bandwidth([0,1]), execution
                obs_space = spaces.Tuple((spaces.MultiDiscrete([self.map_size, self.map_size]),  # 环境的观察
                                          spaces.Box(0, np.inf, [agent.obs_r * 2, agent.obs_r * 2,
                                                                 2])))  # 边缘设备的状态   # Box空间表示一个n维框  #TODO 三个参数什么意思
                # pos, obs map
                self.action_space.append(act_space)
                self.observation_space.append(obs_space)
        self.render()  # 画出环境map

    """环境的step 函数返回需要的信息, step 函数返回四个值observation、reward、done、info"""
    """
    observation (object):一个与环境相关的对象描述你观察到的环境，如相机的像素信息，机器人的角速度和角加速度，棋盘游戏中的棋盘状态。
    reward (float):先前行为获得的所有回报之和，不同环境的计算方式不 一，但目标总是增加自己的总回报。
    done (boolean): 判断是否到了重新设定(reset)环境，大多数任务分为明确定义的episodes，并且完成为True表示episode已终止。
    info (dict):用于调试的诊断信息，有时也用于学习，但正式的评价不允许使用该信息进行学习。 这是一个典型的agent-environment loop 的实现。
    """

    def step(self, agent_action, center_action):
        obs = []
        reward = []
        reward_average = []
        reward_age = []
        done = []
        info = {'n': []}
        self.agents = self.world.agents

        # world step
        logging.info("set actions")
        for i, agent in enumerate(self.agents):  # 每个edge agent 执行动作
            self._set_action(agent_action[i], center_action, agent)

        # world update
        self.world.step()
        # new observation
        logging.info("agent observation")
        for agent in self.agents:
            obs.append(self.get_obs(agent))  # 观察范围
            done.append(self._get_done(agent))  # 完成反馈
            # TODO reward修改
            # # 年龄的reward
            # reward.append(self._get_age())  # 每个agent reward相同，都是平均年龄
            # # 平均任务的reward
            # reward.append(self._get_reward())
            # # 联合的reward
            reward_age.append(self._get_age())
            reward_average.append(self._get_reward())
            info['n'].append(self._get_info(agent))
        self.state = obs

        # # # 单个reward
        # # reward_sum = np.sum(reward)
        #
        # logging.info("get reward")
        # if self.aggregate_reward:  # 源代码这句话不执行，即每个agent不共用相同的sum_reward
        #     reward = [reward_sum] * self.agent_num
        #     # reward = [reward_sum / self.agent_num] * self.agent_num
        # return self.state, reward, done, info
        # 多个reward
        reward_age_sum = np.sum(reward_age)
        reward_average_sum = np.sum(reward_average)
        logging.info("get reward")
        if self.aggregate_reward:  # 源代码这句话不执行，即每个agent不共用相同的sum_reward
            reward_age = [reward_age_sum] * self.agent_num
            reward_average = [reward_average_sum] * self.agent_num
        return self.state, reward_age, reward_average, done, info



    def reset(self):
        # reset world
        self.world.finished_data = []
        # reset renderer
        # self._reset_render()
        # record observations for each agent
        for sensor in self.sensors:
            sensor.data_buffer = []
            sensor.collect_state = False
        for agent in self.agents:
            agent.idle = True
            agent.data_buffer = {}
            agent.total_data = {}
            agent.done_data = []
            agent.collecting_sensors = []  # 记录收集了数据的数据源列表清空

    """edge agent 执行动作"""

    def _set_action(self, act, center_action, agent):
        agent.action.move = np.zeros(2)
        agent.action.execution = act[1]  # 执行缓冲区 执行
        agent.action.bandwidth = center_action[agent.no]  # 获得带宽分配情况
        if agent.movable and agent.idle:
            # print([agent.no, act[0]])
            if np.linalg.norm(act[0]) > agent.move_r:  # 如果 移动距离超出范围
                act[0] = [int(act[0][0] * agent.move_r / np.linalg.norm(act[0])),
                          int(act[0][1] * agent.move_r / np.linalg.norm(act[0]))]
            if not np.count_nonzero(act[0]) and np.random.rand() > 0.5:  # 如果保持原位，有一半的概率执行以下语句，选一个新的移动位置
                mod_x = np.random.normal(loc=0, scale=1)
                mod_y = np.random.normal(loc=0, scale=1)
                mod_x = int(min(max(-1, mod_x), 1) * agent.move_r / 2)
                mod_y = int(min(max(-1, mod_y), 1) * agent.move_r / 2)
                act[0] = [mod_x, mod_y]
            agent.action.move = np.array(act[0])  # move 记录移动到的位置
            new_x = agent.position[0] + agent.action.move[0]
            new_y = agent.position[1] + agent.action.move[1]
            if new_x < 0 or new_x > self.map_size - 1:  # 如果超出map范围，就向相反方向同距离移动
                agent.action.move[0] = -agent.action.move[0]
            if new_y < 0 or new_y > self.map_size - 1:
                agent.action.move[1] = -agent.action.move[1]
            agent.position += agent.action.move
            # agent.position = np.array([max(0, agent.position[0]),
            #                            max(0, agent.position[1])])
            # agent.position = np.array([min(self.map_size - 1, agent.position[0]), min(
            #     self.map_size - 1, agent.position[1])])
        if agent.offloading_idle:  # 卸载动作
            agent.action.offloading = act[2]
        print('agent-{} action: move{}, exe{},off{},band{}'.format(agent.no, agent.action.move, agent.action.execution,
                                                                   agent.action.offloading, agent.action.bandwidth))

    """get info used for benchmarking, 返回agent和world"""

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    """get observation for a particular agent"""

    def get_obs(self, agent):
        obs = np.zeros([agent.obs_r * 2 + 1, agent.obs_r * 2 + 1, 2])
        # left up point
        lu = [max(0, agent.position[0] - agent.obs_r),
              min(self.map_size, agent.position[1] + agent.obs_r + 1)]
        # right down point
        rd = [min(self.map_size, agent.position[0] + agent.obs_r + 1),
              max(0, agent.position[1] - agent.obs_r)]

        # ob_map position
        ob_lu = [agent.obs_r - agent.position[0] + lu[0],
                 agent.obs_r - agent.position[1] + lu[1]]
        ob_rd = [agent.obs_r + rd[0] - agent.position[0],
                 agent.obs_r + rd[1] - agent.position[1]]
        # print([lu, rd, ob_lu, ob_rd])
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            # print([i, map_i])
            obs[i][ob_lu[0]:ob_rd[0]] = self.DS_state[map_i][lu[0]:rd[0]]
        # print(self.DS_state[ob_rd[1]][ob_lu[0]:ob_rd[0]].shape)
        agent.obs = obs
        # print(obs.shape)
        return obs

    def get_statemap(self):
        sensor_map = np.ones([self.map_size, self.map_size, 2])
        agent_map = np.ones([self.map_size, self.map_size, 2])
        for sensor in self.sensors:
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][0] = sum([i[0] for i in sensor.data_buffer])
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][1] = sum(
                [i[1] for i in sensor.data_buffer]) / max(len(sensor.data_buffer), 1)
        for agent in self.agents:
            agent_map[int(agent.position[1])][int(agent.position[0])][0] = sum([i[0] for i in agent.done_data])
            agent_map[int(agent.position[1])][int(agent.position[0])][1] = sum([i[1] for i in agent.done_data]) / max(
                len(agent.done_data), 1)
        return sensor_map, agent_map
        # get dones for a particular agent
        # unused right now -- agents are allowed to go beyond the viewing screen

    """获取center agent的状态，返回edge agent的卸载缓冲区数据大小和edge agent位置"""

    def get_center_state(self):
        buffer_list = np.zeros([self.agent_num, 2, self.max_buffer_size])
        pos_list = np.zeros([self.agent_num, 2])
        for i, agent in enumerate(self.agents):
            pos_list[i] = agent.position
            for j, d in enumerate(agent.done_data):  # d: [78.25333937931646, 7, 8]
                buffer_list[i][0][j] = d[0]  # 处理完成的数据缓冲区（卸载缓冲区）数据大小
                buffer_list[i][1][j] = d[1]  # 数据年龄
        # print(buffer_list)
        # print(pos_list)
        return buffer_list, pos_list

    def get_buffer_state(self):
        exe = []
        done = []
        for agent in self.agents:
            exe.append(len(agent.total_data))
            done.append(len(agent.done_data))
        return exe, done

    """完成反馈， 返回agent 和 world"""

    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        return self.done_callback(agent, self.world)

    """数据源平均年龄 average age"""

    def _get_age(self):
        return np.mean(list(self.world.sensor_age.values()))        # 这里返回的是所有数据源的平均年龄，导致每个agent的reward相同

    """get reward for a particular agent"""

    # def _get_reward(self):
    def _get_reward(self):
        # # 方式一： 平均年龄
        # return np.mean(list(self.world.sensor_age.values()))

        # # 方式二： 源代码注释部分
        # state_reward = sum(sum(self.DS_state)) / self.sensor_num
        # # done_reward = [[i[0], i[1]] for i in self.world.finished_data]
        # finished_data = [[i[0], i[1]] for i in self.world.finished_data]        # 卸载到云端的全部数据信息 List[数据大小，数据年龄]
        # if not finished_data:
        #     done_reward = np.array([0, 0])
        # else:
        #     # print(np.array(done_reward))
        #     done_reward = np.average(np.array(finished_data), axis=0)     # 云端数据[大小平均值，年龄平均值]
        # buffer_reward = 0
        # for agent in self.agents:
        #     if agent.done_data:
        #         buffer_reward += np.mean([d[1] for d in agent.done_data])
        # buffer_reward = buffer_reward / self.agent_num      # 卸载缓冲区中数据年龄的平均值
        # # print(buffer_reward)
        # # print([state_reward, done_reward])
        # return self.alpha * done_reward[1] + self.beta * (state_reward[1] + self.sensor_num - self.map_size * self.map_size) + (1 - self.alpha - self.beta) * buffer_reward

        # 方式三：返回总完成任务数，用于接下来计算总时间的平均完成的任务数
        data_nums = len(self.world.finished_data)      # 完成任务的个数
        return data_nums

    """画出环境map： 包括数据源 edge分布情况 """

    def render(self, name=None, epoch=None, save=False):
        # plt.subplot(1,3,1)
        # plt.scatter(self.world.sensor_pos[0],self.world.sensor_pos[1],alpha=0.7)
        # plt.grid()
        # plt.title('sensor position')
        # plt.subplot(1,3,2)
        # plt.scatter(self.world.agent_pos_init[0],self.world.agent_pos_init[1],alpha=0.7)
        # plt.grid()
        # plt.title('agent initial position')
        # plt.subplot(1,3,3)
        plt.figure()
        plt.scatter(self.world.sensor_pos[0], self.world.sensor_pos[1], c='cornflowerblue', alpha=0.9)

        for agent in self.world.agents:
            plt.scatter(agent.position[0], agent.position[1], c='orangered', alpha=0.9)
            plt.annotate(agent.no + 1, xy=(agent.position[0], agent.position[1]),
                         xytext=(agent.position[0] + 0.1, agent.position[1] + 0.1))
            obs_plot = get_circle_plot(agent.position, self.obs_r)
            collect_plot = get_circle_plot(agent.position, self.collect_r)
            plt.fill_between(obs_plot[0], obs_plot[1], obs_plot[2], where=obs_plot[1] > obs_plot[2], color='darkorange',
                             alpha=0.02)
            plt.fill_between(collect_plot[0], collect_plot[1], collect_plot[2], where=collect_plot[1] > collect_plot[2],
                             color='darkorange', alpha=0.05)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Sensors', 'Edge Agents'])
        plt.axis('square')
        plt.xlim([0, self.map_size])
        plt.ylim([0, self.map_size])
        plt.title('all entity position(epoch%s)' % epoch)
        if not save:
            plt.show()
            return
        plt.savefig('%s/%s.png' % (name, epoch))
        plt.close()
        # plt.pause(0.3)
        # plt.show()

    def close(self):
        return None
