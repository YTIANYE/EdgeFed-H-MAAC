# -*- coding: UTF-8 -*-
import numpy as np
import random
import logging

logging.basicConfig(level=logging.WARNING)


class Action(object):
    def __init__(self):
        self.move = None
        self.collect = None
        self.offloading = []  # agent 卸载缓冲区
        self.bandwidth = 0
        self.execution = []  # agent 执行缓冲区


class AgentState(object):
    def __init__(self):
        self.position = None  # [x, y]    记录agent的位置
        self.obs = None  # TODO


class EdgeDevice(object):
    edge_count = 0

    def __init__(self, i, obs_r, pos, spd, collect_r, max_buffer_size, movable=True, mv_bt=0, trans_bt=0):  # pos(x,y,h)
        # self.no = EdgeDevice.edge_count
        self.no = i  # agent 编号，取消静态变量编号方式
        EdgeDevice.edge_count += 1
        self.obs_r = obs_r  # 观察半径
        self.init_pos = pos  # 初始位置
        self.position = pos  # 实时位置
        self.move_r = spd  # 移动速度
        self.collect_r = collect_r  # 收集半径
        self.mv_battery_cost = mv_bt  # 移动电量消耗
        self.trans_battery_cost = trans_bt  # 传输电量消耗
        self.data_buffer = {}  # edge agent 收集的数据的缓冲区， 记录每个数据源和其对应收集来的数据
        self.max_buffer_size = max_buffer_size  # 缓冲区最大尺寸（即最大能够存储的数据个数）
        self.idle = True  # collecting idle
        self.movable = movable  # 是否可以移动
        self.state = AgentState()
        self.action = Action()
        self.done_data = []  # 卸载缓冲区，处理完的数据的缓冲区 List[数据大小、年龄、编号]
        self.offloading_idle = True  # 卸载表示， True需要卸载
        self.total_data = {}  # 每个agent的收集到的总数据 [数据大小，年龄，数据源索引]
        # TODO 数据处理速率
        self.computing_rate = 2e4  # edge端 数据处理速率 20000
        self.computing_idle = True  # 执行缓冲区任务是否已经执行的标志
        self.index_dim = 2  # TODO
        self.collecting_sensors = []
        self.ptr = 0.2
        self.h = 5
        self.noise = 1e-13
        self.trans_rate = 0  # TODO

    def move(self, new_move, h):
        if self.idle:
            self.position += new_move
            self.mv_battery_cost += np.linalg.norm(new_move)

    """获取 total_data     shape = (2, 5)"""

    def get_total_data(self):
        total_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.total_data:
            for j, k in enumerate(list(self.total_data.keys())):
                # print(self.total_data[k])
                total_data_state[0, j] = self.total_data[k][0]
                total_data_state[1, j] = self.total_data[k][1]
        return total_data_state

    def get_done_data(self):
        done_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.done_data:
            for m, k in enumerate(self.done_data):
                done_data_state[0, m] = k[0]  # 数据大小
                done_data_state[1, m] = k[1]  # 数据年龄
        return done_data_state

    def data_update(self, pak):
        if pak[1] in self.data_buffer.keys():
            self.data_buffer[pak[1]].append(pak)
        else:
            self.data_buffer[pak[1]] = [pak]

    def edge_exe(self, tmp_size, t=1):  # one-sum local execution
        if not self.total_data:
            return [0] * self.max_buffer_size
        # age update
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        if len(self.done_data) >= self.max_buffer_size:
            return tmp_size
        # process
        if self.total_data and sum(self.action.execution):
            data2process = [[k, d] for k, d in self.total_data.items()]
            self.computing_idle = False
            if np.argmax(self.action.execution) >= len(data2process):
                self.action.execution = [0] * self.max_buffer_size
                self.action.execution[np.random.randint(len(data2process))] = 1
            for i, data in enumerate(data2process):
                if len(self.done_data) >= self.max_buffer_size:
                    break
                # print([i, tmp_size])
                tmp_size[i] += min(self.total_data[data2process[i][0]][0],
                                   self.computing_rate * self.action.execution[i] * t)
                self.total_data[data2process[i][0]][0] -= self.computing_rate * self.action.execution[i] * t
                if self.total_data[data2process[i][0]][0] <= 0:
                    self.total_data[data2process[i][0]][0] = tmp_size[i]
                    self.done_data.append(self.total_data[data2process[i][0]])
                    self.total_data.pop(data2process[i][0])
                    tmp_size[i] = 0
        return tmp_size

    """edge 执行缓冲区处理数据的过程"""

    def process(self, tmp_size, t=1):  # one-hot local execution
        if not self.total_data:  # 没有数据
            return 0
        # age update
        for k in self.total_data.keys():
            self.total_data[k][1] += t  # 更改年龄
        if len(self.done_data) >= self.max_buffer_size:
            return 0
        # process
        if self.total_data and sum(self.action.execution):  # 有数据和执行指令
            data2process = [[k, d] for k, d in self.total_data.items()]  # 要进行处理的数据

            if self.action.execution.index(1) >= len(data2process):  # 执行指令的索引超过了执行数据总个数，重新在data2process中选则要执行的数据
                self.action.execution[self.action.execution.index(1)] = 0
                self.action.execution[np.random.randint(len(data2process))] = 1  # 重新随机选择下一个要执行的任务
                # print(self.action.execution)
            self.computing_idle = False  # 执行完毕（还是失败）
            # 调试用
            if self.total_data[data2process[self.action.execution.index(1)][0]][0] >= 20000:
                print("数据大小大于20000的", self.total_data[data2process[self.action.execution.index(1)][0]][0])
            tmp_size += min(self.total_data[data2process[self.action.execution.index(1)][0]][0],
                            self.computing_rate * t)  # 如果要处理的数据大小超过了该时间段内的处理能力，新增加的空间大小则是这段时间里最大处理能力所能处理的数据大小
            self.total_data[data2process[self.action.execution.index(1)][0]][
                0] -= self.computing_rate * t  # 该时间段内 需要处理的数据量减少
            if self.total_data[data2process[self.action.execution.index(1)][0]][0] <= 0:  # 数据全部处理
                self.total_data[data2process[self.action.execution.index(1)][0]][
                    0] = tmp_size  # 这里没有问题，本身就是这样设计的：大于20000的数据在减去20000之后，这个20000可能会加入到下一step执行的某个小于20000的任务（由于下一步执行任务随机，不一定是被减去20000的那个任务），作为一个完成的任务放在done_data中，年龄不受影响，但是两个源自不同数据源的任务的数据大小交换了20000，
                self.done_data.append(self.total_data[data2process[self.action.execution.index(1)][0]])
                self.total_data.pop(data2process[self.action.execution.index(1)][0])  # 这一数据被处理完
                tmp_size = 0
                # tmp_size含义：如果一个数据源的某个任务数据超过20000，需要处理多次，tmp_size记录处理过的数据量，当同一阿哥明天的total_data中的某个数据大小小于20000，处理完该任务，连同之前处理的n个20000加在一起等待卸载，tmp_size记录的就是该agent处理完的数据大小，不区分数据源
        # 调试用
        if tmp_size > 0:
            print(tmp_size)
        return tmp_size  # 数据处理完了 返回0，否则返回暂存区大小      # 因为处理能力20000比较大，通常会返回0


def agent_com(agent_list):
    age_dict = {}
    for u in agent_list:
        for k, v in u.data_buffer.items():
            if k not in age_dict:
                age_dict[k] = v[-1][1]
            elif age_dict[k] > v[-1][1]:
                age_dict[k] = v[-1][1]
    return age_dict


# 数据源
class Sensor(object):
    sensor_cnt = 0
    sensor_data_buffer_max = 10  # 数据源缓冲区最大容量（data_buffer_max个小任务）
    data_buffer_min = 2  # 数据源缓冲区至少2个小任务才能被收集

    def __init__(self, i, pos, data_rate, bandwidth, max_ds, lam=0.5, weight=1):
        # def __init__(self, i, pos, data_rate, bandwidth, max_ds, lam=0.5, weight=1, *sensor_data_buffer_max):
        # self.no = Sensor.sensor_cnt
        self.no = i
        Sensor.sensor_cnt += 1
        self.position = pos
        self.weight = weight
        self.data_rate = data_rate  # generate rate # 数据生成速率
        self.bandwidth = bandwidth
        self.trans_rate = 8e3  # to be completed        #TODO
        self.data_buffer = []  # 数据源端的缓冲区
        self.max_data_size = max_ds
        self.data_state = bool(len(self.data_buffer))  # 缓冲区是否有数据
        self.collect_state = False  # 数据源的收集状态
        self.lam = lam  # 1000
        self.noise_power = 1e-13 * self.bandwidth
        self.gen_threshold = 0.3
        # self.sensor_data_buffer_max = sensor_data_buffer_max    # 可能是上限也可能是None

    """数据源生成数据"""

    def data_gen(self, t=1):
        # 更新年龄 update age
        if self.data_buffer:
            for i in range(len(self.data_buffer)):
                self.data_buffer[i][1] += t
        # TODO 生成新数据
        new_data = self.data_rate * np.random.poisson(self.lam)  # 数据大小服从泊松分布 lam-发生率或已知次数
        # new_data = min(new_data, self.max_data_size)
        if new_data >= self.max_data_size or random.random() >= self.gen_threshold:  # 数据过大抛弃新生成的数据，以一定的概率抛弃生成的数据
            return
        if new_data:
            self.data_buffer.append([new_data, 0, self.no])  # 数据大小， 年龄， 数据源编号
            self.data_state = True  # 数据状态，数据源的缓冲区是否含有数据，是否能被收集
            # 缓冲区下限
            # if len(self.data_buffer) >= 20:
            #     self.data_state = True  # 数据状态，数据源的缓冲区是否含有数据，是否能被收集
            # 缓冲区上限
            # if self.sensor_data_buffer_max and len(self.data_buffer) > self.sensor_data_buffer_max:
            if len(self.data_buffer) > self.sensor_data_buffer_max:
                self.data_buffer.pop(0)  # 去除旧数据

        # 生成数据设置了单个小任务的数据大小，但是数据源的数据缓存是无限的，edge收集单个数据源的数据量是无限的


collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                            'urban': (9.61, 0.16, 1, 20),
                            'dense-urban': (12.08, 0.11, 1.6, 23),
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}

collecting_params = collecting_channel_param['urban']
a = collecting_params[0]
b = collecting_params[1]
yita0 = collecting_params[2]
yita1 = collecting_params[3]
carrier_f = 2.5e9


def collecting_rate(sensor, agent):
    d = np.linalg.norm(np.array(sensor.position) - np.array(agent.position))
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / d) - a)))
    L = Pl * yita0 + yita1 * (1 - Pl)
    gamma = agent.ptr_col / (L * sensor.noise_power ** 2)
    rate = sensor.bandwidth * np.log2(1 + gamma)
    return rate


"""收集数据的过程"""


def data_collecting(sensors, agent, hovering_time):
    for k in agent.total_data.keys():
        agent.total_data[k][1] += 1  # 年龄 +1
    if agent.idle and (len(agent.total_data.keys()) < agent.max_buffer_size):  # 允许收集数据 且 数据缓冲区没满
        # obs_sensor = []
        data_properties = []
        # for k in agent.data_buffer.keys():
        #     for i, d in enumerate(agent.data_buffer[k]):
        #         agent.data_buffer[k][i][1] += 1
        for sensor in sensors:
            if not sensor.data_buffer:  # 数据源缓冲区 空
                continue
            if (np.linalg.norm(np.array(sensor.position) - np.array(agent.position)) <= agent.collect_r) and not (
                    sensor.collect_state) and not (
                    sensor.no in agent.total_data.keys()):  # 收集范围内 数据源未被收集 edge没有收集当前数据源的数据
                sensor.collect_state = True
                agent.collecting_sensors.append(sensor.no)  # edge agent 收集列表 添加该数据源
                agent.idle = False  # 不能收集数据了，即正在传输数据
                if len(agent.total_data.keys()) >= agent.max_buffer_size:
                    continue
                # obs_sensor.append(sensor)
                # if not (sensor.no in agent.data_buffer.keys()):
                #     agent.data_buffer[sensor.no] = []
                tmp_size = 0  # 计算数据源缓冲区中全部任务的数据总大小，然后上传到agent
                # trans_rate = collecting_rate(sensor, agent)
                for data in sensor.data_buffer:  # 数据源的缓冲区加入数据      # TODO 考虑做一个数量上或数据总量上的限制
                    tmp_size += data[0]
                    # data[1] += tmp_size / self.trans_rate  # age update
                if sensor.no in agent.data_buffer.keys():  # edge agent 端收集到数据
                    agent.data_buffer[sensor.no].append(tmp_size)
                else:
                    agent.data_buffer[sensor.no] = [tmp_size]
                data_properties.append(tmp_size / sensor.trans_rate)  # 传输所需要的时间
                agent.total_data[sensor.no] = [tmp_size, sensor.data_buffer[-1][1],
                                               sensor.no]  # edge agent 添加总数据  # TODO 年龄为什么是最后一个任务的年龄，即不断取最新的任务的年龄作为收集自这一数据源的所有数据的年龄
                # agent.total_data[sensor.no] = [tmp_size, np.average([x[1] for x in sensor.data_buffer]), sensor.no]
                sensor.data_buffer = []

        if data_properties:
            hovering_time = max(data_properties)  # 选择传输时间最长的那一个
            # print([data_properties, hovering_time])
            return hovering_time  # 悬挂时间，即正在传输数据消耗的时间
        else:
            return 0
    # 结束数据收集 finish collection
    elif not agent.idle:
        hovering_time -= 1
        if hovering_time <= 0:
            agent.idle = True  # 可以收集数据了，传输完毕
            for no in agent.collecting_sensors:
                sensors[no].collect_state = False
            agent.collecting_sensors = []
            hovering_time = 0
        return hovering_time  # 悬停时间
    else:
        return 0


"""edge 的卸载过程"""


def offloading(agent, center_pos, t=1):
    if not agent.done_data:
        # print('no done')
        return False, {}
    for data in agent.done_data:  # data[947, 1, 6]       数据大小、年龄、编号
        data[1] += t  # 年龄增加

    if sum(agent.action.offloading):  # 有需要卸载的任务
        if agent.action.offloading.index(1) >= len(
                agent.done_data):  # 如果指令不符合，重新选择卸载操作    # list.index(obj) 从列表中找出某个值第一个匹配项的索引位置
            agent.action.offloading[agent.action.offloading.index(1)] = 0
            agent.action.offloading[np.random.randint(len(agent.done_data))] = 1
        agent.offloading_idle = False  # 卸载失败
        dist = np.linalg.norm(np.array(agent.position) - np.array(center_pos))
        agent.trans_rate = trans_rate(dist, agent)  # 传输速率 # to be completed
    else:  # 没有需要卸载的任务
        return False, {}
    # print(agent.done_data)
    # print(agent.action.offloading)
    agent.done_data[agent.action.offloading.index(1)][0] -= agent.trans_rate * t
    if agent.done_data[agent.action.offloading.index(1)][0] <= 0:  # 数据全部卸载完成
        sensor_indx = agent.done_data[agent.action.offloading.index(1)][2]  # 数据源索引
        sensor_aoi = agent.done_data[agent.action.offloading.index(1)][1]  # 年龄
        sensor_data = agent.data_buffer[sensor_indx][0]  # 数据大小
        del agent.data_buffer[sensor_indx][0]
        del agent.done_data[agent.action.offloading.index(1)]
        # return finish flag & total data
        agent.offloading_idle = True  # 卸载成功
        return True, {sensor_indx: [sensor_data, sensor_aoi]}  # 数据源索引:[数据大小， 年龄]
    return False, {}


# 传输速率
def trans_rate(dist, agent):  # to be completed
    W = 1e6 * agent.action.bandwidth
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / dist) - a)))
    fspl = (4 * np.pi * carrier_f * dist / (3e8)) ** 2
    L = Pl * fspl * 10 ** (yita0 / 20) + 10 ** (yita1 / 20) * fspl * (1 - Pl)
    rate = W * np.log2(1 + agent.ptr / (L * agent.noise * W))
    print('agent-{} rate: {},{},{},{},{}'.format(agent.no, dist, agent.action.bandwidth, Pl, L, rate))
    return rate


class MEC_world(object):
    # def __init__(self, map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size=1, sensor_lam=0.5, *sensor_data_buffer_max):
    def __init__(self, map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size=1, sensor_lam=0.5):
        self.agents = []  # 存放agent
        self.sensors = []  # 存放传感器、数据源
        self.map_size = map_size
        self.center = (map_size / 2, map_size / 2)  # 地图环境的中心点坐标
        self.sensor_count = sensor_num
        # self.sensor_data_buffer_max = sensor_data_buffer_max[0] if sensor_data_buffer_max else None         # 数据源缓冲区最大限制（单位：小任务个数）
        self.agent_count = agent_num
        self.max_buffer_size = max_size  # 缓冲区最大尺寸（即最大能够存储的数据个数）# 收集数据和执行数据的最大缓冲区大小
        sensor_bandwidth = 1000  # 数据源带宽
        max_ds = sensor_lam * 2  # 2000 # TODO:
        data_gen_rate = 1  # 数据生成速率
        self.offloading_slice = 1  # 卸载切片
        self.execution_slice = 1  # 执行切片
        self.time = 0
        self.DS_map = np.zeros([map_size, map_size])  # 地图：数据源的位置    # zeros() 返回来一个给定形状和类型的用0填充的数组
        self.DS_state = np.ones([map_size, map_size, 2])  # 数据源状态 前两个维度是位置坐标，最后维度为2表示存储数据大小和数据年龄
        self.hovering_list = [0] * self.agent_count  # 悬停agent列表，记录每个agent的悬停时间，即正在收集数据的时间
        self.tmp_size_list = [0] * self.agent_count  # 存放每个agent正在处理的数据大小
        # [self.tmp_size_list.append([0] * self.max_buffer_size) for i in range(self.agent_count)]
        self.offloading_list = []  # 卸载列表
        # TODO finished_data
        self.finished_data = []  # 彻底处理完毕（卸载到云端）的数据列表
        self.obs_r = obs_r  # 观察半径
        self.move_r = speed  # 移动半径
        self.collect_r = collect_r  # 收集覆盖半径
        self.sensor_age = {}  # 所有数据源的数据的年龄
        """数据源初始化 random.seed(7)"""
        self.sensor_pos = [
            random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num),
            # k 定义返回列表长度的整数
            random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))],
                           k=sensor_num)]  # list[2][30] 30个数据源的(x, y)
        # self.sensor_pos = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.5 * self.map_size))], k=int(sensor_num / 2)) + random.choices(
        #     [i for i in range(int(0.5 * self.map_size), int(0.9 * self.map_size))], k=int(sensor_num / 2)), random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        for i in range(sensor_num):
            self.sensors.append(
                # Sensor(i, np.array([self.sensor_pos[0][i], self.sensor_pos[1][i]]), data_gen_rate, sensor_bandwidth,
                #        max_ds, lam=sensor_lam, self.sensor_data_buffer_max))
                Sensor(i, np.array([self.sensor_pos[0][i], self.sensor_pos[1][i]]), data_gen_rate, sensor_bandwidth,
                       max_ds, lam=sensor_lam))
            self.sensor_age[i] = 0
            self.DS_map[self.sensor_pos[0][i], self.sensor_pos[1][i]] = 1
        """边缘设备初始化"""
        self.agent_pos_init = [
            random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num),
            random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num)]
        for i in range(agent_num):
            self.agents.append(
                EdgeDevice(i, self.obs_r, np.array([self.agent_pos_init[0][i], self.agent_pos_init[1][i]]), speed,
                           collect_r, self.max_buffer_size))
            # EdgeDevice(self.obs_r, np.array([self.agent_pos_init[0][i], self.agent_pos_init[1][i]]), speed,
            #            collect_r, self.max_buffer_size))

    """一个step包含数据年龄更新，数据源生成新数据，edge agent 收集处理卸载数据"""

    def step(self):
        # update sensor age
        for k in self.sensor_age.keys():
            self.sensor_age[k] += 1
        # data generation & DS_state update
        logging.info("data generation")
        for sensor in self.sensors:
            sensor.data_gen()  # 数据源生成数据， 有一定概率不生成数据
            if sensor.data_buffer:
                data_size = sum(i[0] for i in sensor.data_buffer)  # 该数据源总数据大小
                # update data source state, note that the (x,y) is reversed to the matrix index (i,j)
                self.DS_state[sensor.position[1], sensor.position[0]] = [  # 最小的维度2 存储内容
                    data_size, sensor.data_buffer[0][1]]  # 数据大小和年龄

        # edge process  offloading collect
        logging.info("edge operation")
        age_dict = {}  # 记录处理完的数据的年龄
        for i, agent in enumerate(self.agents):
            # 调试用
            # if self.tmp_size_list[i] >= 20000:
            #     print("tmp_size_list大于20000", self.tmp_size_list[i])

            # 执行过程  edge process
            self.tmp_size_list[i] = agent.process(self.tmp_size_list[i])  # tmp_size_list 每个agent正在处理的数据大小
            # 卸载过程  offloading
            finish_flag, data_dict = offloading(agent, self.center)  # data-dict {数据源索引:[数据大小， 年龄]}
            # update reward state
            # print([i, finish_flag, data_dict])
            if finish_flag:  # 如果成功卸载，记录处理完的数据的年龄
                for sensor_id, data in data_dict.items():
                    self.finished_data.append([data[0], data[1], sensor_id])
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append(data[1])  # 记录处理完的数据的年龄
                    else:
                        age_dict[sensor_id] = [data[1]]
                    # self.sensor_age[sensor_id] -=data[1]
            # collecting
            self.hovering_list[i] = data_collecting(self.sensors, agent, self.hovering_list[i])
            # print(self.hovering_list[i])
        for id in age_dict.keys():
            self.sensor_age[id] = sorted(age_dict[id])[0]  # 数据源的数据的年龄
        print('hovering:{}'.format(self.hovering_list))  # 打印每个agent的悬停时间
