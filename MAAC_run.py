import gym
import numpy as np
import random
from MEC_env import mec_def
from MEC_env import mec_env
import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
import MAAC_agent
from matplotlib import pyplot as plt
import json
import time

map_size = 200
# agent_num = 4
# sensor_num = 30
agent_num = 8
sensor_num = 60  # sensor_num = 20 30 40 50 60 70 80 90 100 110 120
obs_r = 60  # 观察半径
collect_r = 40  # 收集覆盖半径
speed = 6  # 移动半径
max_size = 5  # 收集数据和执行数据的最大缓冲区大小
sensor_lam = 1e3  # 1000 # 泊松分布 lam-发生率或已知次数

# 测试周期：经过大量实验实例观察一般2k个epoch开始趋于稳定，故实验周期设置为3k
MAX_EPOCH = 3000
MAX_EP_STEPS = 200
# 大周期
# MAX_EPOCH = 5000
# MAX_EP_STEPS = 200
# 小周期
# MAX_EPOCH = 500
# MAX_EP_STEPS = 20
# 测试短周期
# MAX_EPOCH = 100
# MAX_EP_STEPS = 5
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.85  # reward discount
TAU = 0.8  # soft replacement  目标更新权重
BATCH_SIZE = 128
alpha = 0.9  #
beta = 0.1  #
aggregate_reward = False  # edge 是否共用sum_reward， 源码默认False
# aggregate_reward = True
Epsilon = 0.2  # Probability of random exploration      # Epsilon = 0.1  # Epsilon = 0.4  # Epsilon = 0.05
up_freq = 8  # 目标网络更新频率 每up_freq个epoch更新一次
render_freq = 32
FL = True  # 控制是否联合学习的开关，默认True
# FL = False
FL_omega = 0.5  # todo 关于联合学习因子其他情况还没有进行实验
# random seeds are fixed to reproduce the results
map_seed = 1
rand_seed = 17
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)


# 记录控制台日志
class PRINT_LOGS:
    def __init__(self, m_time):
        # m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.m_time = m_time

    def open(self):
        logs = open('logs/print_logs/%s.txt' % self.m_time, 'a')  # 'w'覆盖 'a'追加
        return logs


"""运行"""


# 传入数据源个数
# def run(sensor_num):
# 传入数据源缓冲区上限
def run(sensor_data_buffer_max):
    # 选取GPU
    print("TensorFlow version: ", tf.__version__)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # 获得当前主机上特定运算设备的列表
    plt.rcParams['figure.figsize'] = (9, 9)  # 设置figure_size尺寸
    # logdir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    """初始化"""
    mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam)
    # 设定数据源缓冲区上下限的初始化方式
    # mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam, sensor_data_buffer_max)
    # env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta)
    env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta, aggregate_reward=aggregate_reward)
    # 建立模型
    MAAC = MAAC_agent.MAACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

    """训练开始"""
    # 记录环境参数
    params = {
        'map_size': map_size,
        'agent_num': agent_num,
        'sensor_num': sensor_num,
        'obs_r': obs_r,  # 观察半径
        'collect_r': collect_r,  # 收集覆盖半径
        'speed': speed,  # 移动半径
        'max_size': max_size,  # 收集数据和执行数据的最大缓冲区大小
        'sensor_lam': sensor_lam,

        'MAX_EPOCH': MAX_EPOCH,
        'MAX_EP_STEPS': MAX_EP_STEPS,
        'LR_A': LR_A,  # learning rate for actor
        'LR_C': LR_C,  # learning rate for critic
        'GAMMA': GAMMA,  # reward discount
        'TAU': TAU,  # soft replacement  目标更新权重。
        'BATCH_SIZE': BATCH_SIZE,
        # 'alpha': alpha,
        # 'beta': beta,
        'alpha': alpha,
        'beta': beta,
        'aggregate_reward': aggregate_reward,
        'Epsilon': Epsilon,  # Probability of random exploration.
        'learning_seed': rand_seed,
        'env_seed': map_seed,
        'up_freq': up_freq,
        'render_freq': render_freq,
        'FL': FL,
        'FL_omega': FL_omega
    }
    m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    f = open('logs/hyperparam/%s.json' % m_time, 'w')
    json.dump(params, f)
    f.close()

    # 记录控制台日志
    f_print_logs = PRINT_LOGS(m_time).open()

    # 记录开始时间
    startTime = time.time()
    print("开始时间:", time.localtime(startTime))
    print("开始时间:", time.localtime(startTime), file=f_print_logs)

    # 训练过程
    MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL,
               FL_omega=FL_omega)

    # 统计执行时间
    endTime = time.time()
    t = endTime - startTime
    print("开始时间:", time.localtime(startTime))
    print("结束时间:", time.localtime(endTime))
    print("运行时间(分钟)：", t / 60)
    print("开始时间:", time.localtime(startTime), file=f_print_logs)
    print("结束时间:", time.localtime(endTime), file=f_print_logs)
    print("运行时间(分钟)：", t / 60, file=f_print_logs)

    # 关闭记录控制台日志
    f_print_logs.close()


# 实验1：研究数据源个数对reward趋势的影响
def experiment_1():
    """
    sample方式二
    变量：数据源个数
    """
    # 可能是添加上限的实验或只是sample方式二
    # sensor_nums = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    # sample方式二
    sensor_nums = [50, 60, 70, 130, 30, 20, 40, 80, 90, 100, 110, 120]  # 140
    for i in range(len(sensor_nums)):
        print("sensor_num:", sensor_num)
        run(sensor_nums[i])


# 实验2：研究上下限对reward趋势的影响
def experiment_2():
    """
    数据源缓冲区上限、数据源缓冲区下限、收集上限、收集下限
    """
    """
    定量：数据源缓冲区上限 data_buffer_max = 10
    变量：数据源个数 100 60 50 70 80 90 110 40 30 
    """
    # run(60)  # 100
    """
    定量：数据源个数 50
    变量：数据源缓冲区上限 data_buffer_max = 10 20 40 100 200
    """
    sensor_data_buffer_maxs = [20, 40, 100, 200]
    for sensor_data_buffer_max in sensor_data_buffer_maxs:
        # 记录控制台日志
        f_print_logs = PRINT_LOGS(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')).open()
        print("sensor_data_buffer_max:", sensor_data_buffer_max)
        print("sensor_data_buffer_max:", sensor_data_buffer_max, f_print_logs)
        # 关闭记录控制台日志
        f_print_logs.close()

        run(sensor_data_buffer_max)


# 实验3

if __name__ == "__main__":
    """实验运行"""
    # 实验1：研究数据源个数对reward趋势的影响
    # experiment_1()

    # 实验2：研究上下限对reward趋势的影响
    # experiment_2()

    # 下一步计划
    # 测试reward方式四的情况，并且做上下限，
    # reward的有没有明显提高，根据设限之后的reward增长趋势，应该会提高并且平稳

    """测试运行"""
    # run(100)
