import gym
import numpy as np
import random
from MEC_env import mec_def
from MEC_env import mec_env
import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
import AC_agent
from matplotlib import pyplot as plt
import json
import time
from print_logs import *

map_size = 200
# agent_num = 4
# sensor_num = 30
agent_num = 8
sensor_num = 60
obs_r = 60
collect_r = 40
speed = 6
max_size = 5
sensor_lam = 1e3

# MAX_EPOCH = 5000
MAX_EPOCH = 3000
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.85  # reward discount
TAU = 0.8  # soft replacement
BATCH_SIZE = 128
alpha = 0.9
beta = 0.1
Epsilon = 0.2
# random seeds are fixed to reproduce the results
map_seed = 1
rand_seed = 17
up_freq = 8
render_freq = 32
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)   # TODO 随机种子的位置有没有问题

"""运行"""


# 传入数据源个数
def run(sensor_num):
    # 选取GPU
    print("TensorFlow version: ", tf.__version__)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    plt.rcParams['figure.figsize'] = (9, 9)
    # logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    """初始化"""
    mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam)
    env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta)
    AC = AC_agent.ACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

    """训练开始"""
    # 记录环境参数
    params = {
        'map_size': map_size,
        'agent_num': agent_num,
        'sensor_num': sensor_num,
        'obs_r': obs_r,
        'collect_r': collect_r,
        'speed': speed,
        'max_size': max_size,
        'sensor_lam': sensor_lam,

        'MAX_EPOCH': MAX_EPOCH,
        'MAX_EP_STEPS': MAX_EP_STEPS,
        'LR_A': LR_A,
        'LR_C': LR_C,
        'GAMMA': GAMMA,
        'TAU': TAU,
        'BATCH_SIZE': BATCH_SIZE,
        # 'alpha': alpha,
        # 'beta': beta,
        'Epsilon': Epsilon,
        'learning_seed': rand_seed,
        'env_seed': map_seed,
        'up_freq': up_freq,
        'render_freq': render_freq
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
    AC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq)

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
    sample方式一
    变量：数据源个数
    """
    # sensor_nums = [60, 50, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # sensor_nums = [50, 50, 50, 50, 60, 60, 60, 60]
    # sensor_nums = [50, 50, 50, 50, 50, 60, 60, 60, 60, 60]
    # for i in range(len(sensor_nums)):
    #     print("sensor_num:", sensor_num)
    #     run(sensor_nums[i])

    """
    研究最近平均任务数
    sample方式二
    变量：数据源个数
    """
    # sensor_nums = [60, 60, 60, 60, 60]  # 最近200 epoch_num = 200
    sensor_nums = [60, 60, 60, 60]  # 最近200 epoch_num = 200
    for i in range(len(sensor_nums)):
        print("sensor_num:", sensor_num)
        run(sensor_nums[i])


def AC_run():
    """实验运行"""
    # 实验1：研究数据源个数对reward趋势的影响
    experiment_1()

    """测试运行"""
    # run(60)

"""下一步计划"""

"""
改正sample方式
"""
