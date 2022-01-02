import gym
import numpy as np
import random
from MEC_env import mec_def
from MEC_env import mec_env
from Params import *
import AC_agent
import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
from matplotlib import pyplot as plt
import json
import time
from print_logs import *


# 传入数据源个数
# def run(sensor_num):
def run(conditions):
    sensor_num = conditions["sensor_num"]
    sample_method = conditions["sample_method"]
    np.random.seed(map_seed)
    random.seed(map_seed)
    tf.random.set_seed(rand_seed)
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
    AC = AC_agent.ACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon, sample_method)

    """训练开始"""
    # 记录环境参数
    m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    f = open('logs/hyperparam/%s.json' % m_time, 'w')
    json.dump(params, f)
    f.close()

    # 记录控制台日志
    f_print_logs = PRINT_LOGS(m_time).open()
    print("运行程序：AC_run")
    print("运行程序：AC_run", file=f_print_logs)

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


# # 实验1：研究数据源个数对reward趋势的影响
# def experiment_1():
#     """
#     sample方式一
#     变量：数据源个数
#     """
#     # sensor_nums = [60, 50, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#     # sensor_nums = [50, 50, 50, 50, 60, 60, 60, 60]
#     # sensor_nums = [50, 50, 50, 50, 50, 60, 60, 60, 60, 60]
#     # for i in range(len(sensor_nums)):
#     #     print("sensor_num:", sensor_num)
#     #     run(sensor_nums[i])
#
#     """
#     研究最近平均任务数
#     sample方式二
#     变量：数据源个数
#     """
#     # sensor_nums = [60, 60, 60, 60, 60]  # 最近200 epoch_num = 200
#     sensor_nums = [60, 60, 60, 60]  # 最近200 epoch_num = 200
#     for i in range(len(sensor_nums)):
#         print("sensor_num:", sensor_num)
#         run(sensor_nums[i])
#
#
# # 实验3：研究联合优化：最近平均任务数 和 数据平均年龄
# def experiment_3():
#     """
#     变量：数据源个数
#     """
#     sensor_nums = [60, 60, 60, 60, 60]
#     for i in range(len(sensor_nums)):
#         print("sensor_num:", sensor_nums[i])
#         run(sensor_nums[i])
#
#
# # 实验4：研究数据平均年龄
# def experiment_4():
#     """
#     变量：数据源个数
#     """
#     sensor_nums = [60, 60, 60, 60, 60]
#     for i in range(len(sensor_nums)):
#         print("sensor_num:", sensor_nums[i])
#         run(sensor_nums[i])

def experiment_5():
    """
    变量：数据源个数
    """
    sensor_nums = [60, 60, 60, 60, 60]

    # sensor_nums = [40, 40, 40, 40, 40,
    #                50, 50, 50, 50, 50,
    #                70, 70, 70, 70, 70,
    #                80, 80, 80, 80, 80]
    sample_methods = [1, 2]  # 默认方式二 # 采样方式一 1；    采样方式二 2
    for sample in sample_methods:
        for i in range(len(sensor_nums)):
            conditions = {'sensor_num': sensor_nums[i], 'sample_method': sample}
            print("sensor_num:", sensor_nums[i])
            run(conditions)


def AC_run():
    print("运行程序：AC_run")

    """实验运行"""
    # 实验1：研究数据源个数对reward趋势的影响
    # experiment_1()

    # 实验3：研究联合优化：最近平均任务数 和 数据平均年龄
    # experiment_3()

    # 实验4：研究数据平均年龄
    # experiment_4()
    experiment_5()
    """测试运行"""
    # run(60)
    # run(60)
