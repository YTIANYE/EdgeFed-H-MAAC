from MEC_env import mec_def
from MEC_env import mec_env
from Params import *
import MAAC_agent

import tensorflow as tf
import datetime

from matplotlib import pyplot as plt
import json
import time
from print_logs import *

FL = False
# params['FL'] = FL


# 传入数据源缓冲区上限
# def run(sensor_data_buffer_max):
# 传入数据源个数
def run(sensor_num):
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
    研究总平均任务数
    sample方式二
    变量：数据源个数
    """
    # 可能是添加上限的实验或只是sample方式二
    # sensor_nums = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

    # sample方式二
    # sensor_nums = [50, 60, 70, 130, 30, 20, 40, 80, 90, 100, 110, 120]  # 140
    # sensor_nums = [60, 60, 60, 60, 50, 50, 50, 50]
    # sensor_nums = [60, 60, 60, 60, 60]

    """
    研究最近平均任务数
    sample方式二
    变量：数据源个数
    """
    # FL = False
    sensor_nums = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]  # 最近200 epoch_num = 200
    for i in range(len(sensor_nums)):
        print("sensor_num:", sensor_nums[i])
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


def MAAC_run():
    print("运行程序：MAAC_run")

    """实验运行"""
    # 实验1：研究数据源个数对reward趋势的影响
    # experiment_1()

    # 实验2：研究上下限对reward趋势的影响
    # experiment_2()

    # 下一步计划
    # 测试reward方式四的情况，并且做上下限，
    # reward的有没有明显提高，根据设限之后的reward增长趋势，应该会提高并且平稳

    """测试运行"""
    run(60)
