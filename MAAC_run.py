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

# 记录开始时间
startTime = time.time()
print("开始时间:", time.localtime(startTime))
# 选取GPU
print("TensorFlow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # 获得当前主机上特定运算设备的列表
plt.rcParams['figure.figsize'] = (9, 9)  # 设置figure_size尺寸
# logdir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

map_size = 200
agent_num = 4
sensor_num = 30
obs_r = 60  # 观察半径
collect_r = 40  # 收集覆盖半径
speed = 6  # 移动半径
max_size = 5  # 收集数据和执行数据的最大缓冲区大小
sensor_lam = 1e3  #

MAX_EPOCH = 5000
# MAX_EPOCH = 500
MAX_EP_STEPS = 200
# MAX_EP_STEPS = 20
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.85  # reward discount
TAU = 0.8  # soft replacement  目标更新权重
BATCH_SIZE = 128
alpha = 0.9  #
beta = 0.1  #
aggregate_reward = False        # edge 是否共用sum_reward， 源码默认False
# aggregate_reward = True
Epsilon = 0.2  # Probability of random exploration
# random seeds are fixed to reproduce the results
map_seed = 1
rand_seed = 17
up_freq = 8  # 目标网络更新频率 每up_freq个epoch更新一次
render_freq = 32
FL = True  # 控制是否联合学习的开关，默认True
# FL = False
FL_omega = 0.5
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)

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

mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam)
# env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta)
env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta, aggregate_reward=aggregate_reward)
# 建立模型
MAAC = MAAC_agent.MAACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)
# 记录环境参数
m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
f = open('logs/hyperparam/%s.json' % m_time, 'w')
json.dump(params, f)
f.close()
# 训练
MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL, FL_omega=FL_omega)

# 统计执行时间
print("开始时间:", time.localtime(startTime))
endTime = time.time()
print("结束时间:", time.localtime(endTime))
t = endTime - startTime
print("运行时间：", t)
