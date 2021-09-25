import tensorflow as tf
import numpy as np
import random

map_size = 200
# agent_num = 4
# sensor_num = 30
agent_num = 8
sensor_num = 60
obs_r = 60  # 观察半径
collect_r = 40  # 收集覆盖半径
speed = 6  # 移动半径
max_size = 5  # 收集数据和执行数据的最大缓冲区大小
sensor_lam = 1e3  # 1000 # 泊松分布 lam-发生率或已知次数

# 测试周期：经过大量实验实例观察一般2k个epoch开始趋于稳定，故实验周期设置为3k
MAX_EPOCH = 3
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
tf.random.set_seed(rand_seed) # TODO 随机种子的位置有没有问题

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
    # 'FL': FL,
    'FL_omega': FL_omega
}
