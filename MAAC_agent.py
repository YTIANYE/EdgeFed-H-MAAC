import tensorflow as tf
# from keras.losses import huber_loss
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io as sio
import gym
import time
import random
import datetime
import os
import imageio
import glob
import tqdm
import json
import platform

# tf.random.set_seed(11)

# tf.keras.backend.set_floatx('float64')

"""返回 可移动范围内的全部坐标点[y, x] 及个数 """


def discrete_circle_sample_count(n):
    count = 0
    move_dict = {}
    for x in range(-n, n + 1):  # [-n, n]
        y_l = int(np.floor(np.sqrt(n ** 2 - x ** 2)))  # np.floor 向下取整
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict


def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.math.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


"""构建 agent actor net"""


# inputs: state map, pos, buffer, operation, bandwidth;
# outputs: move, operation
def agent_actor(input_dim_list, cnn_kernel_size, move_r):  # input_dim_list [(121, 121, 2), (2, 5), (2, 5), 1]
    state_map = keras.Input(shape=input_dim_list[0])
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])  # 执行缓冲区
    done_buffer = keras.Input(shape=input_dim_list[2])  # 完成缓冲区
    bandwidth = keras.Input(shape=input_dim_list[3])
    # CNN for map
    cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(
        state_map)  # 2D卷积层 # filters ，kernel_size过滤器个数和卷积核尺寸, padding取值为 “same”，表示边缘用0填充
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(
        cnn_map)  # 2D输入的平均池层(如图像)    # pool_size 指定池窗口的大小；可以是单个整数,以指定所有空间维度的相同值.
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)  # 在dropout后保持原始值的均值和方差不变.
    move_out = layers.Dense(1, activation='relu')(cnn_map)
    # move_out = move_out / tf.reduce_sum(move_out, [1, 2, 3], keepdims=True)
    # move_out = tf.exp(move_out) / tf.reduce_sum(tf.exp(move_out), [1, 2, 3], keepdims=True)

    # cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(cnn_map)
    # cnn_map = layers.MaxPooling2D(pool_size=cnn_kernel_size)(cnn_map)
    # cnn_map = layers.Dropout(0.2)(cnn_map)
    # cnn_output = layers.Flatten()(cnn_map)
    # cnn_output = layers.Dense(128, activation='relu')(cnn_output)
    # move_dist = layers.Dense(move_count, activation='softmax')(move_out)

    # operation
    # total_mlp = layers.Dense(2, activation='relu')(total_buffer)
    # done_mlp = layers.Dense(2, activation='relu')(done_buffer)
    # buffer_mlp = layers.concatenate([total_mlp, done_mlp], axis=-1)
    # bandwidth_in = tf.expand_dims(bandwidth, axis=-1)
    # bandwidth_in = tf.tile(bandwidth_in, [1, 2, 1])
    # # concatenate on dim[1] batch*new*2
    # op_output = layers.concatenate([buffer_mlp, bandwidth_in], axis=-1)

    # op_dist = layers.Dense(input_dim_list[2][1], activation='softmax')(op_output)
    # 执行缓冲区 执行操作
    total_mlp = tf.transpose(total_buffer, perm=[0, 2,
                                                 1])  # perm:控制转置的操作,以perm = [0,1,2] 3个维度的数组为例, 0–代表的是最外层的一维, 1–代表外向内数第二维, 2–代表最内层的一维,这种perm是默认的值.如果换成[1,0,2],就是把最外层的两维进行转置，比如原来是2乘3乘4，经过[1,0,2]的转置维度将会变成3乘2乘4
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.transpose(total_mlp, perm=[0, 2, 1])
    exe_op = layers.Dense(input_dim_list[1][1], activation='softmax')(total_mlp)  # 执行
    # 卸载缓冲区 卸载操作
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.transpose(done_mlp, perm=[0, 2, 1])
    bandwidth_in = tf.expand_dims(bandwidth, axis=-1)  # 张量拼接
    bandwidth_in = layers.Dense(1, activation='relu')(bandwidth_in)
    done_mlp = layers.concatenate([done_mlp, bandwidth_in], axis=-1)
    off_op = layers.Dense(input_dim_list[2][1], activation='softmax')(done_mlp)  # 卸载

    op_dist = layers.concatenate([exe_op, off_op], axis=1)
    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, bandwidth], outputs=[move_out, op_dist])
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_aa))
    return model


"""构建 center_actor_net"""


# inputs: sensor_map, agent_map, bandwidth_vector;
# outputs: bandwidth_vec

def center_actor(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(
        shape=input_dim_list[0])  # 缓冲区列表形状 shape = (None, 4, 2, 5)       # 实例化一个keras张量,shape: 形状元组（整型）
    pos_list = keras.Input(shape=input_dim_list[1])  # 位置列表形状  shape = (None, 4, 2)

    # buffer        #TODO layers.Dense()()
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)  # dense ：全连接层  相当于添加一个层, inputs：输入该网络层的数据
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果,axis可以用来指定要删掉的为1的维度

    # pos list
    pos = layers.Dense(2, activation='relu')(pos_list)

    bandwidth_out = layers.concatenate([buffer_state, pos],
                                       axis=-1)  # shape = (None, 4, 4)            # axis=n表示从第n个维度进行拼接，对于一个三维矩阵，axis的取值可以为[-3, -2, -1, 0, 1, 2]
    # bandwidth_out = layers.AlphaDropout(0.2)(bandwidth_out)
    bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)  # shape = (None, 4, 1)
    bandwidth_out = tf.squeeze(bandwidth_out, axis=-1)  # shape = (None, 4)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    bandwidth_out = layers.Softmax()(bandwidth_out)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    # bandwidth_out = bandwidth_out / tf.reduce_sum(bandwidth_out, 1, keepdims=True)
    # bandwidth_out = bandwidth_out / tf.expand_dims(tf.reduce_sum(bandwidth_out, 1), axis=-1)

    model = keras.Model(inputs=[done_buffer_list, pos_list], outputs=bandwidth_out, name='center_actor_net')
    # 以下代码原为注释
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])
    #
    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='softmax')(sensor_cnn)
    #
    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='softmax')(agent_cnn)
    #
    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn], axis=-1)
    # bandwidth_out = layers.Dense(input_dim_list[2], activation='softmax')(bandwidth_out)
    #
    # model = keras.Model(inputs=[sensor_map, agent_map], outputs=bandwidth_out, name='center_actor_net')
    # # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    # 以上代码原为注释
    return model


"""构建 agent_critic_net"""


def agent_critic(input_dim_list, cnn_kernel_size):  # input_dim_list[(121, 121, 2), (2, 5), (2, 5), (13, 13), (2, 5), 1]
    state_map = keras.Input(shape=input_dim_list[0])  # (121, 121, 2)
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])  # (2, 5)
    done_buffer = keras.Input(shape=input_dim_list[2])  # (2, 5)
    move = keras.Input(shape=input_dim_list[3])  # (13, 13)
    onehot_op = keras.Input(shape=input_dim_list[4])  # (2, 5)
    bandwidth = keras.Input(shape=input_dim_list[5])

    # map CNN
    # merge last dim
    map_cnn = layers.Dense(1, activation='relu')(state_map)
    map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)
    map_cnn = layers.AlphaDropout(0.2)(map_cnn)
    # map_cnn = layers.Conv2D(input_dim_list[0][2], kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    # map_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(map_cnn)
    # map_cnn = layers.Dropout(0.2)(map_cnn)
    map_cnn = layers.Flatten()(map_cnn)  # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    map_cnn = layers.Dense(2, activation='relu')(map_cnn)

    # mlp
    # pos_mlp = layers.Dense(1, activation='relu')(position)
    band_mlp = layers.Dense(1, activation='relu')(bandwidth)
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.squeeze(total_mlp, axis=-1)
    total_mlp = layers.Dense(2, activation='relu')(total_mlp)
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.squeeze(done_mlp, axis=-1)
    done_mlp = layers.Dense(2, activation='relu')(done_mlp)

    move_mlp = layers.Flatten()(move)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)
    onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)
    onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)  # 从张量形状中移除大小为1的维度.指定 axis 来删除特定的大小为1的维度.

    all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)

    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)
    # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model


"""构建 center_critic_net"""


def center_critic(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])
    pos_list = keras.Input(shape=input_dim_list[1])
    bandwidth_vec = keras.Input(shape=input_dim_list[2])

    # buffer        # TODO 为什么最后要变成 shape = [None, 4]
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)  # shape = [None, 4, 2, 1]
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # shape = [None, 4, 2]
    buffer_state = layers.Dense(1, activation='relu')(buffer_state)  # shape = [None, 4, 1]
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # shape = [None, 4]

    # pos list  shape = [None, 4, 2]
    pos = layers.Dense(1, activation='relu')(pos_list)  # shape = [None, 4, 1]
    pos = tf.squeeze(pos, axis=-1)  # shape = [None, 4]

    # bandvec
    # band_in = layers.Dense(2, activation='relu')(bandwidth_vec)

    r_out = layers.concatenate(
        [buffer_state, pos, bandwidth_vec])  # r_out shape = [None, 12]，参数三者的shape相同，shape = [None, 4]
    # r_out = layers.AlphaDropout(0.2)(r_out)
    r_out = layers.Dense(1, activation='relu')(r_out)  # r_out shape = [None, 1]
    model = keras.Model(inputs=[done_buffer_list, pos_list, bandwidth_vec], outputs=r_out, name='center_critic_net')
    # 以下代码原为注释
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])
    # bandwidth_vec = keras.Input(shape=input_dim_list[2])
    #
    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='relu')(sensor_cnn)
    #
    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='relu')(agent_cnn)
    #
    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn, bandwidth_vec], axis=-1)
    # bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)
    #
    # model = keras.Model(inputs=[sensor_map, agent_map, bandwidth_vec], outputs=bandwidth_out, name='center_critic_net')
    # # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    # 以上代码原为注释
    return model


"""更新 target_net 的权重 weights """


def update_target_net(model, target, tau=0.8):
    weights = model.get_weights()  # 仅仅是获取权重，不保存
    target_weights = target.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * (1 - tau) + target_weights[i] * tau  # tau # soft replacement  目标更新权重
    target.set_weights(target_weights)  # 给模型设置权重


"""联邦学习 根据其他agent的参数更新自己参数的过程"""


def merge_fl(nets, omega=0.5):
    for agent_no in range(len(nets)):
        target_params = nets[agent_no].get_weights()
        other_params = []
        for i, net in enumerate(nets):
            if i == agent_no:
                continue
            other_params.append(net.get_weights())
        for i in range(len(target_params)):
            others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)
            target_params[i] = omega * target_params[i] + others * (1 - omega)
            # print([others.shape, target_params[i].shape])
        nets[agent_no].set_weights(target_params)


"""选定移动位置"""


def circle_argmax(move_dist, move_r):
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))
    # print(tf.squeeze(move_dist, axis=-1))
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)
    # print(max_pos)
    return max_pos[np.argmin(pos_dist)]


class MAACAgent(object):

    def __init__(self, env, tau, gamma, lr_aa, lr_ac, lr_ca, lr_cc, batch,
                 epsilon=0.2):  # aa agent actor; ac agent critic; ca center actor; cc center critic
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.index_dim = 2  # 缓冲区索引（执行缓冲区和完成缓冲区）
        self.obs_r = self.env.obs_r  # 观察半径
        self.state_map_shape = (self.obs_r * 2 + 1, self.obs_r * 2 + 1, self.index_dim)  # 状态map的形状
        self.pos_shape = (2)  # 位置 形状
        self.band_shape = (1)  # 带宽 形状
        self.buffstate_shape = (self.index_dim, self.env.max_buffer_size)  # 缓冲区(执行和完成)状态 形状
        # self.sensor_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        # self.agent_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        self.buffer_list_shape = (self.agent_num, self.index_dim, self.env.max_buffer_size)  # 缓冲区列表形状
        self.pos_list_shape = (self.agent_num, 2)  # 位置列表形状
        self.bandvec_shape = (self.env.agent_num)  # 带宽形状
        self.op_shape = (self.index_dim, self.env.max_buffer_size)  # 操作缓冲区形状 (2, 5)
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.move_r)  # 可移动到的坐标点及其个数
        self.movemap_shape = (self.env.move_r * 2 + 1, self.env.move_r * 2 + 1)  # 移动map的形状
        self.epsilon = epsilon  # Probability of random exploration

        # learning params
        self.tau = tau  # soft replacement  目标更新权重。
        self.cnn_kernel_size = 3
        self.gamma = gamma  # reward discount
        self.lr_aa = lr_aa  # learning rate for agent actor
        self.lr_ac = lr_ac  # learning rate for agent critic
        self.lr_ca = lr_ca  # learning rate for center actor
        self.lr_cc = lr_cc  # learning rate for center critic
        self.batch_size = batch
        self.agent_memory = {}
        self.softmax_memory = {}
        self.center_memory = []
        self.sample_prop = 1 / 4

        # 初始化网络 net init
        # 模型网络
        self.agent_actors = []  # 所有agent的actor网络
        self.center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape],
                                         self.cnn_kernel_size)
        self.agent_critics = []  # 所有agent的critic网络
        self.center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape],
                                           self.cnn_kernel_size)
        # 目标网络
        self.target_agent_actors = []
        self.target_center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape],
                                                self.cnn_kernel_size)
        update_target_net(self.center_actor, self.target_center_actor, tau=0)  # 最初tau=0
        self.target_agent_critics = []
        self.target_center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape],
                                                  self.cnn_kernel_size)
        update_target_net(self.center_critic, self.target_center_critic, tau=0)
        # TODO opt 是干什么的
        self.agent_actor_opt = []
        self.agent_critic_opt = []  # agent_critic 的操作，更新梯度
        self.center_actor_opt = keras.optimizers.Adam(
            learning_rate=lr_ca)  # learn rate center actor       # 优化器keras.optimizers.Adam()是解决这个问题的一个方案。其大概的思想是开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得。
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)  # learn rate center critic
        # TODO
        self.summaries = {}

        for i in range(self.env.agent_num):
            self.agent_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_ac))
            self.agent_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_aa))
            # agent_actor_net 和 target_agent_actor_net
            new_agent_actor = agent_actor(
                [self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape],
                self.cnn_kernel_size, self.env.move_r)
            target_agent_actor = agent_actor(
                [self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape],
                self.cnn_kernel_size, self.env.move_r)
            # new_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # target_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            update_target_net(new_agent_actor, target_agent_actor, tau=0)

            self.agent_actors.append(new_agent_actor)
            self.target_agent_actors.append(target_agent_actor)
            # agent_critic_net 和 target_agent_critic_net
            # new_agent_critic = agent_critic([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape,
            #                                  self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            # t_agent_critic = agent_critic([self.state_map_shape, self.pos_shape,
            # self.buffstate_shape, self.buffstate_shape, self.movemap_shape,
            # self.op_shape, self.band_shape], self.cnn_kernel_size)
            new_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape,
                                             self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            target_agent_critic = agent_critic(
                [self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.movemap_shape, self.op_shape,
                 self.band_shape], self.cnn_kernel_size)
            update_target_net(new_agent_critic, target_agent_critic, tau=0)
            self.agent_critics.append(new_agent_critic)
            self.target_agent_critics.append(target_agent_critic)

        # 打印模型结果图
        keras.utils.plot_model(self.center_actor, 'logs/model_figs/new_center_actor.png', show_shapes=True)
        keras.utils.plot_model(self.center_critic, 'logs/model_figs/new_center_critic.png', show_shapes=True)
        keras.utils.plot_model(self.agent_actors[0], 'logs/model_figs/new_agent_actor.png', show_shapes=True)
        keras.utils.plot_model(self.agent_critics[0], 'logs/model_figs/new_agent_critic.png', show_shapes=True)

    """actor执行动作"""

    def actor_act(self, epoch):
        tmp = random.random()
        if tmp >= self.epsilon and epoch >= 16:  # todo epoch >= 16 经验池已充满？
            # 边缘agent的动作    agent act
            agent_act_list = []  # 所有agent 动作 列表（包含移动，执行，卸载）
            softmax_list = []  # softmax 列表（包含移动 和 对缓冲区的操作）
            cur_state_list = []
            band_vec = np.zeros(self.agent_num)  # 带宽比例
            for i, agent in enumerate(self.agents):
                # actor = self.agent_actors[i]
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)  # shape = (1, 121, 121, 2)
                # pos = tf.expand_dims(agent.position, axis=0)
                # print('agent{}pos:{}'.format(i, pos))
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)  # 增加一个维度 shape = (1, 2, 5)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)  # shape = (1, 2, 5)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)  # shape = (1,)
                # print('band{}'.format(agent.action.bandwidth))
                band_vec[i] = agent.action.bandwidth  # 带宽比例
                assemble_state = [state_map, total_data_state, done_data_state, band]  # 总状态
                # print(['agent%s' % i, sum(sum(state_map))])
                cur_state_list.append(assemble_state)
                # print(total_data_state.shape)
                # agent_actor 预测，根据预测进行操作（移动位置、执行、卸载、）
                action_output = self.agent_actors[i].predict(assemble_state)
                move_dist = action_output[0][0]  # 移动
                sio.savemat('debug.mat', {'state': self.env.get_obs(agent), 'move': move_dist})
                # print(move_dist)
                # print(move_dist.shape)
                op_dist = action_output[1][0]  # 缓冲区操作
                # print(op_dist.shape)
                # move_ori = np.unravel_index(np.argmax(move_dist), move_dist.shape)
                move_ori = circle_argmax(move_dist, self.env.move_r)
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]  # 移动到的位置
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(op_dist[0])] = 1  # 选定执行的任务 # np.argmax 返回一个numpy数组中最大值的索引值
                offloading[np.argmax(op_dist[1])] = 1  # 选定卸载的任务
                move_softmax = np.zeros(move_dist.shape)
                op_softmax = np.zeros(self.buffstate_shape)

                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(op_dist[0])] = 1
                op_softmax[1][np.argmax(op_dist[1])] = 1

                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                op_softmax = tf.expand_dims(op_softmax, axis=0)

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])
            # print(agent_act_list)
            # 中心agent动作     center act
            done_buffer_list, pos_list = self.env.get_center_state()  # 获取边缘agent位置以及卸载缓冲区数据大小和年龄
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)  # TODO 这个有什么用
            new_bandvec = self.center_actor.predict([done_buffer_list, pos_list])
            # print('new_bandwidth{}'.format(new_bandvec[0]))
            # 经过预测后得到的结果
            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec[0])
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)

            # 经验池 record memory
            # edge agent 的经验池
            for i, agent in enumerate(self.agents):
                state_map = new_state_maps[i]  # 观察范围
                # print(['agent%s' % i, sum(sum(state_map))])
                # pos = agent.position
                total_data_state = agent.get_total_data()
                done_data_state = agent.get_done_data()  # 获取完成数据 shape = （2， 5）
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)  # 观察范围
                # pos = tf.expand_dims(agent.position, axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)  # 执行缓冲区全部数据 shape = （1， 2， 5）
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)  # 完成缓冲区数据 shape = （1， 2， 5）
                band = tf.expand_dims(agent.action.bandwidth, axis=0)  # 带宽
                new_states = [state_map, total_data_state, done_data_state, band]  # 新状态
                # agent 的缓冲区添加新状态
                if agent.no in self.agent_memory.keys():
                    self.agent_memory[agent.no].append(
                        [cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [
                        [cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]
            # 中心agent经验池
            self.center_memory.append(
                [[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])

        else:
            # random action  随机执行动作
            # agents
            agent_act_list = []
            for i, agent in enumerate(self.agents):
                move = random.sample(list(self.move_dict.values()), 1)[
                    0]  # move 下一个移动位置的坐标(x, y) # random.sample() 截取列表的指定长度的随机数，但是不会改变列表本身的排序
                execution = [0] * agent.max_buffer_size  # 执行缓冲区
                offloading = [0] * agent.max_buffer_size  # 卸载缓冲区
                execution[np.random.randint(agent.max_buffer_size)] = 1  # 随机选一个执行
                offloading[np.random.randint(agent.max_buffer_size)] = 1  # 随机选一个卸载
                agent_act_list.append([move, execution, offloading])
            # center
            new_bandvec = np.random.rand(self.agent_num)  # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。
            new_bandvec = new_bandvec / np.sum(new_bandvec)
            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec)

        return new_rewards[-1]  # TODO 如果四个agent不是联合的reward， 为什么只返回最后一个？

    """经验重放过程"""

    # @tf.function(experimental_relax_shapes=True)
    def replay(self):
        # agent 经验回放    agent replay
        for no, agent_memory in self.agent_memory.items():
            if len(agent_memory) < self.batch_size:
                continue
            # print([len(agent_memory[-100:]), self.batch_size])
            samples = agent_memory[-int(self.batch_size * self.sample_prop):] + random.sample(
                # todo 这里截取的有问题，应该是agent_memory[-self.batch_size * 2:-int(self.batch_size * self.sample_prop)]
                agent_memory[-self.batch_size * 2:],
                int(self.batch_size * (1 - self.sample_prop)))  # random.sample 截取列表的指定长度的随机数,但是不会改变列表本身的排序
            # t_agent_actor = self.target_agent_actors[no]
            # t_agent_critic = self.target_agent_critics[no]
            # agent_actor = self.agent_actors[no]
            # agent_critic = self.agent_critics[no]
            # 获取sample中的信息
            # state
            state_map = np.vstack([sample[0][0] for sample in samples])
            # pos = np.vstack([sample[0][1] for sample in samples])
            total_data_state = np.vstack([sample[0][1] for sample in samples])
            done_data_state = np.vstack([sample[0][2] for sample in samples])
            band = np.vstack([sample[0][3] for sample in samples])
            # action
            move = np.vstack([sample[1][0] for sample in samples])
            op_softmax = np.vstack([sample[1][1] for sample in samples])
            # reward
            a_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1)
            # new states
            new_state_map = np.vstack([sample[3][0] for sample in samples])
            # new_pos = np.vstack([sample[3][1] for sample in samples])
            new_total_data_state = np.vstack([sample[3][1] for sample in samples])
            new_done_data_state = np.vstack([sample[3][2] for sample in samples])
            new_band = np.vstack([sample[3][3] for sample in samples])
            # # done
            # done = [sample[4] for sample in samples]

            # 下一步的action 和 reward   next actions & rewards
            new_actions = self.target_agent_actors[no].predict(
                [new_state_map, new_total_data_state, new_done_data_state, new_band])
            # new_move = np.array([self.move_dict[np.argmax(single_sample)] for single_sample in new_actions[0]])
            # print(new_actions[1].shape)
            q_future = self.target_agent_critics[no].predict(
                [new_state_map, new_total_data_state, new_done_data_state, new_actions[0], new_actions[1], new_band])
            # print('qfuture{}'.format(q_future))
            target_qs = a_reward + q_future * self.gamma

            # 训练策略网络 train critic
            with tf.GradientTape() as tape:  # 根据某个函数的输入变量来计算它的导数,Tensorflow 会把 ‘tf.GradientTape’ 上下文中执行的所有操作都记录在一个磁带上 (“tape”)。 然后基于这个磁带和每次操作产生的导数，用反向微分法（“reverse mode differentiation”）来计算这些被“记录在案”的函数的导数。
                # tape.watch(self.agent_critics[no].trainable_variables)
                q_values = self.agent_critics[no](
                    [state_map, total_data_state, done_data_state, move, op_softmax, band])
                ac_error = q_values - tf.cast(target_qs, dtype=tf.float32)
                # ac_error = q_values - target_qs
                ac_loss = tf.reduce_mean(tf.math.square(ac_error))  # agent_critic_loss
            # print('agent%s' % no)
            # print([q_values, target_qs, ac_error, ac_loss])
            ac_grad = tape.gradient(ac_loss, self.agent_critics[no].trainable_variables)
            # print(ac_grad)
            self.agent_critic_opt[no].apply_gradients(
                zip(ac_grad, self.agent_critics[no].trainable_variables))  # zip()的目的是映射多个容器的相似索引，以便可以将它们用作单个实体使用

            # 训练动作网络 train actor
            with tf.GradientTape() as tape:
                tape.watch(self.agent_actors[no].trainable_variables)  # 确保某个tensor被tape追踪
                actions = self.agent_actors[no]([state_map, total_data_state, done_data_state, band])
                # actor_move = np.array([self.move_dict[np.argmax(single_sample)] for single_sample in actions[0]])
                new_r = self.agent_critics[no](
                    [state_map, total_data_state, done_data_state, actions[0], actions[1], band])
                # print(new_r)
                aa_loss = tf.reduce_mean(new_r)  # agent actor loss
                # print(aa_loss)
            aa_grad = tape.gradient(aa_loss, self.agent_actors[no].trainable_variables)
            # print(aa_grad)
            self.agent_actor_opt[no].apply_gradients(zip(aa_grad, self.agent_actors[no].trainable_variables))

            # summary info
            self.summaries['agent%s-critic_loss' % no] = ac_loss
            self.summaries['agent%s-actor_loss' % no] = aa_loss

        # 中心经验回放    agent replay
        if len(self.center_memory) < self.batch_size:
            return
        # todo 这里截取的有问题，应该是center_memory[-self.batch_size * 2:-int(self.batch_size * self.sample_prop)]
        center_samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(
            self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
        done_buffer_list = np.vstack([sample[0][0] for sample in center_samples])
        pos_list = np.vstack([sample[0][1] for sample in center_samples])
        bandvec_act = np.vstack([sample[1] for sample in center_samples])
        c_reward = tf.expand_dims([sample[2] for sample in center_samples], axis=-1)
        # new states
        new_done_buffer_list = np.vstack([sample[3][0] for sample in center_samples])
        new_pos_list = np.vstack([sample[3][1] for sample in center_samples])
        # next actions & reward
        new_c_actions = self.target_center_actor.predict([new_done_buffer_list, new_pos_list])
        cq_future = self.target_center_critic.predict([new_done_buffer_list, new_pos_list, new_c_actions])
        c_target_qs = c_reward + cq_future * self.gamma
        self.summaries['cq_val'] = np.average(c_reward[0])

        # 训练中心策略网络 train center critic
        with tf.GradientTape() as tape:
            tape.watch(self.center_critic.trainable_variables)
            cq_values = self.center_critic([done_buffer_list, pos_list, bandvec_act])
            cc_loss = tf.reduce_mean(tf.math.square(cq_values - tf.cast(c_target_qs, dtype=tf.float32)))
            # cc_loss = tf.reduce_mean(tf.math.square(cq_values - c_target_qs))
        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))
        # 训练中心动作网络 train center actor
        with tf.GradientTape() as tape:
            tape.watch(self.center_actor.trainable_variables)
            c_act = self.center_actor([done_buffer_list, pos_list])
            ca_loss = tf.reduce_mean(self.center_critic([done_buffer_list, pos_list, c_act]))
        # print(self.center_critic([sensor_maps, agent_maps, c_act]))
        ca_grad = tape.gradient(ca_loss, self.center_actor.trainable_variables)
        # print(ca_grad)
        self.center_actor_opt.apply_gradients(zip(ca_grad, self.center_actor.trainable_variables))
        # print(ca_loss)
        self.summaries['center-critic_loss'] = cc_loss
        self.summaries['center-actor_loss'] = ca_loss

    # TODO:运行报错
    #  WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. 已编译加载的模型，但尚未构建已编译的度量。
    #  `model.compile_metrics` will be empty until you train or evaluate the model.
    def save_model(self, episode, time_str):
        for i in range(self.agent_num):
            self.agent_actors[i].save('logs/models/{}/agent-actor-{}_episode{}.h5'.format(time_str, i, episode))
            self.agent_critics[i].save('logs/models/{}/agent-critic-{}_episode{}.h5'.format(time_str, i, episode))
        self.center_actor.save('logs/models/{}/center-actor_episode{}.h5'.format(time_str, episode))
        self.center_critic.save('logs/models/{}/center-critic_episode{}.h5'.format(time_str, episode))

    """训练"""

    # @tf.function
    def train(self, max_epochs=2000, max_step=500, up_freq=8, render=False, render_freq=1, FL=False, FL_omega=0.5,
              anomaly_edge=False):
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'logs/fit/' + cur_time
        env_log_dir = 'logs/env/env' + cur_time
        record_dir = 'logs/records/' + cur_time
        os.mkdir(env_log_dir)
        os.mkdir(record_dir)
        summary_writer = tf.summary.create_file_writer(train_log_dir)  # 为给定的日志目录创建摘要文件编写器
        # tf.summary.trace_on(graph=True, profiler=True)
        os.makedirs('logs/models/' + cur_time)
        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0  # 一个episode有固定max_step个step，epochs是所有step的总计数，epochs = episode * step    理论上：一个epoch中存在多个eposide。所有的训练数据都要跑一遍算一个epoch
        finish_length = []  # 完成的数
        finish_size = []  # 完成的量
        sensor_ages = []  # 数据源的年龄
        # sensor_map = self.env.DS_map
        # sensor_pos_list = self.env.world.sensor_pos
        # sensor_states = [self.env.DS_state]
        # agent_pos = [[[agent.position[0], agent.position[1]] for agent in self.agents]]
        # agent_off = [[agent.action.offloading for agent in self.agents]]
        # agent_exe = [[agent.action.execution for agent in self.agents]]
        # agent_band = [[agent.action.bandwidth for agent in self.agents]]
        # agent_trans = [[agent.trans_rate for agent in self.agents]]
        # buff, pos = self.env.get_center_state()
        # agent_donebuff = [buff]
        # exe, done = self.env.get_buffer_state()
        # exebuff = [exe]
        # donebuff = [done]

        anomaly_step = 6000
        anomaly_agent = self.agent_num - 1

        # if anomaly_edge:
        #     anomaly_step = np.random.randint(int(max_epochs * 0.5), int(max_epochs * 0.75))
        #     anomaly_agent = np.random.randint(self.agent_num)
        # summary_record = []

        while epoch < max_epochs:
            print('epoch %s' % epoch)
            # if anomaly_edge and (epoch == anomaly_step):
            #     self.agents[anomaly_agent].movable = False

            # 每20个epoch保存一次环境map
            if render and (epoch % 20 == 1):
                self.env.render(env_log_dir, epoch, True)
                # sensor_states.append(self.env.DS_state)

            # 经过max_step后， 结束一个episode，更新经验池，重新开始
            if steps >= max_step:
                # self.env.world.finished_data = []
                episode += 1
                # self.env.reset()
                for m in self.agent_memory.keys():
                    del self.agent_memory[m][0:-self.batch_size * 2]  # self.batch_size = 128
                del self.center_memory[0:-self.batch_size * 2]
                print(
                    'episode {}:  total reward: {}, steps: {}, epochs: {}'.format(episode, total_reward / steps, steps,
                                                                                  epoch))  # TODO reward?

                with summary_writer.as_default():  # Summary： 所有需要在TensorBoard上展示的统计结果
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)  # 添加标量统计结果
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=train_log_dir)

                summary_writer.flush()
                self.save_model(episode, cur_time)
                steps = 0
                total_reward = 0

            cur_reward = self.actor_act(epoch)  # 获取当前reward
            # print('episode-%s reward:%f' % (episode, cur_reward))
            self.replay()  # 经验重放
            finish_length.append(len(self.env.world.finished_data))  # 完成 数
            finish_size.append(sum([data[0] for data in self.env.world.finished_data]))  # 完成 量
            sensor_ages.append(list(self.env.world.sensor_age.values()))
            # agent_pos.append([[agent.position[0], agent.position[1]] for agent in self.env.world.agents])
            # # print(agent_pos)
            # agent_off.append([agent.action.offloading for agent in self.agents])
            # agent_exe.append([agent.action.execution for agent in self.agents])
            # # agent_band.append([agent.action.bandwidth for agent in self.agents])
            # agent_trans.append([agent.trans_rate for agent in self.agents])
            # buff, pos = self.env.get_center_state()
            # # agent_donebuff.append(buff)
            # exe, done = self.env.get_buffer_state()
            # exebuff.append(exe)
            # donebuff.append(done)

            # summary_record.append(self.summaries)
            # 联合学习参数更新 以及 目标网络权重参数更新 update target
            if epoch % up_freq == 1:
                print('update targets, finished data: {}'.format(len(self.env.world.finished_data)))

                # finish_length.append(len(self.env.world.finished_data))
                if FL:  # 联合学习更新网络参数
                    merge_fl(self.agent_actors, FL_omega)
                    merge_fl(self.agent_critics, FL_omega)
                    # merge_fl(self.target_agent_actors, FL_omega)
                    # merge_fl(self.target_agent_critics, FL_omega)
                for i in range(self.agent_num):
                    update_target_net(self.agent_actors[i], self.target_agent_actors[i], self.tau)
                    update_target_net(self.agent_critics[i], self.target_agent_critics[i], self.tau)
                update_target_net(self.center_actor, self.target_center_actor, self.tau)
                update_target_net(self.center_critic, self.target_center_critic, self.tau)

            total_reward += cur_reward
            steps += 1
            epoch += 1

            # tensorboard 喂入需要监听的数据
            with summary_writer.as_default():
                if len(self.center_memory) > self.batch_size:
                    tf.summary.scalar('Loss/center_actor_loss', self.summaries['center-actor_loss'],
                                      step=epoch)  # 用来显示标量信息
                    tf.summary.scalar('Loss/center_critic_loss', self.summaries['center-critic_loss'], step=epoch)
                    tf.summary.scalar('Loss/agent_actor_loss', self.summaries['agent0-actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/agent_critic_loss', self.summaries['agent0-critic_loss'], step=epoch)
                    tf.summary.scalar('Stats/cq_val', self.summaries['cq_val'], step=epoch)
                    for acount in range(self.agent_num):
                        tf.summary.scalar('Stats/agent%s_actor_loss' % acount,
                                          self.summaries['agent%s-actor_loss' % acount], step=epoch)
                        tf.summary.scalar('Stats/agent%s_critic_loss' % acount,
                                          self.summaries['agent%s-critic_loss' % acount], step=epoch)
                tf.summary.scalar('Main/step_average_age', cur_reward, step=epoch)

            summary_writer.flush()

        # save final model
        self.save_model(episode, cur_time)
        sio.savemat(record_dir + '/data.mat',
                    {'finish_len': finish_length,
                     'finish_data': finish_size,
                     'ages': sensor_ages})
        # sio.savemat(record_dir + '/data.mat',
        #             {'finish_len': finish_length,
        #              'finish_data': finish_size,
        #              'sensor_map': sensor_map,
        #              'sensor_list': sensor_pos_list,
        #              'sensor_state': sensor_states,
        #              'agentpos': agent_pos,
        #              'agentoff': agent_off,
        #              'agentexe': agent_exe,
        #              'agenttran': agent_trans,
        #              'agentbuff': agent_donebuff,
        #              'agentexebuff': exebuff,
        #              'agentdonebuff': donebuff,
        #              'agentband': agent_band,
        #              'anomaly': [anomaly_step,
        #                          anomaly_agent]})
        # with open(record_dir + '/record.json', 'w') as f:
        #     json.dump(summary_record, f)

        # 画出环境map gif
        self.env.render(env_log_dir, epoch, True)
        img_paths = glob.glob(env_log_dir + '/*.png')
        # linux(/)和windows(\)文件路径斜杠不同，注意区分
        system = platform.system()  # 获取操作系统类型
        if system == 'Windows':
            img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))
        elif system == 'Linux':
            img_paths.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))

        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))
        imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=15)
