'''
搭建一个多层，带L2正则项的神经网络;
计算每一个banch数据对应的正则项
'''

import tensorflow as tf

def get_weights(shape, lambda):
    # 产生一个特定维度的随机变量
    var = tf.Variable(tf.random_normal(shape), type=tf.float32)
    # 将var变量对应的L2正则化损失项加入集合中
    tf.add_to_collection(
        'lose', tf.contrib.layer.l2_regularizer(lambda)(var))
    return var


# 定义输入样本集
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 2))

banch_size = 8
# 神经网络结构
layer_dimension = [2, 10, 10, 10, 1]
# NN层数
n_layers = len(layer_dimension)

# 初始网络层
cur_input = x
in_dimension = layer_dimension[0]

# 循环产生神经网络
for i in range(1: n_layers):
    out_dimension = layer_dimension[i]
    # 定义i层网络的参数
    weight = get_weights([in_dimension,out_dimension], 0.01)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 激活函数ReLU
    cur_input = tf.nn.relu(tf.matmul(cur_input, weight) + bias)
    # 更新输入层数
    in_dimension = layer_dimension[i]

# 损失函数
mse_loss = tf.reduce_mean(tf.square(y_- cur_input))

# 把损失函数计算值放入集合中
tf.add_to_collection(
    'mse_loss', mse_loss
)

# 计算所有参数的正则化值
reg = tf.add_n(tf.get_collection('lose'))

