import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
learning_rate = 0.001
# 定义权值
w1 = tf.Variable(tf.random_normal([2,3], mean=0, stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 设置batch数据
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
# 交叉熵
cross_entropy = -tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)
# 采用Adam优化算法
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

rdm = RandomState(1)
data_size = 256
X = rdm.rand(data_size, 2)

# 给输出设置标签
Y = [[int((x1+x2) < 1)] for (x1,x2) in X]

# 创建一个绘画用于训练
with tf.Session() as sess:
    # 所有变量初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run(init))

    train_iter = 5000 # 训练轮数
    for i in range(train_iter):
        START = (i*batch_size) % data_size # batch的训练是一个循环利用数据的过程
        END = min(START+batch_size, data_size)
        sess.run(train_step, feed_dict = {x: X[START:END], y_: Y[START:END]})
        if i % 500 == 0:
            # 每迭代500次，打印出优化结果
            total_cross_entropy = sess.run(cross_entropy, feed_dict = {x: X, y_:Y})
            print('cross entropy on {} steps is {}.'.format(i, total_cross_entropy))
    sess.run(w1)
    sess.run(w2)
    print('W1:\n')
    print(sess.run(w1))
    print('W2:\n')
    print(sess.run(w2))







