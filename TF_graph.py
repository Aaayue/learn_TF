import tensorflow as tf

# 三层神经网络，输入->隐藏->输出
# 节点权值w1,w2
w1 = tf.Variable(tf.random_normal((2,3),mean=0,stddev=2,seed=1))
w2 = tf.Variable(tf.random_normal((3,1),mean=0,stddev=2,seed=1))

# 输入值
x = tf.constant([0.7,0.9],shape=[1,2])

# 数据传递
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 结果运行
sess = tf.Session()
# 权值初始化
sess.run(w1.initializer)
sess.run(w2.initializer)
'''
初始化所有变量：
init_op = tf.global_variable_initializer()
sess.run(init_op)
这种初始化方式会自动处理变量间的依赖关系
'''
print(sess.run(y))
sess.close()
