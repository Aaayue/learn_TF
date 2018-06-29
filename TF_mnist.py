'''
单层网络实现mnist实例
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist数据集相关参数（使用大写表示程序中的参数常量）

IUPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络相关参数
LAYER1_NODE = 500
BANCH_SIZE = 100

LEARNING_RATE_ORIGIN = 0.8
LEARNING_RATE_DECAY = 0.99
LAMBDA = 0.1
TRAINING_STEP = 10000
MOVING_AVG_DECAY = 0.99

def inference(input_tensor, avg_class, weight1, bias1, weight2, bias2):
    # 不使用滑动平均模型改善参数变量，直接使用ReLU激活函数
    if ave_class == None:
        layer1_out = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)

        return tf.matmul(layer1_out, weight2) + bias2

    # 使用滑动平均模型优化参数变量
    else:
        layer1_out = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1))
                                + avg_class.average(bias1))

        return tf.matmul(layer1_out, avg_class.average(weight2)) + avg_class.average(bias2)


# 训练模型的过程
def train(mnist):
    x = tf.palceholder(tf.float32, shape=[None, IUPUT_NODE], name='x-input')
    y_ = tf.palceholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')

    # 定义变量初始值
    weight1 = tf.Variable(tf.truncated_normal([IUPUT_NODE, LAYER1_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播输出结果
    y = inference(x, None, weight1, bias1, weight2, bias2)

    # 定义一个滑动平均模型
    global_step = tf.Variable(0, trainable=False)
    variable_avg = tf.train.ExponentialMovingAverage(
        MOVING_AVG_DECAY, global_step
    )

    # 将滑动平均模型应用到可训练的变量中
    variable_avg_op = variable_avg.apply(tf.trainable_variables())

    # 计算使用了滑动平均后的前向传播输出结果
    avg_y_ = inference(x, variable_avg_op, weight1, bias1, weight2, bias2)

    # 交叉熵定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 计算当前banch中所有样例的交叉熵平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy, 0)

    # 计算l2正则化损失
    regulizer = tf.contrib.layers.l2_regularizer(LAMBDA)
    regulize_loss = regulizer(weight1) + regulizer(weight2)
    total_loss = cross_entropy_mean + regulize_loss

    #定义学习率衰减
    learning_rate = tf.train.exponential_dacay(
        LEARNING_RATE_ORIGIN,
        global_step,
        mnist.trian.num_examples/BANCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )


    # 优化过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_avg_op]):
        train_op = tf.no_op(name='train')

    # 判断预测值与真实值是否相等
    correct_predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.Session() as sess:
        tf.global_variable_initializer().run()

        #CV set
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        #Test set
        test_feed = {x, mnist.test.images, y_:mnist.test.labels}

        # 迭代训练
        for i in range(TRAINING_STEP):

            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, validate_feed)
                print('After %d training steps, validation accuracy using average model is %g' %(i, validate_acc))

            xs, ys = mnist.train.next_banch(BANCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_:ys})

        test_acc = sess.run(accuracy, test_feed)
        print('After %d training steps, test accuracy using average model is %g' %(i, test_acc))



def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train()

if __name__ == '__main__':
    tf.app.run()


