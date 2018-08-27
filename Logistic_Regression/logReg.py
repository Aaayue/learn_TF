import numpy as np
from math import *


class LogRegression(object):

    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def model_optimise(self, weight, bias, inx, iny, alpha):
        """
        input:
        weights = mat[nx1]
        bias = constant
        inx = mat[mxn]
        iny = mat[mx1]
        output:
        dw:
        """
        m, n = inx.shape
        y_ = self.sigmoid(np.matmul(inx, weight)+bias)
        temp = np.tile((y_-iny), n)
        # print(tmp.shape)
        dw = np.sum(np.multiply(temp, inx), axis=0)/m
        # print(dw.shape)
        da = np.sum(y_-iny)/m
        new_weight = weight - alpha*dw.T
        new_bias = bias - alpha*da
        return new_weight, new_bias

    def classVec(self, input_y):
        input_y[input_y >= 0.5] = 1
        input_y[input_y < 0.5] = 0
        return input_y

    def log_reg(self, train_data, train_labels, test_data, test_labels, iters, alpha=0.1):
        """
        function to do logistic regression
        :param train_data: sample data = mat[mxn]
        :param train_labels: sample train_labels = mat[mx1]
        :param test_data: sample data = mat[mxn]
        :param test_labels: sample train_labels = mat[mx1]
        :param iters: iterations
        :param alpha: learning rate
        :return:
        """
        # init weights and bias
        train_data = np.mat(train_data)
        test_data = np.mat(test_data)
        train_labels = np.mat(train_labels).T
        test_labels = np.mat(test_labels).T
        m, n = train_data.shape
        m_test = test_data.shape[0]
        init_weight = np.random.randn(n+1, 1)
        weight = init_weight[1:]
        bias = init_weight[0]
        # print(weight.shape, bias.shape)
        tmp_res = np.matmul(train_data, weight)+bias
        # print(tmp_res.shape)
        res = self.sigmoid(tmp_res)
        # print(res.shape, train_labels.shape)
        cost_func = -(np.sum(np.multiply(train_labels, np.log(res)) +
                             np.multiply((1 - train_labels), np.log(1 - res))))/m
        print('initial weight+bias shape:', init_weight.shape)
        print('initial cost function:', cost_func)
        min_cost = np.inf
        opt_weight = weight
        opt_bias = bias
        for i in range(iters+1):
            # np.random.shuffle(rand_array)
            weight, bias = self.model_optimise(weight, bias, train_data, train_labels, alpha)
            # print(weight.shape, bias.shape)
            tmp_res = np.matmul(train_data, weight) + bias
            res = self.sigmoid(tmp_res)
            train_cost = -np.sum(np.multiply(train_labels, np.log(res)) +
                                 np.multiply((1 - train_labels), np.log(1 - res)))/m

            res_vec = self.classVec(res)
            train_err = len(np.where(res_vec != train_labels)[0])
            train_err_rate = train_err/float(m)
            test_cost, test_res = self.test(test_data, test_labels, weight, bias)
            if min_cost >= test_cost:
                min_cost = test_cost
                opt_weight = weight
                opt_bias = bias
            test_err = len(np.where(test_labels != test_res)[0])
            test_err_rate = test_err/float(m_test)
            if i % 20 == 0:
                print('STEP:', i)
                print('TRAIN COST:', train_cost)
                print('TEST COST:', test_cost)
                # print('bias + weight:', bias, weight.T)
                print('train error: {}, {}%'.format(train_err, train_err_rate*100))
                print('test error: {}, {}%'.format(test_err, test_err_rate*100))
        return opt_weight, opt_bias

    def test(self, test_data, test_label, weight, bias):
        tmp_res = np.matmul(test_data, weight) + bias
        res = self.sigmoid(tmp_res)
        cost_func = -np.mean(np.sum(np.multiply(test_label, np.log(res)) +
                                    np.multiply((1 - test_label), np.log(1 - res))))
        res_vec = self.classVec(res)
        return cost_func, res_vec


if __name__ == "__main__":
    path = '/home/zy/data_pool/U-TMP/excersize/0401_0630_17_1_CoSoOt_L_REG_TEST_17.npz'
    tmp = np.load(path)
    data_samp = tmp['features']
    label = tmp['labels']
    label[label == 6] = 1
    # label[label == 0] = 0
    data_len = len(label)
    train_data = data_samp[:int(data_len*0.6)]
    train_label = label[:int(data_len*0.6)]
    test_data = data_samp[int(data_len*0.6):]
    test_label = label[int(data_len*0.6):]
    LR = LogRegression()
    LR.log_reg(train_data, train_label, test_data, test_label, 2000, 0.3)

