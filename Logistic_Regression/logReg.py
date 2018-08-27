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
        m, n = inx.shape()
        y_ = self.sigmoid(np.matmul(inx, weight.T)+bias)
        tmp = np.tile((y_-iny), n, axis=1)
        dw = np.sum(np.multiply(tmp, inx), axis=0)/m
        da = np.sum(y_-iny)/m
        new_weight = weight - alpha*dw
        new_bias = bias - alpha*da
        return new_weight, new_bias


    def log_reg(self, data_arr, labels, iters, alpha):
        """
        function to do logistic regression
        :param data_arr: sample data = mat[mxn], the first column is all one
        :param labels: sample labels = mat[mx1]
        :param iters: iterations
        :return:
        """
        # init weights and bias
        m, n = data_arr.shape
        init_weight = np.random.randn(n+1, 1)
        weight = init_weight[1:]
        bias = init_weight[0]
        tmp_res = np.multiply(data_arr, weight)+bias
        res = self.sigmoid(tmp_res)
        cost_func = -np.mean(np.sum(labels*np.log(res)+(1-labels)*log(1-res)))
        print('initial weight:', init_weight)
        print('initial cost function:', cost_func)
        min_cost = np.inf
        for i in range(iters):
            weight, bias = model_optimise(self, weight, bias, inx, iny, alpha)
            tmp_res = np.multiply(data_arr, weight) + bias
            res = self.sigmoid(tmp_res)
            cost_func = -np.mean(np.sum(labels * np.log(res) + (1 - labels) * log(1 - res)))
            if min_cost >= cost_func:
                min_cost = cost_func
            if i % 10 == 0:
                print('STEP:', i)
                print('COST:', cost_func)
                print('weight + bias:', weight, bias)


		
	

