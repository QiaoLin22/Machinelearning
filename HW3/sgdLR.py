import argparse
import numpy as np
import pandas as pd
import time
import sklearn
from lr import LinearRegression, file_to_numpy
from sklearn import linear_model
import random
from random import shuffle


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        #beta = np.linalg.inv(xTrain.transpose().dot(xTrain)).dot(xTrain.transpose()).dot(yTrain)
        LinearRegression.beta = [[0],[0]]
        intercept = 0
        n = len(xTrain)
        startall = time.time()
        list = [x for x in range(0,len(xTrain))]
        for i in range(self.mEpoch):
            random.shuffle(list)
            Dbetalist = []
            Dinterceptlist = []
            start = time.time()
            for j in range(self.bs):
                # Dbetalist = []
                # Dinterceptlist = []
                indexstart = len(list)/self.bs * j
                indexend = len(list)/self.bs * j + len(list)/self.bs
                subset = xTrain[int(indexstart):int(indexend)]
                #print(subset)
                ypredict = np.dot(subset,LinearRegression.beta) + intercept
                subyTrain = yTrain[int(indexstart):int(indexend)]
                Dbeta = (-2/n)*sum(np.dot(subset.transpose(),(subyTrain-ypredict)))
                Dbetalist.append(Dbeta)
                Dintercept = (-2/n)*sum(subyTrain-ypredict)
                Dinterceptlist.append(Dintercept)

            end = time.time()
            timeElapse = end - start

            avgDbeta = sum(Dbetalist) / len(Dbetalist)
            avgDintercept = sum(Dinterceptlist) / len(Dinterceptlist)
            #print(avgDbeta)
            # print(avgDintercept)
            LinearRegression.beta = LinearRegression.beta - self.lr * avgDbeta
            #print(LinearRegression.beta)
            # print(beta)
            intercept = intercept - self.lr * avgDintercept

            value = {}
            value['time'] = timeElapse
            value['train-mse'] = LinearRegression.mse(LinearRegression, xTrain, yTrain)
            value['test-mse'] = LinearRegression.mse(LinearRegression, xTest, yTest)
            trainStats[i*self.bs] = value

            endall = time.time()
            totaltime = endall - startall


        return trainStats



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

