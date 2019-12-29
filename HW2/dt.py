import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import random
import statistics

class DecisionTree(object):
    maxDepth = 1  # maximum depth of the decision tree
    minLeafSample = 1 # minimum number of samples in a leaf
    criterion = 'entropy'   # splitting criterion
    xFeat = pd.read_csv("q4xTrain.csv")
    y = pd.read_csv("q4yTrain.csv")['label']
    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    class Decision_Node:
        def __init__(self,
                     bestsplit,
                     splitthreshold,
                     false_branch,
                     true_branch):
            self.bestsplit = bestsplit
            self.splitthreshold = splitthreshold
            self.false_branch = false_branch
            self.true_branch = true_branch


    def build_tree(self,data,currentdepth, minLeafSample,y):

        bestsplit, splitthreshold, maxinfogain = self.splitwhichfeature(data,y)
        #print(bestsplit)
        #print(splitthreshold)
        if currentdepth >= self.maxDepth:
            return self.to_leaf(data)

        left_rows, right_rows = self.feature_split(bestsplit,splitthreshold,data)
        left_rows = pd.DataFrame(left_rows)
        right_rows = pd.DataFrame(right_rows)
#        print("bestsplit" + str(bestsplit))
#        print("splitthreshold" + str(splitthreshold))
#        print("maxinfogain" + str(maxinfogain))
#        print("currentdepth" + str(currentdepth))
        if len(left_rows) > minLeafSample:
#            print("len left_rows")
#            print(len(left_rows))
            false_branch = self.build_tree(left_rows,currentdepth+1,minLeafSample,y)
        else:
            return self.to_leaf(left_rows)
        #print("currentrightrowlen "+str(len(left_rows)))
        #print("currentdepth "+ str(currentdepth))
        if len(right_rows) > minLeafSample:
#            print("len right_rows")
#            print(len(right_rows))
            true_branch = self.build_tree(right_rows,currentdepth+1,minLeafSample,y)
        else:
            return self.to_leaf(right_rows)

        return self.Decision_Node(bestsplit, splitthreshold,false_branch, true_branch)

    def print_tree(node, spacing=""):
        if isinstance(node,int):
            print(spacing + "Predict", node)
            return

        # Print the question at this node
        print(spacing + str(node.bestsplit) +">"+ str(node.splitthreshold))

        # Call this function recursively on the true branch
        print(spacing + '--> False:')
        DecisionTree.print_tree(node.false_branch, spacing + "  ")


        # Call this function recursively on the false branch
        print(spacing + '--> True:')
        DecisionTree.print_tree(node.true_branch, spacing + "  ")


    def to_leaf(self,data):
        count0 = 0
        count1 = 0
        for i in range(len(data)):
            if data.iloc[i][len(data.columns)-1] == 0:
                count0 += 1
            else:
                count1 += 1

        if count0 > count1:
            return 0
        elif count1 > count0:
            return 1
        elif count0 == count1:
            return 0
            #return random.choice([0,1])
        #result = [row[-1] for row in data]
        #return max(set(result), key=result.count)



    def feature_split(self,featureindex, threshold, dataframe):
        left = []
        right = []
        for i in range(len(dataframe)):
            if dataframe.iloc[i][featureindex] < threshold:
                left.append(dataframe.iloc[i])
            else:
                right.append(dataframe.iloc[i])
        #left = pd.DataFrame(left)
        #right = pd.DataFrame(right)
        return left,right


    def splitwhichfeature(self,data,y):
        infogain = []
        bestsplit = 0
        splitthreshold = 0
        maxinfogain = 0
        for i in range(len(data.columns)-1):
            infogain.append(self.info_gain(data[data.columns[i]],y))
        order = sorted(infogain,reverse=True)
        bestsplit=(infogain.index(order[0]))
        splitthreshold=(infogain[infogain.index(order[0])][1])
        maxinfogain=(infogain[infogain.index(order[0])][0])
        #print(order)
        #print("allfeaturemaxinfogain" + str(maxinfogain))
        #print("allfeaturesplitthreshold" + str(splitthreshold))
        return bestsplit,splitthreshold,maxinfogain

    def info_gain(self,feature,y):
        threshold = []
        entropylist = []
        infogain = 0
        maxinfogain = 0
        maxthreshold = 0
        threshold1 = statistics.mean(feature)
        sd = np.std(feature)
        threshold.append(threshold1 - 0.2 * sd)
        threshold.append(threshold1 - 0.1*sd)
        threshold.append(threshold1)
        threshold.append(threshold1 + 0.1*sd)
        threshold.append(threshold1 + 0.2 * sd)
        #threshold = feature.unique().tolist()

        for j in range(len(threshold)):
            number1 = 0
            number2 = 0
            success1 = 0
            success2 = 0
            for i in range(len(feature)):
                if feature.iloc[i] >= threshold[j]:
                    number1 +=1
                    if y.iloc[i] == 1:
                        success1 += 1
                else:
                    number2 +=1
                    if y.iloc[i] == 1:
                        success2 += 1
            if number1 == 0:
                continue
            if number2 == 0:
                continue
            if success1 == 0:
                continue
            if success2 == 0:
                continue

            if success1 == number1:
                entropy = (number1/len(feature)) * (-1 * (math.log(1, 2)))+ ((number2/len(feature)) * (-1 * ((success2/number2)*math.log(success2/number2, 2) +(1-(success2/number2))*math.log(1-(success2/number2), 2))))
                infogain = 1 - entropy
            elif success2 == number2:
                entropy = (number1 / len(feature)) * (-1 * ((success1 / number1) * math.log(success1 / number1, 2) + (
                            1 - (success1 / number1)) * math.log(1 - (success1 / number1), 2)))
                + ((number2 / len(feature)) * (-1 * (math.log(1, 2))))
                infogain = 1 - entropy
            else:
                entropy = (number1/len(feature)) * (-1 * ((success1/number1)*math.log(success1/number1, 2) +(1-(success1/number1))*math.log(1-(success1/number1), 2)))
                + (number2/len(feature)) * (-1 * ((success2/number2)*math.log(success2/number2, 2) +(1-(success2/number2))*math.log(1-(success2/number2), 2)))
                infogain = 1 - entropy
            if infogain > maxinfogain:
                maxinfogain = infogain
                maxthreshold = threshold[j]
        #return maxinfogain,maxthreshold
        return [maxinfogain,maxthreshold]



    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        xFeat = pd.concat([xFeat, y], axis=1)
        return self.build_tree(xFeat,0,self.minLeafSample,y)


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        mytree = self.train(xFeat=self.xFeat, y=self.y)
        for i in range(len(xFeat)):
            predict = self.classify(xFeat.iloc[i], mytree)
            yHat.append(predict)
        return yHat

    def classify(self,row, node):
        if isinstance(node, int):

            return node

        if row[node.bestsplit] > node.splitthreshold:
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    #dt1 = DecisionTree('gini', args.md, args.mls)
    #trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    #print("GINI Criterion ---------------")
    #print("Training Acc:", trainAcc1)
    #print("Test Acc:", testAcc1)

    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    #DecisionTree.print_tree(dt.train(xTrain,yTrain['label']))



if __name__ == "__main__":
    main()
'''
q4xTrain = pd.read_csv("q4xTrain.csv")
q4yTrain = pd.read_csv("q4yTrain.csv")
q4xTest = pd.read_csv("q4xTest.csv")
q4yTest = pd.read_csv("q4yTest.csv")
q4xTrain = pd.concat([q4xTrain, q4yTrain], axis=1)

#featureindex = DecisionTree.splitwhichfeature(DecisionTree,q4xTrain,q4yTrain)[0]
#print(featureindex)
#splitthreshold = DecisionTree.splitwhichfeature(DecisionTree,q4xTrain,q4yTrain)[1]
#print(splitthreshold)
#left_rows, right_rows = (DecisionTree.feature_split(featureindex,splitthreshold,q4xTrain))
#print(left_rows)

#mytree = DecisionTree.train(DecisionTree,xFeat=q4xTrain,y=q4yTrain)
#mytree = DecisionTree.train(DecisionTree,xFeat=dataset,y=datasety)
#DecisionTree.print_tree(mytree)

#predict = DecisionTree.classify(DecisionTree,q4xTrain.iloc[0], mytree)
#print(predict)
'''









