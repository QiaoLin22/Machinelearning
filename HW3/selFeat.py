import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import time
from pydoc import help
#from scipy.stats.stats import pearsonr
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

xTrain = pd.read_csv("eng_xTrain.csv")
yTrain = pd.read_csv("eng_yTrain.csv")
xTest = pd.read_csv("eng_xTest.csv")
yTest = pd.read_csv("eng_yTest.csv")

def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    newcolumn = []
    for i in range(len(df)):
        #newcolumn = (df[df.columns[0]])

        ts = df[df.columns[0]][i]
        date, time = ts.split(' ')
        hour, minute = time.split(':')
        featime = int(hour) * 60 + int(minute)
        newcolumn.append(featime)


    df = df.drop(columns=['date'])
    #df = df.append()
    df.insert(0, 'newtime', newcolumn)
    return df



def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO
    #dfnew["label"] = yTrain

    Var_Corr = df.corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(xTrain.corr())
    plt.show()
    labelsort = Var_Corr.iloc[26]
    labelsort2 = sorted(labelsort,key=abs,reverse=True)
    print(labelsort2)
    df2 = pd.DataFrame()
    df2["newtime"] = df["newtime"]
    df2["lights"] = df["lights"]

    return df2


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    stdScale = StandardScaler()
    stdScale.fit(testDF)
    trainDF = stdScale.transform(trainDF)
    trainDF = pd.DataFrame(trainDF)
    testDF = stdScale.transform(testDF)
    testDF = pd.DataFrame(testDF)
    return trainDF, testDF

# xTrain = extract_features(xTrain)
# xTrain["label"] = yTrain
# print(xTrain)
# #pearson = pearsonr(xTrain[xTrain.columns[0]], xTrain[xTrain.columns[-1]])
# #print(pearson)
# Var_Corr = xTrain.corr()
# print(Var_Corr)
# plt.figure(figsize=(8,8))
# sns.heatmap(xTrain.corr())
# plt.show()
#xTrain2 = extract_features(xTrain)
#xTrain3 = select_features(xTrain2)
#print(xTrain3)
#xTrain4 = preprocess_data(xTrain3,xTest)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
