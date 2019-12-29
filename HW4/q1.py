import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
xTrain = pd.read_csv("xTrain.csv")

def model_assessment(xTrain,yTrain):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """
    xTrain, xValidation, yTrain, yValidation = train_test_split(xTrain, yTrain, test_size=0.3)
    xTrain.to_csv("xTrain.csv", index=False)
    xValidation.to_csv("xValidation.csv", index=False)
    yTrain.to_csv("yTrain.csv", index=False)
    yValidation.to_csv("yValidation.csv", index=False)
    xTrain = pd.read_csv("xTrain.csv")
    xValidation = pd.read_csv("xValidation.csv")
    yTrain = pd.read_csv("yTrain.csv")
    yValidation = pd.read_csv("yValidation.csv")
    return xTrain,xValidation,yTrain,yValidation


def build_vocab_map(df):
    vocab_map = term_fre(df)
    vocab_map = dict((k, v) for k, v in vocab_map.items() if v >= 30)
    sorted_map = sorted(vocab_map.items(), key=lambda x: x[1])
    return vocab_map



def construct_binary(df):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    vocab_map = build_vocab_map(xTrain)
    vocab_list = list(vocab_map.keys())
    binary = []
    for i in range(len(df)):
        row = df.iloc[i][0]
        words = row.split()
        words = list(dict.fromkeys(words))
        vector = []
        for word in vocab_list:
            if word in words:
                vector.append(1)
            else:
                vector.append(0)
        binary.append(vector)
    binary = pd.DataFrame(binary)
    return binary


def construct_count(df):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    vocab_map = build_vocab_map(xTrain)
    vocab_list = list(vocab_map.keys())
    count = []
    for i in range(len(df)):
        row = df.iloc[i][0]
        words = row.split()
        # words = list(dict.fromkeys(words))
        vector = [0] * len(vocab_list)
        for word in words:
            if word in vocab_list:
                index = vocab_list.index(word)
                vector[index] = vector[index]+1
        count.append(vector)
    count = pd.DataFrame(count)
    return count



def construct_tfidf(df):
    """
    Construct the email datasets based on
    the TF-IDF representation of the email.
    """
    vocab_map = build_vocab_map(xTrain)
    vocab_list = list(vocab_map.keys())
    count = []
    for i in range(len(df)):
        row = df.iloc[i][0]
        words = row.split()
        #print(type(words))
        #words = list(dict.fromkeys(words))
        vector = [0] * len(vocab_list)
        for word in words:
            if word in vocab_list:
                index = vocab_list.index(word)
                vector[index] = vector[index] + 1
        count.append(vector)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(count)
    return pd.DataFrame(tfidf.toarray())

def term_fre(df):
    counts = dict()
    for i in range(len(df)):
        row = df.iloc[i][0]
        words = row.split()
        words = list(dict.fromkeys(words))
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()

    spam = pd.read_csv("spamAssassin.data")
    spam = pd.DataFrame(spam)
    X = []
    Y = []
    for i in range(len(spam)):
        line = (spam.iloc[i][0])
        x, y = spam.iloc[i][0][1:], spam.iloc[i][0][:1]
        X.append(x)
        Y.append(y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3)
    xTrain.to_csv("xTrain.csv", index=False)
    xTest.to_csv("xTest.csv", index=False)
    yTrain.to_csv("yTrain.csv", index=False)
    yTest.to_csv("yTest.csv", index=False)
    xTrain = pd.read_csv("xTrain.csv")
    xTest = pd.read_csv("xTest.csv")
    yTrain = pd.read_csv("yTrain.csv")
    yTest = pd.read_csv("yTest.csv")
    # changed parameter
    xTrain, xValidation, yTrain, yValidation = model_assessment(xTrain, yTrain)
    vocab_map = build_vocab_map(xTrain)
    binary = construct_binary(xTrain)
    count = construct_count(xTrain)
    tfidf = construct_tfidf(xTrain)
    binary.to_csv("xTrainbinary.csv", index=False)
    count.to_csv("xTraincount.csv", index=False)
    tfidf.to_csv("xTraintfidf.csv", index=False)
    xValidationbinary = construct_binary(xValidation)
    xValidationcount = construct_count(xValidation)
    xValidationtfidf = construct_tfidf(xValidation)
    xValidationbinary.to_csv("xValidationbinary.csv", index=False)
    xValidationcount.to_csv("xValidationcount.csv", index=False)
    xValidationtfidf.to_csv("xValidationtfidf.csv", index=False)
    xTestbinary = construct_binary(xTest)
    xTestcount = construct_count(xTest)
    xTesttfidf = construct_tfidf(xTest)
    xTestbinary.to_csv("xTestbinary.csv", index=False)
    xTestcount.to_csv("xTestcount.csv", index=False)
    xTesttfidf.to_csv("xTesttfidf.csv", index=False)


if __name__ == "__main__":
    main()








