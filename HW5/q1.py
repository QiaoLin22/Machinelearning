import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import decomposition, datasets, model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
xTrain = pd.read_csv("q1xTrain.csv")
xTest = pd.read_csv("q1xTest.csv")
yTrain = pd.read_csv("q1yTrain.csv")
yTest = pd.read_csv("q1yTest.csv")
#(a)
transformer = Normalizer().fit(xTrain)
Normalizer(copy=True, norm='l2')
xTrain = (transformer.transform(xTrain))
yTrain = (transformer.transform(yTrain))
xTest = (transformer.transform(xTest))
yTest = (transformer.transform(yTest))
yTrain = np.ravel(yTrain)
yTest = np.ravel(yTest)
clf = LogisticRegression(penalty='none',random_state=0, solver='lbfgs',max_iter= 1000,multi_class='multinomial').fit(xTrain, yTrain)
clf.predict(xTest)
clf.predict_proba(xTest)
#(b)
xTrain = pd.read_csv("q1xTrain.csv")
xTest = pd.read_csv("q1xTest.csv")
yTrain = pd.read_csv("q1yTrain.csv")
yTest = pd.read_csv("q1yTest.csv")
yTrain = np.ravel(yTrain)
yTest = np.ravel(yTest)
pca = PCA(n_components=2)
pca.fit(xTrain)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
variancelist = pca.explained_variance_ratio_
(pca.fit_transform(xTrain))
totalvariance = 0
for i in variancelist:
    totalvariance += i
    print(totalvariance)
model = ExtraTreesClassifier(n_estimators=10)
model.fit(xTrain, yTrain)
#print(model.feature_importances_)
#(c)
nmf = decomposition.NMF(n_components=10).fit(xTrain)
def score(model, data, score=metrics.explained_variance_score):
    prediction = model.inverse_transform(model.transform(data))
    return score(data, prediction)
#print(score(nmf, xTrain))
#(d)
xTrainnormal = (transformer.transform(xTrain))
xTrainpca = pca.fit_transform(xTrain)
model = NMF(n_components=10, init='random', random_state=0)
xTrainnmf = model.fit_transform(xTrain)
xTestnormal = (transformer.transform(xTest))
xTestpca = pca.fit_transform(xTest)
xTestnmf = model.fit_transform(xTest)
clfnormal = LogisticRegression(penalty='none',random_state=0, solver='lbfgs',max_iter= 1000,multi_class='multinomial').fit(xTrainnormal, yTrain)
clf.predict(xTestnormal)
ypredict1 = clf.predict_proba(xTestnormal)
clfpca = LogisticRegression(penalty='none',random_state=0, solver='lbfgs',max_iter= 1000,multi_class='multinomial').fit(xTrainpca, yTrain)
clfpca.predict(xTestpca)
ypredict2 = clfpca.predict_proba(xTestpca)
clfnmf = LogisticRegression(penalty='none',random_state=0, solver='lbfgs',max_iter= 1000,multi_class='multinomial').fit(xTrainnmf, yTrain)
clfnmf.predict(xTestnmf)
clfnmf.predict_proba(xTestnmf)
ypredict3 = clfnmf.predict_proba(xTestnmf)
yprob1 = (ypredict1[:,[1]])
yprob2 = ypredict2[:,[1]]
yprob3 = ypredict3[:,[1]]
fpr, tpr, _ = metrics.roc_curve(yTest,  yprob1)
fpr2, tpr2, _ = metrics.roc_curve(yTest,  yprob2)
fpr3, tpr3, _ = metrics.roc_curve(yTest,  yprob3)
roc1 = metrics.roc_auc_score(yTest, yprob1)
roc2 = metrics.roc_auc_score(yTest, yprob2)
roc3 = metrics.roc_auc_score(yTest, yprob3)
plt.plot(fpr,tpr,label="normal, roc="+str(roc1))
plt.plot(fpr2,tpr2,label="PCA, roc="+str(roc2))
plt.plot(fpr3,tpr3,label="NMF, roc="+str(roc3))
plt.legend(loc=4)
plt.show()





