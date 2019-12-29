from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#(a)
dataset = load_iris()
df = pd.DataFrame(dataset['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
df['Species'] = dataset['target']
#(b)

sns.boxplot(x="Species",y="Sepal Length",data = df)
plt.show()

sns.boxplot(x="Species",y="Sepal Width",data = df)
plt.show()

sns.boxplot(x="Species",y="Petal length",data = df)
plt.show()

sns.boxplot(x="Species",y="Petal Width",data = df)
plt.show()

#(c)
sns.FacetGrid(df,hue="Species").map(plt.scatter,"Petal length","Petal Width").add_legend()
plt.show()
sns.FacetGrid(df,hue="Species").map(plt.scatter,"Sepal Length","Sepal Width").add_legend()
plt.show()

#(d)
'''
From the scatter plot of Sepal Length vs Spepal Width, it is obvious that  Virginica > versicolor
> setosa. For petal size setosa tend to have a smaller length but larger width, while the difference
between versicolor and virginica is not too obvious. According to the distribution of boxplot of the four features for each species, a set of rule I come up to 
classify the species is:

(1)if sepal length is around 1.5, sepal width is around 0.25, petal length is around 5, petal width
is around 3.5, then this iris can be categorized as setosa.
(2)if sepal length is around 4.5, sepal width is around 1.25, petal length is around 6, petal width
is around 2.8, then this iris can be categorized as versicolor.
(3)if sepal length is around 5.5, sepal width is around 2, petal length is around 6.5, petal width
is around 3.0, then this iris can be categorized as virginica.

'''








