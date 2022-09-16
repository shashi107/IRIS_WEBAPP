import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
import pickle as plk

iris = datasets.load_iris()
X = iris.data[:, :4]  # we take four features.
y = iris.target
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.30)
 
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
score_gbc = gbc.score(x_test,y_test)
print(score_gbc)

with open('iris.pkl','wb') as file:
    plk.dump(gbc,file)