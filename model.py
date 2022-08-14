import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
import pickle

heart_data=pd.read_csv('heart_statlog_cleveland_hungary_final2.csv')

heart_data['cholesterol']=heart_data['cholesterol'].replace(0,heart_data['cholesterol'].mean())

X=heart_data.drop(columns='target', axis=1)
Y=heart_data['target']

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=3)

classifier_R=RandomForestClassifier(n_estimators= 40, criterion="gini", max_depth=None)
classifier_R.fit(X_train,Y_train)  

pickle.dump(classifier_R,open('model.pkl','wb'))
