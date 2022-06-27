#from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")
data = pd.read_csv('d:/Downloads/TADPOLE_D1_D2.csv')
dataframe_subset = data[['Ventricles', 'Hippocampus', 'WholeBrain','Fusiform', 'Entorhinal', 'MidTemp','RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting' ,'FDG','AGE','PTEDUCAT','APOE4','PTGENDER','MMSE']]
data_subset=dataframe_subset.replace(np.nan,0)
data_subset= pd.get_dummies(data_subset, columns = ['PTGENDER'])
x=data_subset.drop(['MMSE'],axis=1).values
y=data_subset['MMSE'].values
arr = data_subset.to_numpy()
data_subset = np.array(data_subset)
data_subset = pd.DataFrame(StandardScaler().fit_transform(data_subset))
y = y.astype('float')
x= x.astype('float')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
m1 = DecisionTreeRegressor(random_state = 0)
m1.fit(x_train, y_train)
with open('modelg.pkl','wb') as file:
        pickle.dump(m1, file)













#data = pd.read_csv('d:/Downloads/TADPOLE_D1_D2.csv')
#data= pd.get_dummies(data, columns = ['PTGENDER'])
#print(dataframe_subset)
#data1 = data[['Ventricles', 'Hippocampus', 'WholeBrain', 'Fusiform', 'Entorhinal', 'MidTemp', 'RAVLT_immediate','RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FDG', 'AGE', 'PTEDUCAT', 'APOE4','PTGENDER_Female','PTGENDER_Male','MMSE']]
#data1=data1.replace(np.nan, 0)
#data1 = data1.reindex(labels=data.columns,axis=1)
#data1 = np.array(data1)
#data1 = pd.DataFrame(StandardScaler().fit_transform(data1))
#x = data1.iloc[:,:-1].values
#y = data1.iloc[:,-1].values
#x = data1[1:, 0]
#y = data1[1:, -1]
#x = data1.drop(['MMSE'], axis=1).values
#y = data1['MMSE'].values
#x = data1.iloc[:, 1].values
#y = data1.iloc[:, 2].values
#y = y.astype('float')
#x= x.astype('float')
#x=x.reshape(-1,1)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
#m1 = DecisionTreeRegressor(random_state = 0)

#x_train.reshape(-1, 1)
#m1.fit(x_train, y_train)
#with open('model6.pkl','wb') as file:
 #       pickle.dump(m1, file)
# pickle.dump(m1, open("../model4.pkl", "wb"))
# model = pickle.load(open("../mod4.pkl", "rb"))