# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train=pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2022/train.csv')
test=pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2022/test.csv')

id=test['ID']
x_train=train.drop(columns=['ID','Segmentation'])
x_test=test.drop(columns=['ID'])
y_train=train['Segmentation']

# print(x_train.isnull().sum()) # 결측치 없음
# print(x_test.isnull().sum())


from sklearn.preprocessing import LabelEncoder
col=['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
encoder=LabelEncoder()
for i in col:
    x_train[i]=encoder.fit_transform(x_train[i])
    x_test[i]=encoder.transform(x_test[i])
    x_train[i]=x_train[i].astype('category')
    x_test[i]=x_test[i].astype('category')

x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

from sklearn.preprocessing import StandardScaler
val=[['Age','Work_Experience','Family_Size']]
scaler=StandardScaler()
for i in val:
    x_train[i]=scaler.fit_transform(x_train[i])
    x_test[i]=scaler.transform(x_test[i])
    
from sklearn.model_selection import train_test_split
X_train,X_valid,Y_train,Y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=10)

from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier()
model1.fit(X_train,Y_train)
pred1=model1.predict(X_valid)

# from xgboost import XGBClassifier # ValueError: Invalid classes inferred from unique values of `y`. Expected: [0 1 2 3], got [1 2 3 4]
# model2=XGBClassifier()
# model2.fit(X_train,Y_train)
# pred2=model2.predict(X_valid)

# # 하이퍼파라미터 튜닝
# from sklearn.model_selection import GridSearchCV
# parameters={'n_estimators':[50,100],'max_depth':[4,6]}
# model3=RandomForestClassifier()
# clf=GridSearchCV(estimator=model3, param_grid=parameters, cv=3)
# clf.fit(X_train,Y_train)
# # print('최적의 파라미터: ',clf.best_params_) # {'max_depth': 6,'n_estimators': 50}

model4=RandomForestClassifier(max_depth=6, n_estimators=50)
model4.fit(X_train,Y_train)
pred4=model4.predict(X_valid)

# from sklearn.metrics import f1_score
# print('RF1',f1_score(Y_valid,pred1,average='macro')) # 0.4697
# print('RF4',f1_score(Y_valid,pred4,average='macro')) # 0.5093

result=model4.predict(x_test)
pd.DataFrame({'ID':id,'Segmentation':result}).to_csv('submission.csv',index=False)