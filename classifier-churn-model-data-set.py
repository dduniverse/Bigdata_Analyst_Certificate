# %% [code] {"execution":{"iopub.status.busy":"2022-11-29T12:13:15.105663Z","iopub.execute_input":"2022-11-29T12:13:15.106269Z","iopub.status.idle":"2022-11-29T12:13:15.155566Z","shell.execute_reply.started":"2022-11-29T12:13:15.106175Z","shell.execute_reply":"2022-11-29T12:13:15.154796Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-11-29T12:14:35.243746Z","iopub.execute_input":"2022-11-29T12:14:35.244147Z","iopub.status.idle":"2022-11-29T12:14:35.309109Z","shell.execute_reply.started":"2022-11-29T12:14:35.244116Z","shell.execute_reply":"2022-11-29T12:14:35.307710Z"},"jupyter":{"outputs_hidden":false}}


# 데이터 불러오기
import pandas as pd

X_test=pd.read_csv('/kaggle/input/churn-model-data-set-competition-form/X_test.csv')
X_train=pd.read_csv('/kaggle/input/churn-model-data-set-competition-form/X_train.csv')
y_train=pd.read_csv('/kaggle/input/churn-model-data-set-competition-form/y_train.csv')

# 분석에 필요없는 컬럼 삭제
CustomerId=X_test['CustomerId']
X_train.drop(columns=['CustomerId','Surname'],inplace=True)
X_test.drop(columns=['CustomerId','Surname'],inplace=True)
y_train.drop(columns='CustomerId',inplace=True)

# 데이터 형식 변환
X_train['IsActiveMember']=X_train['IsActiveMember'].astype('object')
X_test['IsActiveMember']=X_test['IsActiveMember'].astype('object')
X_train['HasCrCard']=X_train['HasCrCard'].astype('object')
X_test['HasCrCard']=X_test['HasCrCard'].astype('object')

# 더미 변환
X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)

# 분석에 용이하게 'Age' 컬럼의 십의 자리만 남김
X_train['Age']=X_train['Age']//10
X_test['Age']=X_test['Age']//10

# 수치형 변수 스케일링
from sklearn.preprocessing import MinMaxScaler
value=[['CreditScore','Age','Tenure','Balance','NumOfProducts']]
scaler=MinMaxScaler()
for i in value:
    X_train[i]=scaler.fit_transform(X_train[i])
    X_test[i]=scaler.transform(X_test[i])

# 검증 데이터 분리
from sklearn.model_selection import train_test_split
X_TRAIN, X_VALID, Y_TRAIN, Y_VALID=train_test_split(X_train,y_train,test_size=0.2,random_state=10)

# 모델 생성1 (RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier()
model1.fit(X_TRAIN,Y_TRAIN)
pred1=model1.predict(X_VALID)

# 모델 생성2 (LogisticRegression)
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_TRAIN,Y_TRAIN)
pred2=model2.predict(X_VALID)

# 모델 생성3 (XGBClassifier)
import xgboost as xgb
model3=xgb.XGBClassifier()
model3.fit(X_TRAIN, Y_TRAIN)
pred3=model3.predict(X_VALID)

# # 성능 확인 (accuracy_score)
# from sklearn.metrics import accuracy_score
# print('RF',accuracy_score(Y_VALID,pred1)) # 0.8569
# print('LR',accuracy_score(Y_VALID,pred2)) # 0.8046
# print('XGB',accuracy_score(Y_VALID,pred3)) # 0.8592

# 테스트 데이터 예측 및 결과 제출
result=model3.predict(X_test)
pd.DataFrame({'CustomerId':CustomerId,'Exited':result}).to_csv('12345.csv',index=False)

# 예측 정확도 확인
# y_test=pd.read_csv('/kaggle/input/churn-model-data-set-competition-form/test_label/y_test.csv')
# pred=pd.read_csv('12345.csv')
# print(accuracy_score(y_test.iloc[:,1],pred.iloc[:,1])) # 0.8454