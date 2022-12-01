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


# 데이터 불러오기
test=pd.read_csv('/kaggle/input/big-data-analytics-certification/t2-1-test.csv')
train=pd.read_csv('/kaggle/input/big-data-analytics-certification/t2-1-train.csv')

x_train=train.drop(columns='TravelInsurance')
x_test=test
y_train=train['TravelInsurance']

tt=pd.concat([x_train,x_test],axis=0)

# 불필요한 컬럼 제거
id=x_test['id']
x_train.drop(columns='id',inplace=True)
x_test.drop(columns='id',inplace=True)

# 결측치 처리
# print(x_train.isnull().sum())
x_train['AnnualIncome']=x_train['AnnualIncome'].fillna(x_train['AnnualIncome'].mean())
x_test['AnnualIncome']=x_test['AnnualIncome'].fillna(x_test['AnnualIncome'].mean())

# # 'Age'의 십의 자리만 남김 -> 그대로 남겨두는 것이 성능이 높음
# x_train['Age']=x_train['Age']//10
# x_test['Age']=x_test['Age']//10

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
col=['Employment Type','GraduateOrNot','FrequentFlyer','EverTravelledAbroad']
encoder=LabelEncoder()
for i in col:
    encoder.fit(tt[i]) # train 에서 fit_transform을 해주면 test에서 transform 시 unseen labels 에러 발생 -> 전체 train 데이터에 대한 fit 진행 후 각각 transform
    x_train[i]=encoder.transform(x_train[i])
    x_test[i]=encoder.transform(x_test[i]) 
    x_train[i]=x_train[i].astype('category')
    x_test[i]=x_test[i].astype('category')

x_dummies = pd.get_dummies(pd.concat([x_train, x_test])) # 마찬가지로 x_train,x_test 각각 더미변수 생성 시 x_trian(12),x_test(13)의 feature 개수가 달라 모델학습시 에러 발생
x_train = x_dummies[:x_train.shape[0]]
x_test = x_dummies[x_train.shape[0]:]

# 스케일링
from sklearn.preprocessing import StandardScaler
val=[['Age','FamilyMembers','ChronicDiseases']]
scaler=StandardScaler()
for i in val:
    x_train[i]=scaler.fit_transform(x_train[i])
    x_test[i]=scaler.transform(x_test[i])
    
# 검증 데이터 분리
from sklearn.model_selection import train_test_split
X_train,X_valid,Y_train,Y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=10)

# 모델 생성1(RF)
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier()
model1.fit(X_train,Y_train)
pred1=model1.predict_proba(X_valid)

# 모델 생성2(LR)
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_train,Y_train)
pred2=model2.predict_proba(X_valid)

# 모델 생성3(XGB)
from xgboost import XGBClassifier
model3=XGBClassifier()
model3.fit(X_train,Y_train)
pred3=model3.predict_proba(X_valid)

# # 모델 성능 평가
# from sklearn.metrics import roc_auc_score # age 변환 전 후
# print('RF',roc_auc_score(Y_valid,pred1[:,1])) # 0.7747 # 0.7870
# print('LR',roc_auc_score(Y_valid,pred2[:,1])) # 0.2331 # 0.2311
# print('XGB',roc_auc_score(Y_valid,pred3[:,1])) # 0.8025 # 0.8061

# 결과 제출
result=model3.predict_proba(x_test)
pd.DataFrame({'id':id,'TravelInsurance':result[:,1]}).to_csv('travelinsurance_predict.csv',index=False)
