# %% [code] {"jupyter":{"outputs_hidden":false}}
# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
# (단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력)

# - 데이터셋 : ../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script


import pandas as pd

df=pd.read_csv('../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv')

df_ratio=df.groupby('country')[['ratio']].mean().reset_index(drop=False)
df_ratio.sort_values('ratio',ascending=False,inplace=True)
df_ratio=df_ratio[df_ratio['ratio']<100]
print(df_ratio)

high10=df_ratio.head(10).mean()
low10=df_ratio.tail(10).mean()
print(round(high10-low10,2))


# import pandas as pd

# df = pd.read_csv("../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv")
# print(df.head())

# df2 = df.groupby('country').max() #시간에 따라 접종률이 점점 올라감
# df2 = df2.sort_values(by='ratio', ascending = False)
# # print(df2['ratio'].head())

# df2 = df2[1:] #이상치 제거
# # print(df2['ratio'].head())

# top = df2['ratio'].head(10).mean()
# bottom = df2['ratio'].tail(10).mean()

# print(round(top - bottom,1))
# # 결과값은 데이터 업데이트에 따라 달라질 수 있음