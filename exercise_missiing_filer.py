#用户：xiangjianqun   

#日期：2019-02-20   

#时间：21:47   

#文件名称：PyCharm

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_data=pd.read_csv('cs-training.csv')
pd.set_option('display.max_columns',None)
print(train_data.describe())
print(train_data.isnull().any())
train_data.drop('Unnamed: 0',axis=1,inplace=True)
print(train_data.describe())

#monthlyincome和numberofdependents有缺失值
#用随机森林来对缺失值进行预测

def set_missing(df):
    #把已有的数值型特征取出来
    process_df=df.ix[:,[5,0,1,2,3,4,6,7,8,9]]
    know=process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknow = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    X=know[:,1:]
    y=know[:,0]
    rfr=RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,
                              n_jobs=-1)
    rfr.fit(X,y)
    predicted=rfr.predict(unknow[:,1:]).round(0)
    print(predicted)
    df.loc[(df.MonthlyIncome.isnull()),'MonthlyIncome']=predicted
    return df


data=set_missing(train_data)
data=data.dropna()
data=data.drop_duplicates()
#data.to_csv('MissingData.csv',index=False)

data=pd.read_csv('MissingData.csv')
print(data.describe())

#异常值处理
data=data[data.age>0]
dayslate=['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse']
dayslate_data=[]
for item in dayslate:
    dayslate_data.append(data[item])
import matplotlib.pyplot as plt
plt.boxplot(
    x=dayslate_data,
    patch_artist=True,
    labels=dayslate,
    boxprops={'color':'black','facecolor':'steelblue'},
    flierprops={'marker':'o','markerfacecolor':'red'}
)
plt.ylabel('')
plt.show()
data=data[data['NumberOfTime30-59DaysPastDueNotWorse']<90]
#变量SeriousDlqin2yrs取反
data['SeriousDlqin2yrs']=1-data['SeriousDlqin2yrs']

#数据切分
from sklearn.cross_validation import train_test_split
Y=data['SeriousDlqin2yrs']
x=data.ix[:,1:]
X_train,X_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.3,random_state=0)
train=pd.concat([Y_train,X_train],axis=1)
test=pd.concat([Y_test,X_test],axis=1)
clasTest=test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
train.to_csv('TrainData.csv',index=False)
test.to_csv('TestData.csv',index=False)

#探索性分析
import seaborn as sns
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
sns.distplot(data.age,bins=20,kde=False,hist_kws={'color':'steelblue'})
plt.subplot(2,2,2)
sns.distplot(data.age,hist=False,kde_kws={'color':'red','linestyle':'--'},
             norm_hist=True)
plt.subplot(2,2,3)
sns.distplot(data.MonthlyIncome,bins=20,kde=False,hist_kws={'color':'steelblue'})
plt.subplot(2,2,4)
sns.distplot(data.MonthlyIncome,hist=False,kde_kws={'color':'red','linestyle':'-'},
             norm_hist=True)
plt.show()

#相关性分析
corr=data.corr()
xticks=['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
yticks=list(corr.index)
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
sns.heatmap(corr,annot=True,cmap='rainbow',ax=ax1,annot_kws={'size':9,
                                                             'weight':'bold',
                                                             'color':'blue'})
ax1.set_xticklabels(xticks,rotation=0,fontsize=10)
ax1.set_yticklabels(yticks,rotation=0,fontsize=10)
plt.show()
