
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
# quandl for financial data
import quandl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgbm
#import plotly.graph_objs as go
from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import gc
import time
import sys
import datetime

from sklearn.metrics import mean_squared_error

# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import plot_importance


# # Creating the Dataset

# In[4]:


quandl.ApiConfig.api_key = '4tsMeS-sBH-7zqjZYxdP'
gdp = quandl.get("FRED/GDPC1", authtoken="4tsMeS-sBH-7zqjZYxdP") # quarterly in Billions 
avghrs = quandl.get("FRED/PRS30006022", authtoken="4tsMeS-sBH-7zqjZYxdP") # quarterly in Billions 


# In[5]:


gdp.reset_index(inplace=True)
avghrs.reset_index(inplace=True)


# In[11]:


avghrs.rename(columns={"Value": "avghrs"},inplace=True)


# # weekly Unemployment claims from the FRED

# In[6]:


weeklyclaims = pd.read_csv("WeeklyUnemploymentClaims.csv")


# In[7]:


weeklyclaims['WeeklyClaims_qrt'] = weeklyclaims['Initial Claims'].groupby(weeklyclaims['Quarter Date']).transform('mean')


# In[8]:


weeklyclaims2 = weeklyclaims.drop(['Date','Initial Claims','Quarter'],axis=1).drop_duplicates().dropna()


# In[9]:


weeklyclaims2 


# Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft

# In[12]:


neworders = quandl.get("FRED/NEWORDER", authtoken="4tsMeS-sBH-7zqjZYxdP",collapse="quarterly") # quarterly in Millions of Dollars Seasonally Adjusted


# In[13]:


neworders.reset_index(inplace=True)


# In[14]:


neworders.rename(columns={"Value": "neworders"},inplace=True)


# In[15]:


neworders


# ISM  Manufacturing New Orders Index

# In[16]:


ism_neworder = quandl.get("ISM/MAN_NEWORDERS", authtoken="4tsMeS-sBH-7zqjZYxdP",collapse="quarterly")


# In[17]:


ism_neworder.reset_index(inplace=True)


# In[18]:


ism_neworder.rename(columns={"Index": "ism_neworder"},inplace=True)


# In[19]:


ism_neworder


#  New Private Housing Units Authorized by Building Permits

# In[20]:


building_premits = quandl.get("FRED/PERMIT", authtoken="4tsMeS-sBH-7zqjZYxdP",collapse="quarterly") 
#Thousands of Units Seasonally Adjusted


# In[21]:


building_premits.reset_index(inplace=True)


# In[22]:


building_premits.rename(columns={"Value": "building_premits"},inplace=True)


# In[23]:


building_premits


# SP500

# In[24]:


sp = quandl.get("YALE/SPCOMP", authtoken="4tsMeS-sBH-7zqjZYxdP", collapse="quarterly")


# In[25]:


sp.reset_index(inplace=True)


# In[26]:


sp.tail()


# 10-Year Treasury Constant Maturity Minus Federal Funds Rate

# In[27]:


T10YFF = quandl.get("FRED/T10YFF", authtoken="4tsMeS-sBH-7zqjZYxdP", collapse="quarterly")


# In[28]:


T10YFF.reset_index(inplace=True)


# In[29]:


T10YFF.rename(columns={"Value": "10yTRY-FF"},inplace=True)


# In[30]:


T10YFF


# Consumer Opinion Surveys: Confidence Indicators: Composite Indicators: OECD Indicator for the United States 

# In[31]:


quandl.get("OECD/KEI_CSCICP02_USA_ST_M", collapse="quarterly")


# In[32]:


consumer_confidence = quandl.get("OECD/KEI_CSCICP02_USA_ST_M", authtoken="4tsMeS-sBH-7zqjZYxdP", collapse="quarterly")


# In[33]:


consumer_confidence.reset_index(inplace=True)


# In[34]:


consumer_confidence.rename(columns={"Value": "Consumer_confidence"},inplace=True)


# In[35]:


consumer_confidence


# # Building the Dataset

# In[36]:


gdp


# In[37]:


gdp['Change'] = ((gdp.Value / gdp.Value.shift(1)) ** 4) - 1


# In[38]:


gdp['gdp_sign'] = np.sign(gdp['Change'])


# In[39]:


gdp.info()


# In[40]:


weeklyclaims2['Quarter Date']= pd.to_datetime(weeklyclaims2['Quarter Date'],format='%Y-%m-%d')


# In[41]:


weeklyclaims2[['Quarter Date','WeeklyClaims_qrt']]


# In[42]:


gpdAvgHrs = gdp.merge(avghrs, how='left',left_on='Date', right_on='Date')


# In[43]:


gdpMerge = gpdAvgHrs.merge(weeklyclaims2,how='left',left_on='Date', right_on='Quarter Date')
gdpMerge


# In[44]:


gdpMerge.drop(['Quarter Date'],axis=1, inplace=True)


# In[45]:


gdpMerge['Quarter'] = gdpMerge['Date'].dt.quarter 
gdpMerge['Year'] = gdpMerge['Date'].dt.year 


# In[46]:


gdpMerge


# In[47]:


a = ism_neworder.merge(neworders,how='left', left_on='Date', right_on='Date')

b = a.merge(building_premits,how='left', left_on='Date', right_on='Date')

c = b.merge(sp, how='left', left_on='Date', right_on='Year')

d = c.merge(T10YFF,how='left', left_on='Date', right_on='Date')

e = d.merge(consumer_confidence,how='left', left_on='Date', right_on='Date')


# In[48]:


e


# In[49]:


e.drop(['% Better','% Same','% Worse','Net','Year','Dividend','Earnings','CPI','Long Interest Rate','Real Price','Real Dividend','Real Earnings','Cyclically Adjusted PE Ratio'],axis=1,inplace=True)


# In[50]:


e


# In[51]:


e['Quarter'] = e['Date'].dt.quarter 


# In[52]:


e['Year'] = e['Date'].dt.year 


# In[53]:


e


# In[664]:


df = gdpMerge.merge(e, how='left',left_on=['Quarter','Year'], right_on=['Quarter','Year'])


# In[665]:


df.drop('Date_y',axis=1, inplace=True)


# In[666]:


df.rename(columns={'Date_x':'Date','Value':'GDP', "Change":"GDP_Change" },inplace=True)


# In[667]:


df['SP500_pct'] = df['S&P Composite'].pct_change()


# In[668]:


df.tail()


# In[669]:


df[df['gdp_sign'] == 1].GDP.count() /len(df)


# In[670]:


df[df['gdp_sign'] == -1].GDP.count() /len(df)


# # What to do with Nulls Values?

# In[61]:


df.info()


# In[62]:


data_noNull = df.drop(['avghrs', 'neworders'], axis=1)


# In[63]:


data_noNull = data_noNull.dropna()


# In[67]:


data_noNull


# In[60]:


df['WeeklyClaims_qrt'].plot.hist(bins=100, alpha=0.5)


# In[61]:


df['WeeklyClaims_qrt'].isna().sum()


# In[62]:


df.hist(column='avghrs')


# In[63]:


df['WeeklyClaims_qrt'].mean()


# In[64]:


df['WeeklyClaims_qrt'].median(skipna = True)


# In[65]:


df['WeeklyClaims_qrt'].mode(dropna=True)


# In[66]:


df[(df['avghrs']==-1.4)].count()


# In[67]:


df.describe()


# In[68]:


df['avghrs'].nsmallest()


# In[69]:


df.corr()


# In[108]:


# Take out date
df2 = df.iloc[:,1:]


# In[109]:


cols = df2.columns


# In[110]:


cols


# In[111]:


cols= ['GDP', 'GDP_Change', 'gdp_sign', 'avghrs', 'WeeklyClaims_qrt',
       'Quarter', 'Year', 'ism_neworder', 'neworders', 'building_premits',
       'S&P Composite', '10yTRY-FF', 'Consumer_confidence', 'SP500_pct']


# Inpute Nearest Neighbors

# In[112]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df_filled = imputer.fit_transform(df2)
dataframe=pd.DataFrame(df_filled, columns = cols)


# In[323]:


imputer = KNNImputer(n_neighbors=2)


# In[656]:


df_filled = imputer.fit_transform(df2)


# In[657]:


df_filled


# In[658]:


dataframe=pd.DataFrame(df_filled, columns = cols)


# In[659]:


dataframe.tail()


# Add pct changes

# In[169]:


dataframe['avghrs_pct'] = dataframe['avghrs'].pct_change()
dataframe['WeeklyClaims_qrt_pct'] = dataframe['WeeklyClaims_qrt'].pct_change()
dataframe['ism_neworder_pct'] = dataframe['ism_neworder'].pct_change()
dataframe['neworders_pct'] = dataframe['neworders'].pct_change()
dataframe['building_premits_pct'] = dataframe['building_premits'].pct_change()
dataframe['Consumer_confidence_pct'] = dataframe['Consumer_confidence'].pct_change()
dataframe['10yTRY-FF_pct'] = dataframe['10yTRY-FF'].pct_change()



# In[661]:


dataframe.columns


# # Remove Outliers

# In[71]:


y = data_noNull['GDP_Change']
removed_outliers = y.between(y.quantile(.05), y.quantile(.95))

index_names = data_noNull[~removed_outliers].index # INVERT removed_outliers!!
print(index_names) # The resulting 22 dates to drop.
data_noNull_df = data_noNull.drop(index_names)


# In[72]:


removed_outliers


# In[73]:


index_names = data_noNull[~removed_outliers].index # INVERT removed_outliers!!
print(index_names) # The resulting 22 dates to drop.


# In[74]:


data_noNull_df = data_noNull.drop(index_names)


# In[75]:


data_noNull_df


# In[76]:


data_noNull_df2 = data_noNull_df.drop(['Date','GDP','gdp_sign', 'Year','S&P Composite'],axis=1)


# In[120]:


dataframe.columns


# In[118]:


data = dataframe.drop(['Year', 'GDP', 'S&P Composite', 'gdp_sign','avghrs','avghrs_pct','WeeklyClaims_qrt_pct',
        'ism_neworder','neworders_pct', 'neworders', 'building_premits_pct',
       'S&P Composite', '10yTRY-FF_pct', 'Consumer_confidence_pct'], axis =1)


# In[173]:


data


# In[174]:


data = data.iloc[1:,:]


# In[82]:


#Separate Features and target
X_Target = data_noNull_df2['GDP_Change'] # the 
feature_df= data_noNull_df2.loc[:, data_noNull_df2.columns != 'GDP_Change']
#Split the data
X_train, X_test, y_train, y_test = train_test_split(feature_df, X_Target, test_size=0.3, random_state=0)


# In[83]:


#Split the data
X_train, X_test, y_train, y_test = train_test_split(feature_df, X_Target, test_size=0.3, random_state=0)


# In[84]:


features = list(feature_df.columns)
features =  [i for i in features if i != 'GDP_Change' ]
features


# # Regression Model

# Hypertuning

# In[85]:



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [0, 1, 2, 3, 4, 5, 6, 7],
              'min_child_weight': [1e-5, 1e-3, 1e-2],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [100, 200, 300, 400, 500]}


# In[86]:


xgbm_reg = xgb.XGBRegressor()
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
xgrid_search = GridSearchCV(
    estimator = xgbm_reg,
    param_grid = parameters, 
    n_jobs = 5,
    refit =True,
    cv = 2,
    verbose = 3)
#n_iter = n_iter,random_state = 42,
xgrid_search.fit(X_train,y_train)
xg_best=xgrid_search.best_estimator_
opt_parameters = xgrid_search.best_params_


# In[87]:


opt_parameters


# In[ ]:


opt_parameters_reg = {'colsample_bytree': 0.7,
 'learning_rate': 0.03,
 'max_depth': 1,
 'min_child_weight': 1e-05,
 'n_estimators': 200,
 'nthread': 4,
 'objective': 'reg:linear',
 'silent': 1,
 'subsample': 0.7}


# In[ ]:


xg_best


# In[ ]:


opt_parameters_reg


# In[88]:


xgb_model_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.7, learning_rate = 0.03,
                max_depth = 1,silent= 1 ,nthread= 4, alpha = 10,min_child_weight = 1e-05, n_estimators = 150)
xgb_model_reg.fit(X_train, y_train)


# Predicting and checking the results

# In[89]:


scores = cross_val_score(xgb_model_reg, X_train,y_train,cv=3)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[90]:


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgb_model_reg, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# In[91]:


ypred = xgb_model_reg.predict(X_test)
mse = mean_squared_error(y_test,ypred)
print("MSE: %.4f" % mse)

print("RMSE: %.4f" % np.sqrt(mse))
 


# In[ ]:


y_test.to_csv('y_test.csv')


# In[ ]:


ypred_df = pd.DataFrame(ypred)


# In[352]:


ypred_df.to_csv('ypred.csv')


# In[92]:


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test,ypred)


# In[93]:


from sklearn.metrics import max_error
max_error(y_test,ypred)


# In[94]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, ypred)
r2


# In[95]:


Adjr2 = 1-(1-r2)*(293-1)/(293-9-1)
Adjr2


# In[96]:


from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, ypred)


# In[97]:


x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[110]:


# make predictions for test data
y_pred = xgb_model_reg.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[111]:


y_test


# In[112]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))


# In[113]:


X, y = df3.iloc[:,:-1],df3.iloc[:,-1]


# In[ ]:


data_dmatrix = xgb.DMatrix(data=X,label=y)


# In[ ]:


data_dmatrix


# In[ ]:



cv_results = xgb.cv(dtrain=data_dmatrix, params=opt_parameters_reg, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[ ]:


cv_results.head()


# In[ ]:


print((cv_results["test-rmse-mean"]).tail(1))


# In[ ]:


xg_reg = xgb.train(params=opt_parameters_reg, dtrain=data_dmatrix, num_boost_round=10)


# In[98]:


xgb.plot_importance(xgb_model_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[100]:


data_noNull_df2.corr().style.background_gradient(cmap='Blues')


# In[116]:


xgb.plot_importance(cv_results)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[ ]:


feature_names = pd.Series(feature_df.columns)


# In[ ]:


print(xgb.feature_importances_)


# In[ ]:


feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = feature_names
feature_importance_df["importance"]=xg_reg.feature_importances_


# In[281]:


data_noNull_df


# In[327]:


dataframe['avghrs_pct'] = dataframe['avghrs'].pct_change()
dataframe['WeeklyClaims_qrt_pct'] = dataframe['WeeklyClaims_qrt'].pct_change()
dataframe['ism_neworder_pct'] = dataframe['ism_neworder'].pct_change()
dataframe['neworders_pct'] = dataframe['neworders'].pct_change()
dataframe['building_premits_pct'] = dataframe['building_premits'].pct_change()
dataframe['Consumer_confidence_pct'] = dataframe['Consumer_confidence'].pct_change()
dataframe['10yTRY-FF_pct'] = dataframe['10yTRY-FF'].pct_change()


# In[328]:


dataframe2 = dataframe.iloc[1:,]
dataframe2.columns


# In[329]:


dataframe3 = dataframe2.drop(['GDP','gdp_sign','Year','S&P Composite','avghrs_pct','10yTRY-FF_pct'],axis=1)
dataframe3.info()


# # ElasticNet Regression 

# In[283]:


from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# In[330]:


#Separate Features and target
X_Target = dataframe3['GDP_Change'] # the 
feature_df= dataframe3.loc[:, dataframe3.columns != 'GDP_Change']
#Split the data
X_train, X_test, y_train, y_test = train_test_split(feature_df, X_Target, test_size=0.3, random_state=0)

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

for a in alphas:
    model = ElasticNet(alpha=a).fit(feature_df, X_Target)   
    score = model.score(feature_df, X_Target)
    pred_y = model.predict(feature_df)
    mse = mean_squared_error(X_Target, pred_y)   
    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, score, mse, np.sqrt(mse)))


# In[331]:



alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]


# In[332]:


for a in alphas:
    model = ElasticNet(alpha=a).fit(feature_df, X_Target)   
    score = model.score(feature_df, X_Target)
    pred_y = model.predict(feature_df)
    mse = mean_squared_error(X_Target, pred_y)   
    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, score, mse, np.sqrt(mse)))


# The result shows that we can use 0.01 value for our model.

# In[333]:


elastic=ElasticNet(alpha=0.01).fit(X_train, y_train)
ypred = elastic.predict(X_test)
score = elastic.score(X_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}"
      .format(score, mse, np.sqrt(mse)))


# In[338]:


x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[335]:


elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(X_train, y_train)
print(model.alpha_)

print(model.intercept_)


# In[336]:


elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(X_train, y_train)

ypred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}"
      .format(score, mse, np.sqrt(mse)))


# In[337]:


score


# In[322]:


dataframe3.iloc[:,:10]


# In[339]:


dataframe4 = dataframe3.drop('neworders',axis=1)


# In[341]:



dataframe4.iloc[:,:9].corr().style.background_gradient(cmap='Blues')


# In[343]:


dataframe


# In[359]:


df


# In[375]:


import matplotlib.dates as mdates


# In[363]:


fig, ax1 = plt.subplots(figsize=(15, 5))
df.set_index('Year')[['GDP_Change']].plot(figsize=(20, 8), linewidth=2.5,marker='o', markerfacecolor='black',ax=ax1)
df.set_index('Year')[['ism_neworder','Consumer_confidence']].plot(figsize=(20, 8), linewidth=2.5,secondary_y=True,ax=ax1)
plt.ylabel("Return",fontsize=20)
plt.xlabel("index",fontsize=20)
plt.title("GDP growth vs  ISM new orders and Consumer Confidence", y=1.02, fontsize=30);
#df.plot(style='.-')

#plt.xlabel(fontsize=40)

#plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)


# In[448]:


fig, ax1 = plt.subplots(figsize=(25, 5))
dataframe.set_index('Year')[['GDP_Change']].plot(figsize=(20, 8), linewidth=2.5,marker='o', markerfacecolor='black',ax=ax1)
dataframe.set_index('Year')[['ism_neworder','Consumer_confidence']].plot(figsize=(20, 8), linewidth=2.5,secondary_y=True,ax=ax1)
plt.ylabel("Return",fontsize=20)

plt.title("GDP growth vs  ISM new orders and Consumer Confidence", y=1.02, fontsize=30);
#df.plot(style='.-')

#plt.xlabel(fontsize=40)

#plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)


# In[383]:


fig, ax1 = plt.subplots(figsize=(15, 5))

dataframe.set_index('Year')['GDP_Change'].plot(kind='bar', color='y')
#dataframe.set_index('Year')['ism_neworder'].plot(kind='line', marker='d', secondary_y=True)
#set ticks every week
ax1.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y'))


# In[442]:


dataframe['Year']= pd.to_datetime(dataframe['Year'],format='%Y')


# In[443]:


dataframe.info()


# In[415]:


dataframe[dataframe.index>'1980']


# In[508]:


graph['Date'] = graph['Date'].values.astype('datetime64[D]')


# In[505]:


graph['Date'] = pd.to_timedelta(graph['Date'])


# In[507]:


graph['Date']


# In[515]:


graph = df[df.Year>2000]


# In[525]:


graph.set_index('year_qrt',inplace=True)


# In[523]:


graph['year_qrt'] = graph['Year'].astype(str) + "-Q" +graph['Quarter'].astype(str)


# In[524]:


graph.info()


# In[520]:


fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.bar(graph['year_qrt'], graph['GDP_Change'])


# In[598]:


fig, ax = plt.subplots(figsize=(30,15))
chart = graph[['GDP_Change']].plot.bar(rot= 80, title="GDP Growth vs ISM New Orders and Consumer Confidence",ax=ax)
graph[['ism_neworder','Consumer_confidence']].plot(kind='line', rot= 80, marker='d', secondary_y=True,ax=ax)
plt.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
chart.set_xticklabels(chart.get_xticklabels(), rotation=85,fontsize=16)
#ax.legend(loc=9, bbox_to_anchor=(1, -0.2),fontsize=14)

#ax1.legend(loc=9, bbox_to_anchor=(1, -0.3),fontsize=14)


# In[611]:


graph['Quarter'].nunique()


# In[ ]:


mask1 = y < 0.5
mask2 = y >= 0.5


# In[608]:


from matplotlib import pyplot as plt
from itertools import cycle, islice
import pandas, numpy as np 


# In[626]:


my_colors = list(islice(cycle(['b', 'r', 'g', 'y']), None, len(graph)))


# In[634]:


fig, ax1 = plt.subplots(figsize=(20, 8))
graph['GDP_Change'].plot(kind='bar',stacked=True, color = my_colors, ax=ax1)


# In[649]:


fig, ax1 = plt.subplots(figsize=(20, 8))
#plt.style.use('fast')
ax2 = ax1.twinx()
chart = graph['GDP_Change'].plot(kind='bar',color = my_colors, ax=ax1)
graph[['ism_neworder','Consumer_confidence']].plot(kind='line', marker='d', ax=ax2)

ax1.set_ylabel("GDP % change",fontsize=14)
ax2.set_ylabel("ISM(Millions $) and Confidence(base=100)",fontsize=14)
ax1.set_ylim([-0.10,0.08])

chart.set_xticklabels(chart.get_xticklabels(), rotation=85,fontsize=16)
ax2.legend(loc=9, bbox_to_anchor=(1.02, -0.2),fontsize=14)

plt.title("GDP Growth vs ISM New Orders and Consumer Confidence", y=1.02, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
# Turns off grid on the left Axis.
ax1.grid(False)
ax2.grid(False)
# Turns off grid on the secondary (right) Axis.
#ax1.right_ax(False)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='g', lw=4),
               Line2D([0], [0], color='y', lw=4)]


#lines = ax.plot(data)
ax1.legend(custom_lines, ['Q1', 'Q2', 'Q3','Q4'],bbox_to_anchor=(1, -0.3),fontsize=14)
#ax1.legend(loc=9, bbox_to_anchor=(1, -0.3),fontsize=14)


# In[655]:


dataframe.tail()


# Added other varaibles to improve the model

# In[653]:


UNRATE= pd.read_csv('UNRATE.csv')


# In[682]:


UNRATE


# In[681]:


UNRATE.info()


# In[680]:


UNRATE['Data']= pd.to_datetime(UNRATE['Data'],format='%Y-%m-%d')


# In[677]:


df.info()


# In[687]:


m1 = df.merge(UNRATE,how='left',left_on='Date', right_on='Data')


# In[689]:


m1.head(10)


# In[698]:


pce= pd.read_csv('PCEPILFE.csv')


# In[699]:


pce


# In[700]:


pce['DATE']=pd.to_datetime(pce['DATE'],format='%Y-%m-%d')


# In[701]:


m2 = m1.merge(pce,how='left',left_on='Date', right_on='DATE')


# In[702]:


m2


# In[716]:


mgt = pd.read_csv('MDSP.csv')


# In[718]:


mgt['DATE'] = pd.to_datetime(mgt['DATE'],format='%Y-%m-%d')


# In[719]:


m3 = m2.merge(mgt,how='left',left_on='Date', right_on='DATE')


# In[720]:


m3


# In[721]:


housesupply = pd.read_csv('MSACSR.csv')


# In[722]:


housesupply['DATE'] = pd.to_datetime(housesupply['DATE'],format='%Y-%m-%d')


# In[723]:


m4 = m3.merge(housesupply,how='left',left_on='Date', right_on='DATE')


# In[731]:


m5 = m4.drop(['Date','Data','DATE_x','DATE_y','DATE'],axis=1 )


# In[732]:


m5.columns


# In[733]:


cols = ['GDP', 'GDP_Change', 'gdp_sign', 'avghrs', 'WeeklyClaims_qrt',
       'Quarter', 'Year', 'ism_neworder', 'neworders', 'building_premits',
       'S&P Composite', '10yTRY-FF', 'Consumer_confidence', 'SP500_pct',
       'Un_Rate', 'PCE_PC1', 'mgt_CH1', 'MSACSR']


# In[734]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)


# In[736]:


df_filled = imputer.fit_transform(m5)


# In[737]:


dataframe2=pd.DataFrame(df_filled, columns = cols)


# In[744]:


dataframe3 =dataframe2.drop(['GDP','gdp_sign','S&P Composite','Year'],axis=1 )


# In[745]:


#Separate Features and target
X_Target = dataframe3['GDP_Change'] # the 
feature_df= dataframe3.loc[:, dataframe3.columns != 'GDP_Change']
#Split the data
X_train, X_test, y_train, y_test = train_test_split(feature_df, X_Target, test_size=0.3, random_state=0)

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

for a in alphas:
    model = ElasticNet(alpha=a).fit(feature_df, X_Target)   
    score = model.score(feature_df, X_Target)
    pred_y = model.predict(feature_df)
    mse = mean_squared_error(X_Target, pred_y)   
    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, score, mse, np.sqrt(mse)))


# In[746]:


elastic=ElasticNet(alpha=0.01).fit(X_train, y_train)
ypred = elastic.predict(X_test)
score = elastic.score(X_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}"
      .format(score, mse, np.sqrt(mse)))


# In[753]:


x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[748]:


elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(X_train, y_train)

ypred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}"
      .format(score, mse, np.sqrt(mse)))


# In[750]:


dataframe3.corr()


# In[754]:


dataframe3.corr().style.background_gradient(cmap='Blues')


# In[755]:


ab = dataframe3.drop('neworders',axis=1)


# In[756]:


ab.rename(columns={'MSACSR':'housesupply'},inplace=True)


# In[757]:


ab


# In[758]:


ab.corr().style.background_gradient(cmap='Blues')


# In[772]:


q220 = pd.read_csv('Test2020Q2.csv')


# In[773]:


q220


# In[760]:


dataframe2.tail(1)


# In[764]:


dataframe3.tail(1).iloc[:,1:]


# In[774]:


ypred_2020Q2 = elastic.predict(q220)


# In[777]:


ypred_2020Q2


# 2020 quater 2 will experience a -24% drop in real GDP
