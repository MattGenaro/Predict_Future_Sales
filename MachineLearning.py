# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Data visualization
import matplotlib.pyplot as plt 
import seaborn as sns
from xgboost import plot_importance

#Linear algebra
import numpy as np

#Metrics 
from sklearn.metrics import r2_score, mean_squared_error

#Normalization
from sklearn.preprocessing import MinMaxScaler

#Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


#Dataframes of work
df_items = pd.read_csv('/Assets/Predict_Future_Sales/items.csv')
df_item_cat = pd.read_csv('/Assets/Predict_Future_Sales/item_categories.csv')
df_sales_train = pd.read_csv('/Assets/Predict_Future_Sales/sales_train.csv')
df_shops = pd.read_csv('/Assets/Predict_Future_Sales/shops.csv')
df_test = pd.read_csv('/Assets/Predict_Future_Sales/test.csv')

#Optimizes the memory usage to load data faster
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

df_sales_train = downcast_dtypes(df_sales_train)


#Creating new dataframe 'monthly sales'to aggregate features and generate addionatal information for future feature enginnering
df_union = df_sales_train.join(df_items, on='item_id', rsuffix='_').join(df_shops, on='shop_id', rsuffix='_').join(  #Concatenates all information into a single dataframe
        df_item_cat, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1) #dropping repeated columns

monthly_sales = df_union[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', #Monthly sales dataframe
                                 'item_price', 'item_cnt_day']]
monthly_sales = monthly_sales.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', #Group by month
                                         'item_id'], as_index=False)
monthly_sales = monthly_sales.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']}) #Aggregate item's prices and counts in the respective functions
monthly_sales.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', #Rename features.
                         'mean_item_cnt', 'transactions'] 

#Implementing missing records for each shop and item for each month, in order to simulate a real data pattern
monthly_sales['shop_id'].nunique() #amount of unique ids
monthly_sales['item_id'].nunique() 
shop_ids = monthly_sales['shop_id'].unique() #Creating new list with unique shop and items ids
item_ids = monthly_sales['item_id'].unique()
df_combo = []
for i in range(monthly_sales['date_block_num'].max()+1): #range is attached with the max number of months in 'date_block_num'
    for shop in shop_ids:
        for item in item_ids:
            df_combo.append([i, shop, item])
    
df_combo = pd.DataFrame(df_combo, columns=['date_block_num','shop_id','item_id']) #Combines all the possibilities of shops and items ids with months, assuring that no data is missing

monthly_sales = pd.merge(df_combo, monthly_sales, on=['date_block_num','shop_id','item_id'], how='left') #Outer left join df_combo in monthly_sales
monthly_sales.fillna(0, inplace=True) #Data that have no real registered record in the database replaced with zero values

#Due to memory ram issues, here we will use a slice of the dataset to be able to compile the algorithms
monthly_sales.shape[0] #44486280 rows
monthly_sales = monthly_sales.sample(frac=0.6) #Fractionating 60% of the dataset and gathering random data
monthly_sales.shape[0] #New data with 26691768 rows

monthly_sales = downcast_dtypes(monthly_sales)
monthly_sales.head(10)
monthly_sales.describe()

monthly_sales['year'] = monthly_sales['date_block_num'].apply(lambda x: ((x//12) + 2013)) #Creates new feature 'year' using 'date_block_num', which is a codification of months
monthly_sales['month'] = monthly_sales['date_block_num'].apply(lambda x: (x % 12)) #Creates new feature 'months' using 'date_block_num', which is a codification of months

#Now with the complete database aggregated all together, it's possible to make an extra basic EDA
#New parameters of aggregated data for "item_cnt_day" and "item_category_id" features
category_sum = monthly_sales.groupby(['item_category_id'], as_index=False)['item_cnt'].sum()
category_avg = monthly_sales.groupby(['item_category_id'], as_index=False)['item_cnt'].mean()
#Plot
f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
sns.barplot(x="item_category_id", y="item_cnt", data=category_sum, color='royalblue', ax=axes[0])
axes[0].set_title('Total Monthly Sales per Item Category ID', size=16)
axes[0].set_xlabel('Item Category ID', size=12)
axes[0].set_ylabel('Monthly Sales', size=12)
sns.barplot(x="item_category_id", y="item_cnt", data=category_avg, color='darkorange', ax=axes[1])
axes[1].set_title('Avg. Monthly Sales per Item Category ID', size=16)
axes[1].set_xlabel('Item Category ID', size=12)
axes[1].set_ylabel('Monthly Sales', size=12)
plt.tight_layout()
#plt.savefig('MonthlySalesCat.png')
plt.show()

#New parameters of aggregated data for "item_cnt_day" and "shop_id" features
shop_sum = monthly_sales.groupby(['shop_id'], as_index=False)['item_cnt'].sum()
shop_avg = monthly_sales.groupby(['shop_id'], as_index=False)['item_cnt'].mean()
#Plot
f, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True)
sns.barplot(x="shop_id", y="item_cnt", data=shop_sum, color='royalblue', ax=axes[0])
axes[0].set_title('Total Monthly Sales per Item Shop ID', size=16)
axes[0].set_xlabel('Item Shop ID', size=12)
axes[0].set_ylabel('Monthly Sales', size=12)
sns.barplot(x="shop_id", y="item_cnt", data=shop_avg, color='darkorange', ax=axes[1])
axes[1].set_title('Avg. Monthly Sales per Item Shop ID', size=16)
axes[1].set_xlabel('Item Shop ID', size=12)
axes[1].set_ylabel('Monthly Sales', size=12)
plt.tight_layout()
plt.savefig('MonthlySalesShop.png')
plt.show()

#Checking 'item_cnt' and 'item_price' outliers in 'mothly_sales' 
fig, axes = plt.subplots(ncols=2, figsize=(10,4), dpi=100) #Visualization of outliers
plt.xlim(-100, 3000)
sns.boxplot(y=monthly_sales.item_cnt, ax=axes[0])
axes[0].set_ylabel("Item Count")
plt.xlim(monthly_sales.item_price.min(), monthly_sales.item_price.max()*1.1)
sns.boxplot(y=monthly_sales.item_price, ax=axes[1])
axes[1].set_ylabel("Item Price")
plt.suptitle('Outliers Checking', size=16, y=1.1)
plt.tight_layout()
plt.savefig('OutliersMonthlySales.png', bbox_inches='tight')
plt.show()

#Outliers numbers
monthly_sales[monthly_sales.item_cnt<0] #915 items with negative 'item_cnt': represents 0,002% of the total database
monthly_sales[monthly_sales.item_cnt>600] #62 items with negative 'item_cnt': represents 0,00014% of the total database
monthly_sales[monthly_sales.item_price<0] #None found
monthly_sales[monthly_sales.item_price>400000] #22 items in 'item_price' with value greater than 400000 : represents 0,0005% of the total database

#Dropping Outliers
monthly_sales = monthly_sales[monthly_sales.item_cnt>=0]
monthly_sales = monthly_sales[monthly_sales.item_cnt<600]
monthly_sales = monthly_sales[monthly_sales.item_price<400000] 

#Checking duplicated data
dup = monthly_sales[monthly_sales.duplicated(keep=False)] 
len(monthly_sales[monthly_sales.duplicated()]) #No data duplicated


"""
***********************
* Feature Engineering *
************************
"""

#Creating new target label to achieve the objective predictions on months
monthly_sales['item_cnt_month'] = monthly_sales.sort_values('date_block_num').groupby(
        ['shop_id', 'item_id'])['item_cnt'].shift(-1)

#Item price per unit
monthly_sales['item_price_unit'] = monthly_sales['item_price'] // monthly_sales['item_cnt'] 
monthly_sales['item_price_unit'].fillna(0, inplace=True) #filling missing records with zero values

#Max and min values for 'item_price' grouped by months ('date_block_num')
monthly_item_price = monthly_sales.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg(
        {'item_price':[np.min, np.max]})
monthly_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']
monthly_sales = pd.merge(monthly_sales, monthly_item_price, on='item_id', how='left') #Concatenates databases with outer left join

#Price variation in months
monthly_sales['price_increase'] = monthly_sales['item_price'] - monthly_sales['hist_min_item_price']
monthly_sales['price_decrease'] = monthly_sales['hist_max_item_price'] - monthly_sales['item_price']

#Lag features
for i in range(1,4):
    feature_name = ('item_cnt_shifted%s' % i)
    monthly_sales[feature_name] = monthly_sales.sort_values('date_block_num').groupby(
            ['shop_id', 'item_category_id', 'item_id'])['item_cnt'].shift(i)
    
monthly_sales[feature_name].fillna(0, inplace=True) #Fill missing values in features with zero values

#Trend of items monthly sales
monthly_sales['item_trend'] = monthly_sales['item_cnt']

for i in range(1,4): #Simulates a autocorrelation function by subtracting 'item_cnt' from the shifted features to create the trend feature
    feature_name = ('item_cnt_shifted%s' % i)
    monthly_sales['item_trend'] -= monthly_sales[feature_name]

monthly_sales['item_trend'] /= len(range(1,4))+1 #Amount of element in lag input

#Moving parameters for features
mov_min = lambda x: x.rolling(window=3, min_periods=1).min() #Min value
mov_max = lambda x: x.rolling(window=3, min_periods=1).max() #Max value
mov_avg = lambda x: x.rolling(window=3, min_periods=1).mean() #Simple average value
mov_std = lambda x: x.rolling(window=3, min_periods=1).std() #Standard deviation

func_list = [mov_min, mov_max, mov_avg, mov_std]
func_param = ['min', 'max', 'avg', 'std']

for i in range(len(func_list)): #Applies the moving functions to data in order to create the new features
    monthly_sales[('item_cnt_%s' % func_param[i])] = monthly_sales.sort_values('date_block_num').groupby(
            ['shop_id', 'item_category_id', 'item_id'])['item_cnt'].apply(func_list[i])
    
monthly_sales['item_cnt_std'].fillna(0, inplace=True) #Fill missing values in 'item_cnt_std' feature with zero values

#Visualization of the current dataset
monthly_sales.head(10)
monthly_sales.describe()
monthly_sales.head(10).T
monthly_sales.head(10).describe().T


#Training, validation and test datasets splits
train = monthly_sales[(monthly_sales['date_block_num']>=3)&(monthly_sales['date_block_num']<28)]
train.head(10) 
train.date_block_num.max()
val = monthly_sales[(monthly_sales['date_block_num']>=28)&(monthly_sales['date_block_num']<33)]
val.head(10)
val.date_block_num.max()
test = monthly_sales[(monthly_sales['date_block_num']==33)]
test.head(10)
test.date_block_num.max()
#Dropping missing/NaN values from 'item_cnt_month' which is the target feature we want to predict
train['item_cnt_month'].dropna(inplace=True)
val['item_cnt_month'].dropna(inplace=True)
#Dropping missing/NaN values in general
train.isnull().sum()
train.dropna(inplace=True)
val.dropna(inplace=True)

train.shape[0]  #Amount of rows in training dataset
(train.shape[0]/monthly_sales.shape[0])*100 #percentage
val.shape[0] #Amount of rows in validation dataset
(val.shape[0]/monthly_sales.shape[0])*100 #percentage
test.shape[0] #Amount of rows in test dataset
(test.shape[0]/monthly_sales.shape[0])*100 #percentage

#Enconding new attributes
avg_shop = train.groupby(['shop_id']).agg({'item_cnt_month': ['mean']}) #Average on item count per month grouped by item shop id
avg_shop.columns = ['shop_avg']
avg_shop.reset_index(inplace=True)

avg_item = train.groupby(['item_id']).agg({'item_cnt_month': ['mean']}) #Average on item count per month grouped by item id
avg_item.columns = ['item_avg']
avg_item.reset_index(inplace=True)

avg_item_shop = train.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']}) #Average on item count per month grouped by item and shop id
avg_item_shop.columns = ['item_shop_avg']
avg_item_shop.reset_index(inplace=True)

avg_year = train.groupby(['year']).agg({'item_cnt_month': ['mean']}) #Average on item count per month grouped by year
avg_year.columns = ['year_avg']
avg_year.reset_index(inplace=True)

avg_month = train.groupby(['month']).agg({'item_cnt_month': ['mean']}) #Average on item count per month grouped by months
avg_month.columns = ['month_avg']
avg_month.reset_index(inplace=True)

#Averaged encoded features to train dataset
train = pd.merge(train, avg_shop, on=['shop_id'], how='left')
train = pd.merge(train, avg_item, on=['item_id'], how='left')
train = pd.merge(train, avg_item_shop, on=['shop_id', 'item_id'], how='left')
train = pd.merge(train, avg_year, on=['year'], how='left')
train = pd.merge(train, avg_month, on=['month'], how='left')
#Averaged encoded features to validation dataset
val = pd.merge(val, avg_shop, on=['shop_id'], how='left')
val = pd.merge(val, avg_item, on=['item_id'], how='left')
val = pd.merge(val, avg_item_shop, on=['shop_id', 'item_id'], how='left')
val = pd.merge(val, avg_year, on=['year'], how='left')
val = pd.merge(val, avg_month, on=['month'], how='left')

#Assigning features and target for machinel learning models
X_train = train.drop(['item_cnt_month', 'date_block_num'], axis=1)
Y_train = train['item_cnt_month'].astype(int)
X_val = val.drop(['item_cnt_month', 'date_block_num'], axis=1)
Y_val = val['item_cnt_month'].astype(int)

#Target creation from 'df_test' to evaluate model metrics
latest_records = pd.concat([train, val]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
X_test = pd.merge(df_test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])
X_test['year'] = 2015
X_test['month'] = 9
X_test.drop('item_cnt_month', axis=1, inplace=True)
X_test = X_test[X_train.columns]

#Filling missing/NaN values with average 
sets = [X_train, X_val, X_test]
for dataset in sets:
    for shop_id in dataset['shop_id'].unique():
        for column in dataset.columns:
            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()
            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median
            
X_test.fillna(X_test.mean(), inplace=True)

#"item_category_id", we don't have it on test set and would be a little hard to create categories for items that exist only on test set.
X_train.drop(['item_category_id'], axis=1, inplace=True)
X_val.drop(['item_category_id'], axis=1, inplace=True)
X_test.drop(['item_category_id'], axis=1, inplace=True)


X_test.head(10)
X_test.head(10).T
X_test.describe()
X_test.describe().T


"""
*****************************
* Machine learning modeling *
*****************************
"""

#KNN Model
knn_features = ['item_cnt', 'item_cnt_avg', 'item_cnt_std', 'item_cnt_shifted1', #Use only revelant features for KNN
                'item_cnt_shifted2', 'shop_avg', 'item_shop_avg', 
                'item_trend', 'mean_item_cnt']

#Slicing the dataset in random samples for time efficiency
X_train_sampled = X_train.sample(n=100000)
Y_train_sampled = Y_train.sample(n=100000)
#Assigning datasets for the KNN model
knn_train = X_train_sampled[knn_features]
knn_val = X_val[knn_features]
knn_test = X_test[knn_features]
#Features normalization
knn_scaler = MinMaxScaler()
knn_scaler.fit(knn_train)
knn_train = knn_scaler.transform(knn_train)
knn_val = knn_scaler.transform(knn_val)
knn_test = knn_scaler.transform(knn_test)
#Model and fit
knn = KNeighborsRegressor(n_neighbors=9, leaf_size=13, n_jobs=-1)
knn.fit(knn_train, Y_train_sampled)
#Predictions
knn_train_pred = knn.predict(knn_train)
knn_val_pred = knn.predict(knn_val)
knn_test_pred = knn.predict(knn_test)
#Metrics to evaluate the effectiveness of the model
knn_mse_train = mean_squared_error(Y_train_sampled, knn_train_pred) #1.061207037037037
knn_mse_val = mean_squared_error(Y_val, knn_val_pred)  #1.2505306935237535
knn_r2_train = r2_score(Y_train_sampled, knn_train_pred) #-0.00581595765440257
knn_r2_val = r2_score(Y_val, knn_val_pred) #-0.017930425994803656
knn_rmse_train = np.sqrt(mean_squared_error(Y_train_sampled, knn_train_pred)) #1.0301490363229182
knn_rmse_val = np.sqrt(mean_squared_error(Y_val, knn_val_pred)) #1.1182712969238517


#Linear Regression Model
lr_features = ['item_cnt', 'item_cnt_shifted1', 'item_trend', 'mean_item_cnt', 'shop_avg'] #Use only revelant features for Linear Regression
lr_train = X_train[lr_features]
lr_val = X_val[lr_features]
lr_test = X_test[lr_features]
#Features normalization
lr_scaler = MinMaxScaler()
lr_scaler.fit(lr_train)
lr_train = lr_scaler.transform(lr_train)
lr_val = lr_scaler.transform(lr_val)
lr_test = lr_scaler.transform(lr_test)
#Model
lr = LinearRegression(n_jobs=-1)
lr.fit(lr_train, Y_train)
#Predictions
lr_train_pred = lr.predict(lr_train)
lr_val_pred = lr.predict(lr_val)
lr_test_pred = lr.predict(lr_test)
#Metrics to evaluate the effectiveness of the model
lr_mse_train = mean_squared_error(Y_train, lr_train_pred) #0.9374013718267399
lr_mse_val = mean_squared_error(Y_val, lr_val_pred)  #0.7882697108509823
lr_r2_train = r2_score(Y_train, lr_train_pred) #0.3970577089684093
lr_r2_val = r2_score(Y_val, lr_val_pred) #0.3583494377860309
lr_rmse_train = np.sqrt(mean_squared_error(Y_train, lr_train_pred)) #0.9681949038425786
lr_rmse_val = np.sqrt(mean_squared_error(Y_val, lr_val_pred)) #0.8878455444788707


#Random Forest
rf_features = ['shop_id', 'item_id', 'item_cnt', 'transactions', 'year',  #Use only relevant of features for RF
               'item_cnt_avg', 'item_cnt_std', 'item_cnt_shifted1', 
               'shop_avg', 'item_avg', 'item_trend', 'mean_item_cnt']
#Slicing the dataset in random samples due to memory usage
X_train_sampled = X_train.sample(n=1000000)
Y_train_sampled = Y_train.sample(n=1000000)
#Assigning datasets for the RF model
rf_train = X_train_sampled[rf_features]
rf_val = X_val[rf_features]
rf_test = X_test[rf_features]
#Model
rf = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0, n_jobs=-1)
rf.fit(rf_train, Y_train_sampled)
#Importances visualization
rf_feature = rf.feature_importances_
importances_rf = pd.DataFrame({'Features': rf_train.iloc[:, 0:13].columns, 'Importance': np.round(rf.feature_importances_, 3)})
plt.rcParams["figure.figsize"] = (15, 6)
sns.set_style("darkgrid")
sns.barplot("Features", "Importance", data=importances_rf.sort_values(by='Importance', ascending=False), color='royalblue', alpha=0.6)
plt.savefig('RFImportances.png')
plt.show()
#Predictions
rf_train_pred = rf.predict(rf_train)
rf_val_pred = rf.predict(rf_val)
rf_test_pred = rf.predict(rf_test)
#Metrics to evaluate the effectiveness of the model
rf_mse_train = mean_squared_error(Y_train_sampled, rf_train_pred) #1.80641172311407
rf_mse_val = mean_squared_error(Y_val, rf_val_pred)  #1.2686622011736688
rf_r2_train = r2_score(Y_train_sampled, rf_train_pred) #0.1334953046546541
rf_r2_val = r2_score(Y_val, rf_val_pred) #-0.03268945062457829
rf_rmse_train = np.sqrt(mean_squared_error(Y_train_sampled, rf_train_pred)) #1.3440281705061357
rf_rmse_val = np.sqrt(mean_squared_error(Y_val, rf_val_pred)) #1.1263490583179216


#XGBoost
xgb_features = ['item_cnt','item_cnt_avg', 'item_cnt_std', 'item_cnt_shifted1', #Use only relevant of features for XGBoost
                'item_cnt_shifted2', 'item_cnt_shifted3', 'shop_avg', 
                'item_shop_avg', 'item_trend', 'mean_item_cnt']
#Slicing the dataset in random samples due to memory usage
X_train_sampled = X_train.sample(n=1000000)
Y_train_sampled = Y_train.sample(n=1000000)
#Assigning datasets for the XGB model
xgb_train = X_train_sampled[xgb_features]
xgb_val = X_val[xgb_features]
xgb_test = X_test[xgb_features]
#Model
xgb = XGBRegressor(max_depth=8, n_estimators=500, min_child_weight=1000, 
                   colsample_bytree=0.7, subsample=0.7, eta=0.3, seed=0)
#Fit
xgb.fit(xgb_train, Y_train_sampled, 
        eval_metric="rmse", eval_set=[(xgb_train, Y_train_sampled), (xgb_val, Y_val)],
        verbose=20, early_stopping_rounds=20)
#Importances visualization
plt.rcParams["figure.figsize"] = (15, 6)
plot_importance(xgb)
plt.savefig('XGBImportances.png')
plt.show()
#Predictions
xgb_train_pred = xgb.predict(xgb_train)
xgb_val_pred = xgb.predict(xgb_val)
xgb_test_pred = xgb.predict(xgb_test)
#Metrics to evaluate the effectiveness of the model
xgb_mse_train = mean_squared_error(Y_train_sampled, xgb_train_pred) #1.7566673927358547
xgb_mse_val = mean_squared_error(Y_val, xgb_val_pred)  #1.2347392207036014
xgb_r2_train = r2_score(Y_train_sampled, xgb_train_pred) #-0.0013224421341628823
xgb_r2_val = r2_score(Y_val, xgb_val_pred) #-0.005076186800076377
xgb_rmse_train = np.sqrt(mean_squared_error(Y_train_sampled, xgb_train_pred)) #1.3253932973785008
xgb_rmse_val = np.sqrt(mean_squared_error(Y_val, xgb_val_pred)) #1.1111882021978101


#Plotting RMSE values
rmse_values = [knn_rmse_val, lr_rmse_val, rf_rmse_val, xgb_rmse_val]
rmse_columns = ['KNN', 'Linear Regression', 'Random Forest', 'XGBoost']
df_rmse = {'Models': rmse_columns, 'RMSE values': rmse_values}
df_rmse = pd.DataFrame(df_rmse)
plt.rcParams["figure.figsize"] = (6, 4)
sns.set_style("darkgrid")
fig, ax = plt.subplots()    
sns.barplot("Models", "RMSE values", data=df_rmse, color='royalblue', alpha=0.6)
plt.xlabel('Models', size=12)
plt.ylabel('RMSE values', size=12)
plt.title('RMSE Values for Each Model', size=14)
plt.savefig('RMSEModels.png')
plt.show()


#Meta modeling via ensemble model with 2 levels

#Creation of train datasets for the first level model
model_lvl_1 = pd.DataFrame(knn_val_pred, columns=['knn'])
model_lvl_1['xgboost'] = xgb_val_pred
model_lvl_1['random_forest'] = rf_val_pred
model_lvl_1['linear_regression'] = lr_val_pred
model_lvl_1['label'] = Y_val.values
model_lvl_1.head(10)
#Creation of test datasets for the first level model
model_lvl_1_test = pd.DataFrame(knn_test_pred, columns=['knn'])
model_lvl_1_test['xgboost'] = xgb_test_pred
model_lvl_1_test['random_forest'] = rf_test_pred
model_lvl_1_test['linear_regression'] = lr_test_pred
model_lvl_1_test.head(10)
#Model level 2
meta_model = LinearRegression(n_jobs=-1)
#Fitting
model_lvl_1.drop('label', axis=1, inplace=True) #target feature as label dropped
meta_model.fit(model_lvl_1, Y_val)
#Predictions
ensemble_pred = meta_model.predict(model_lvl_1)
final_pred = meta_model.predict(model_lvl_1)
#Metrics to evaluate the effectiveness of the model
meta_mse = mean_squared_error(ensemble_pred, Y_val)  #0.7678863018147837
meta_r2 = r2_score(ensemble_pred, Y_val) #-0.6670826793773117
meta_rmse = np.sqrt(mean_squared_error(ensemble_pred, Y_val)) #0.8762912197521916
#Plotting values
df_meta = pd.DataFrame({'MSE': meta_mse, 'RMSE': meta_rmse, 'R2 Score': meta_r2}, index=[0])
plt.rcParams["figure.figsize"] = (6, 4)
sns.set_style("darkgrid")
fig, ax = plt.subplots()    
sns.barplot(data=df_meta, color='darkorange', alpha=0.7)
plt.xlabel('Metrics', size=12)
plt.ylabel('Scores', size=12)
plt.title('Metrics Scores of Ensemble Model', size=14)
plt.savefig('MetaScores.png')
plt.show()

