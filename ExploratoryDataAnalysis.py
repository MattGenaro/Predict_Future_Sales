# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Date time formats
import datetime

#Statistical classes and functions
import statsmodels.api as sm

#Data visualization
import matplotlib.pyplot as plt 
import seaborn as sns
from pylab import rcParams

#Linear algebra
import numpy as np

#Dataframes of work
df_items = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Predict_Future_Sales/items.csv', engine='python')
df_item_cat = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Predict_Future_Sales/item_categories.csv', engine='python')
df_sales_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Predict_Future_Sales/sales_train.csv', engine='python')
df_shops = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Predict_Future_Sales/shops.csv', engine='python')
df_test = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Predict_Future_Sales/test.csv', engine='python')

#First look into sales data
df_sales_train.head(5)
df_sales_train.shape 
df_sales_train.columns
df_sales_train.dtypes
df_sales_train.describe()

#Optimizes the memory usage to load data faster
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

df_sales_train = downcast_dtypes(df_sales_train)
df_sales_train.info()

#Change the date time to proper format
df_sales_train.date = df_sales_train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
df_sales_train.date.head(10)
df_sales_train.info()

#Checking and clearing data
df_sales_train.isnull().sum() #Checking missing values. No missing values found
df_sales_train.isnull().values.any()
df_sales_train.isna().sum() #Checking null values. None found.

df_sales_train.max() #Checking outliers and/or mislabeled values
df_sales_train.drop('date', axis=1).to_numpy().max()
df_sales_train.min()
df_sales_train[df_sales_train.item_price<0] 

fig, axes = plt.subplots(ncols=2, figsize=(10,4), dpi=100) #Visualization of outliers
plt.xlim(-100, 3000)
sns.boxplot(y=df_sales_train.item_cnt_day, ax=axes[0])
axes[0].set_ylabel("Item Count per Day")
plt.xlim(df_sales_train.item_price.min(), df_sales_train.item_price.max()*1.1)
sns.boxplot(y=df_sales_train.item_price, ax=axes[1])
axes[1].set_ylabel("Item Price")
plt.suptitle('Outliers Checking', size=16, y=1.1)
plt.tight_layout()
#plt.savefig('Outliers.png', bbox_inches='tight')
plt.show()

df_sales_train = df_sales_train[df_sales_train.item_price<100000] #Dropping outliers
df_sales_train = df_sales_train[df_sales_train.item_cnt_day<1001]
outlier_median = df_sales_train[(df_sales_train.shop_id==32)&(df_sales_train.item_id==2973)&(df_sales_train.date_block_num==4)&( #negative item price data
        df_sales_train.item_price>0)].item_price.median()
df_sales_train.loc[df_sales_train.item_price<0, 'item_price'] = outlier_median #Replace negative item price outlier with median of the same id

dup = df_sales_train[df_sales_train.duplicated(keep=False)] #Checking duplicated data
len(df_sales_train[df_sales_train.duplicated()]) #Found 6 rows duplicated
dup.reset_index().groupby('date_block_num')['date_block_num'].count() #Amount of times that the specific row is repeated
df_sales_train = df_sales_train.drop_duplicates() #Drop duplicates


#Correlations values between features
corr = df_sales_train.corr(method='pearson', min_periods=1)
corr_k = df_sales_train.corr(method='kendall', min_periods=1)
corr_s = df_sales_train.corr(method='spearman', min_periods=1)


#Item Price per Amount of Items plot
g = (sns.jointplot(x="item_cnt_day", y="item_price", data=df_sales_train, height=8,
                   alpha=0.7).set_axis_labels("Amount of Items", "Item Price"))
plt.savefig('PriceAmountItems.png')
plt.show()


#Amount of items per category
amount_items_cat = df_items.groupby(['item_category_id']).count()
amount_items_cat = amount_items_cat.sort_values(by='item_id', ascending=False)
amount_items_cat = amount_items_cat.iloc[0:10].reset_index()
amount_items_cat.drop('item_name', axis=1, inplace=True)
amount_items_cat.columns = ['item_category_id', 'items_amount']
amount_items_cat
#Visualization
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
plt.figure(figsize=(8,4))
sns.barplot(x='item_category_id', y='items_amount', data=amount_items_cat, color='darkorange', alpha=0.7)
plt.ylabel('Amount of items', fontsize=12)
plt.xlabel('Category ID', fontsize=12)
plt.title("Amount of Items per Category", size=14)
plt.savefig("AmountItemsCat.png")
plt.show()



#Creating a 'revenue' column to evaluate income per date
df_sales_train['revenue'] = df_sales_train['item_price']*df_sales_train['item_cnt_day']
total_revenue = df_sales_train.groupby(["date_block_num"])["revenue"].sum()

#New parameters of aggregated data for "item_cnt_day" and "revenue" features
total_sales = df_sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()
total_rev = df_sales_train.groupby(["date_block_num"])["revenue"].sum()
avg_sales = df_sales_train.groupby(["date_block_num"])["item_cnt_day"].mean()
avg_rev = df_sales_train.groupby(["date_block_num"])["revenue"].mean()


#Total sales and total revenue throughout the months
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6}) #plot style
fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
axes[0].plot(total_sales)
axes[0].set_title('Total Sales throughout the Months', size=16)
axes[0].set_xlabel('Labeled Months', size=12) #Since 'date_block_num' is a just a form of labeling months, from 0 to max, "Labeled Months" describes into a better understanding than "date_blocks_num"
axes[0].set_ylabel('Sales', size=12)
axes[1].plot(total_rev, color='darkorange')
axes[1].set_title('Total Revenue throughout the Months', size=16)
axes[1].set_xlabel('Labeled Months', size=12)
axes[1].set_ylabel('Revenue', size=12)
plt.tight_layout()
plt.savefig("TotalSalesRevenueMonths.png")
plt.show()

#Total sales and total revenue throughout the months
sns.set_style("whitegrid") #plot style
fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
axes[0].plot(avg_sales)
axes[0].set_title('Avg. Sales throughout the Months', size=16)
axes[0].set_xlabel('Labeled Months', size=12) #Since 'date_block_num' is a just a form of labeling months, from 0 to max, "Labeled Months" describes into a better understanding than "date_blocks_num"
axes[0].set_ylabel('Sales', size=12)
axes[1].plot(avg_rev, color='darkorange')
axes[1].set_title('Avg. Revenue throughout the Months', size=16)
axes[1].set_xlabel('Labeled Months', size=12)
axes[1].set_ylabel('Revenue', size=12)
plt.tight_layout()
plt.savefig("AvgSalesRevenueMonths.png")
plt.show()

#Moving average and standard deviations 
fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=100)
axes[0].plot(total_sales.rolling(
        window=12, center=False).mean(), label='Moving average');
axes[0].plot(total_sales.rolling(
        window=12, center=False).std(), label='Moving Standard Deviation');
axes[0].legend()
axes[0].set_title('Total Sales throughout the Months', size=16)
axes[0].set_xlabel('Labeled Months', size=12)
axes[0].set_ylabel('Sales', size=12)
axes[1].plot(total_rev.rolling(window=12, center=False).mean(), label='Moving average');
axes[1].plot(total_rev.rolling(window=12, center=False).std(), label='Moving Standard Deviation');
axes[1].legend()
axes[1].set_title('Total Revenue throughout the Months', size=16)
axes[1].set_xlabel('Labeled Months', size=12)
axes[1].set_ylabel('Revenue', size=12)
plt.tight_layout()
plt.savefig("SalesRevenueMoving.png")
plt.show()

#Naive decomposition with amultiplicative model: Y[t] = T[t] * S[t] * e[t]
rcParams['figure.figsize'] = 11, 9
res_sales = sm.tsa.seasonal_decompose(total_sales.values, period=12, model="additive") #Total sales decompose
res_sales_plot = res_sales.plot()
res_sales_plot.savefig('SalesSeasonalDecompose.png', dpi=150)
res_rev = sm.tsa.seasonal_decompose(total_revenue.values, period=12, model="additive") #Total revenue decompose
res_rev_plot = res_rev.plot()
res_rev_plot.savefig('RevSeasonalDecompose.png', dpi=150)


