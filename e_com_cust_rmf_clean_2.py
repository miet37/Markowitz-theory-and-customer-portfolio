# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

import random
number_list = [111, 222, 333, 444, 555] # random item from list
print(random.choice(number_list)) # Output 222

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pd.options.display.float_format = '{:,.4f}'.format
#pd.io.formats.format.IntArrayFormatter

def p25 (x): 
    return np.percentile(x, q=25)
def p50 (x): 
    return np.percentile(x, q=50)
def p75 (x): 
    return np.percentile(x, q=75)
def p90 (x): 
    return np.percentile(x, q=90)
def p95 (x): 
    return np.percentile(x, q=95)
def p99 (x): 
    return np.percentile(x, q=99)

xls_file = 'e-com_dataset_mp_c2.xlsx'
print('\nreading order_items')
dfxx = pd.read_excel(xls_file,sheet_name="order_items")
dfxx.columns

# for datetime processing
dfxx['year'] = dfxx['order_date'].dt.isocalendar().year
dfxx['month'] = dfxx['order_date'].dt.month
dfxx['week'] = dfxx['order_date'].dt.isocalendar().week
dfxx['weekday'] = dfxx['order_date'].dt.isocalendar().day
dfxx['date_dt'] = dfxx['order_date'].dt.date
dfxx['year_week'] = dfxx.apply(lambda r: 'y'+str(r.year)+'-'+str(r.week).rjust(2, "0"), axis=1)
dfxx['value_row'] = dfxx.apply(lambda r: round(r.price*r.quantity,2), axis=1)

# Start year-week basis analysis
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
print(dfxx.columns)

dfx = pd.read_excel(xls_file,sheet_name="orders")

# frignt is on order level
# distribution freight costs to order lines in order to include to line value
dfxx_fr_share = dfxx.groupby('order_id')['product_id'].count().reset_index()
dfx = dfx.set_index('order_id').join(dfxx_fr_share.set_index('order_id'))
dfx.info()
dfx.dropna(inplace=True)
dfx['fr_cost_per_line'] = dfx.apply(lambda x: x.freight_cost/x.product_id,axis=1 )

dfx['xx'] = dfx['product_id']*dfx['fr_cost_per_line']
dfx[['freight_cost', 'product_id', 'fr_cost_per_line','xx']].sum()
dfx['product_id'].describe()
dfx.columns

dfxx = dfxx.set_index('order_id',drop=False).join(dfx['fr_cost_per_line'])
dfxx.columns

dfxx['value'] = dfxx['value_row'] + dfxx['fr_cost_per_line']
print(dfxx[['fr_cost_per_line', 'value_row','value']].sum())

dfxx['value'].describe()
dfxx['value_log10'] = np.log10(dfxx['value']+1)
dfxx['value_log10'].describe()

plt.scatter(x=dfxx.date_dt,y=dfxx.value,s=2)
plt.scatter(x=dfxx.date_dt,y=dfxx.value_log10,s=2, c='red')





# ------------>
print('# devide to test train dataset')
print(dfxx.groupby('year')['order_id'].count())

# --------- data for training --------->

dfx_train = dfxx[dfxx.year == 2017]
print(dfx_train.columns)

print('#category_main')
dfx_rmf_x = dfx_train.groupby(['category_main']).agg(
    {'order_id': 'count',
    'value': [sum, np.median, np.mean],
    'price': [p25, np.median, p75, p90, np.mean],
    'quantity': [sum, np.median, np.mean],
    'product_id': 'nunique',
    'customer_id': 'nunique'
    }).reset_index()
dfx_rmf_x.to_excel('xx.xlsx')
print(dfx_rmf_x)


# weeks
dfx_rmf = dfx_train.groupby(['customer_id'])['year_week'].nunique().reset_index()
dfx_rmf.columns = ['customer_id', 'year_week_nunique']
dfx_rmf.plot(title='year_week_nunique')

# orders
dfx_rmf_x = dfx_train.groupby(['customer_id'])['order_id'].nunique().reset_index()
dfx_rmf_x['order_id_nunique_log10'] = np.log10(dfx_rmf_x['order_id'])
dfx_rmf_x.columns = ['customer_id', 'order_id_nunique', 'order_id_nunique_log10']
dfx_rmf_x['order_id_nunique_out'] = dfx_rmf_x.apply(lambda x: 100 if x.order_id_nunique >= 100 else x.order_id_nunique, axis=1)
dfx_rmf_x['order_id_nunique'].plot(title='order_id_nunique')
dfx_rmf_x['order_id_nunique_log10'].plot(title='order_id_nunique_log10')
dfx_rmf = dfx_rmf.set_index('customer_id',drop=False).join(dfx_rmf_x.set_index('customer_id'))


# frequency
dfx_rmf_x = dfx_train.groupby(['customer_id'])['order_id'].count().reset_index()
dfx_rmf_x['frequency_log10'] = np.log10(dfx_rmf_x['order_id'])
dfx_rmf_x.columns = ['customer_id', 'frequency','frequency_log10']
dfx_rmf_x['frequency_out'] = dfx_rmf_x.apply(lambda x: 400 if x.frequency >= 400 else x.frequency, axis=1)
dfx_rmf_x['frequency'].plot(title='frequency')
dfx_rmf_x['frequency_log10'].plot(title='frequency_log10')
dfx_rmf = dfx_rmf.join(dfx_rmf_x.set_index('customer_id'))

# recency
dfx_rmf_x = dfx_train.groupby(['customer_id'])['week'].max().reset_index()
dfx_rmf_x.columns = ['customer_id', 'recency']
dfx_rmf_x['recency_out'] = dfx_rmf_x.apply(lambda x: 26 if x.recency <= 26 else x.recency, axis=1)
plt.scatter( x=range(len(dfx_rmf_x)), y=(dfx_rmf_x['recency']), s=2 )
dfx_rmf = dfx_rmf.join(dfx_rmf_x.set_index('customer_id'))

# recency - first purchase
dfx_rmf_x = dfx_train.groupby(['customer_id'])['week'].min().reset_index()
dfx_rmf_x.columns = ['customer_id', 'first_purchase']
dfx_rmf_x.plot(title='first_purchase')
dfx_rmf = dfx_rmf.join(dfx_rmf_x.set_index('customer_id'))

# recency - freq by time between start-end purchase
dfx_rmf['freq_time'] = dfx_rmf['frequency']/(1+(dfx_rmf['recency']-dfx_rmf['first_purchase']))
dfx_rmf['freq_time'].plot(title='freq_time')

# monetary
dfx_rmf_x = dfx_train.groupby(['customer_id'])['value'].sum().reset_index()
#dfx_rmf_x['value'] = dfx_rmf_x.apply(lambda x: 2000000 if x.value >= 2000000 else x.value, axis=1)
dfx_rmf_x['value_log10'] = np.log10(dfx_rmf_x['value'])
dfx_rmf_x.columns = ['customer_id', 'value',  'value_log10']
dfx_rmf_x['value'].plot(title='value')
dfx_rmf_x['value_log10'].plot(title='value_log10')
dfx_rmf = dfx_rmf.join(dfx_rmf_x.set_index('customer_id'))

# avg price
dfx_rmf_x = dfx_train.groupby(['customer_id'])['price'].median().reset_index()
#dfx_rmf_x['value'] = dfx_rmf_x.apply(lambda x: 2000000 if x.value >= 2000000 else x.value, axis=1)
dfx_rmf_x['value_log10'] = np.log10(dfx_rmf_x['price'])
dfx_rmf_x.columns = ['customer_id', 'price_median',  'price_log10']
dfx_rmf_x['price_median'].plot(title='price_median')
dfx_rmf_x['price_log10'].plot(title='price_log10')
dfx_rmf = dfx_rmf.join(dfx_rmf_x.set_index('customer_id'))




print(dfx_rmf.columns)


print('# drop same columns')
dfx_rmf.drop(columns=['customer_id'],inplace=True)
#dfx_rmf.drop(columns=['year_week_nunique'],inplace=True)
#dfx_rmf.drop(columns=['order_id_nunique_log10'],inplace=True)
#dfx_rmf.drop(columns=['frequency_log10'],inplace=True)
#dfx_rmf.drop(columns=['frequency'],inplace=True)
#dfx_rmf.drop(columns=['first_purchase'],inplace=True)
#dfx_rmf.drop(columns=['value'],inplace=True)


dfx_rmf.describe()
dfx_rmf.to_excel('dfx_rmf.xlsx')


# Seaborn visualization library
# Create the default pairplot
sns.pairplot(dfx_rmf)

# feature selection
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# dfx_rmf1 = pd.DataFrame( sel.fit_transform(dfx_rmf) )

# corelations
dfx_rmf.corr().to_excel('dfx_rmf_corr.xlsx')

# spearmanr
def pair_correlation(dfx_rmf):
    import scipy
    df = pd.DataFrame()
    feat1s = []
    feat2s = []
    corrs = []
    p_values = []
    for feat1 in dfx_rmf.columns:
        for feat2 in dfx_rmf.columns:
            if feat1 != feat2:
                feat1s.append(feat1)
                feat2s.append(feat2)
                corr, p_value = scipy.stats.spearmanr(dfx_rmf[feat1], dfx_rmf[feat2])
                corrs.append(corr)
                p_values.append(p_value)
    
    df['Feature_1'] = feat1s
    df['Feature_2'] = feat2s
    df['Correlation'] = corrs
    df['p_value'] = p_values
    return df

print(pair_correlation(dfx_rmf))

# pearson
#import scipy
#corr, p_values = scipy.stats.pearsonr(dfx_rmf['frequency'], dfx_rmf['value_log10'])
#print(round(corr,4), round(p_values,4))


# from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
dfx_rmf_scaled = pd.DataFrame(scaler.fit_transform(dfx_rmf), columns=dfx_rmf.columns) # normalisation

#scaler = StandardScaler()
#dfx_rmf_scaled = pd.DataFrame(scaler.fit_transform(dfx_rmf_scaled), columns=dfx_rmf_scaled.columns) # normalisation



#scaler = RobustScaler()
#dfx_rmf_scaled = pd.DataFrame(scaler.fit_transform(dfx_rmf), columns=dfx_rmf.columns) # normalisation

dfx_rmf_scaled.to_excel('dfx_rmf_scaled.xlsx')
dfx_rmf_scaled.columns


errors = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(dfx_rmf_scaled)
    errors.append(model.inertia_)
    
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('Error of Cluster')
sns.pointplot(x=list(range(1, 11)), y=errors)
plt.show()


col_for_model = [
#    'year_week_nunique', 
    'order_id_nunique',
    'order_id_nunique_log10',
#    'order_id_nunique_out', 
#    'frequency',
    'frequency_log10',
#    'frequency_out', 
#    'recency',
    'recency_out',
#    'first_purchase', 
    'freq_time', 
#    'value', 
    'value_log10'
    ]

model = KMeans(n_clusters = 5, random_state=42)
model.fit(dfx_rmf_scaled[col_for_model])
dfx_rmf_scaled['cLabel_k_means'] = model.labels_
dfx_rmf_scaled['ccolor'] = dfx_rmf_scaled['cLabel_k_means'].map({0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y'})

# dimentionality reduction
pca = PCA()
Xt = pca.fit_transform(dfx_rmf_scaled[col_for_model])
# Clusters visualisation
plot = plt.scatter(Xt[:,0], Xt[:,1], c=dfx_rmf_scaled['ccolor'])
plt.show()

# sum
dfx_rmf.sum()
dfx_rmf['cl'] = model.labels_

from pprint import pprint
pprint(dfx_rmf.columns.to_list())
pprint(dfx_train.columns.to_list())

# transfering cl to orginal data
dfx_train_cl = dfx_train.set_index('customer_id',drop=False).join(dfx_rmf['cl'])

# calculating return per week
dfx_rmf_cl_week = dfx_train_cl.groupby(['cl','year_week'])['value'].sum().reset_index()
dfx_rmf_cl_week['return'] = dfx_rmf_cl_week['value'].diff()
dfx_rmf_cl_week.groupby('cl')['return'].sum()

dfx_rmf.groupby('cl')['value'].sum()
dfx_rmf[dfx_rmf['cl']==0]['value'].plot()
dfx_rmf[dfx_rmf['cl']==1]['value'].plot()
dfx_rmf[dfx_rmf['cl']==2]['value'].plot()
dfx_rmf[dfx_rmf['cl']==3]['value'].plot()
dfx_rmf[dfx_rmf['cl']==4]['value'].plot()

dfx_rmf.columns
# set up cluster to the data
data_cl = dfx_train.set_index('customer_id').join(dfx_rmf['cl'])
print(data_cl.columns)

data_seg = pd.pivot(data=data_cl.groupby(['date_dt','cl'])['value'].sum().reset_index(),
                    index = 'date_dt',
                    columns = 'cl',
                    values='value')
data_seg.columns


def tab_plot(table2):
  plt.figure(figsize=(14, 7))
  for c in table2.columns.values:
      plt.plot(table2.index, table2[c], lw=1, label=c)
  plt.legend(loc='upper left', fontsize=12)
  plt.ylabel('price in $')
  plt.xlabel('timestep')    

tab_plot(data_seg)    

data_seg.isnull().sum()
data_seg.fillna(method='ffill', inplace=True )
data_seg.fillna(method='bfill', inplace=True )

#data_seg.iloc[:,4].fillna(data_seg.iloc[:,4].mean(), inplace=True )    
    
#markowitz

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record    
    
returns = data_seg.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 2500
risk_free_rate = 0.02    
    
results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)

max_sharpe_idx = np.argmax(results[2])
sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=data_seg.columns,columns=['allocation'])
max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation.T    
    
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=data_seg.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=data_seg.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print( "-"*80)
    print( "Maximum Sharpe Ratio Portfolio Allocation\n")
    print( "Annualised Return:", round(rp,2))
    print( "Annualised Volatility:", round(sdp,2))
    print( "\n")
    print( max_sharpe_allocation)
    print( "-"*80)
    print( "Minimum Volatility Portfolio Allocation\n")
    print( "Annualised Return:", round(rp_min,2))
    print( "Annualised Volatility:", round(sdp_min,2))
    print( "\n")
    print( min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)    
    
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)    

sum_tot = sum([data_seg.iloc[:,i].sum() for i in data_seg.columns])
print(sum_tot)

seg_share = 0.0
for i in data_seg.columns:
    seg_share = round(100*data_seg.iloc[:,i].sum() / sum_tot ,2)
    print(i, seg_share)
    
    
    
