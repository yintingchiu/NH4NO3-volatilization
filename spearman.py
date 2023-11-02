#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:58:22 2023

@author: yintingchiu
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import seaborn as sns

plt.rcParams['figure.dpi'] = 500
df = pd.read_csv('Research/EPA data/PM2.5_NO3_mass_CA.csv')
df['Latitude'] = df['Latitude'].map('{:.4f}'.format)
df['Longitude'] = df['Longitude'].map('{:.4f}'.format)
print(df.columns)
df = df[['State Name', 'County Name', 'Date Local','Arithmetic Mean','Average Ambient Temperature','Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df['Date Local'] = pd.to_datetime(pd.to_datetime(df['Date Local'], format="%Y-%m-%d"))
df['Year'] = df['Date Local'].dt.year
df['Month'] = df['Date Local'].dt.month
df['Day'] = df['Date Local'].dt.day
df['Sulfate PM2.5 LC'] = df['Sulfate PM2.5 LC'].clip(lower=0)
df['Total Nitrate PM2.5 LC'] = df['Total Nitrate PM2.5 LC'].clip(lower=0)
df['Arithmetic Mean'] = df['Arithmetic Mean'].clip(lower=0)
df = df.pivot_table(index=['State Name','County Name','Year', 'Month', 'Day'], values=['Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC', 'Arithmetic Mean', 'Average Ambient Temperature']).reset_index()

rh = pd.read_csv('Research/EPA data/rh_epa_daily.csv')
rh = rh.rename(columns={'Arithmetic Mean':'RH'})
rh['Date Local'] = pd.to_datetime(pd.to_datetime(rh['Date Local'], format="%Y-%m-%d"))
rh['Year'] = rh['Date Local'].dt.year
rh['Month'] = rh['Date Local'].dt.month
rh['Day'] = rh['Date Local'].dt.day
rh = rh.pivot_table(index=['State Name', 'County Name', 'Year','Month','Day'], values='RH', aggfunc=np.median).reset_index()


no3v = pd.read_csv('Research/Nitrate/NO3v3.csv')
print(no3v.columns)
no3v['Date Local'] = pd.to_datetime(pd.to_datetime(no3v['Date Local'], format="%Y-%m-%d"))
no3v['Year'] = no3v['Date Local'].dt.year
no3v['Month'] = no3v['Date Local'].dt.month
no3v['Day'] = no3v['Date Local'].dt.day  
no3v['NO3'] = no3v['NO3'].clip(lower=0)
no3v['NO3'] = no3v['NO3_v'].clip(lower=0)
no3v = no3v[['State Name','County Name','Year', 'Month', 'Day','NO3_v']]
ca = pd.merge(no3v, df,how='right',on=['State Name','County Name','Year', 'Month', 'Day'])
ca = pd.merge(ca, rh,how='left',on=['State Name','County Name','Year', 'Month', 'Day'])

il = pd.read_csv('Research/EPA data/PM2.5_NO3_mass_IL.csv')
il['Latitude'] = il['Latitude'].map('{:.4f}'.format)
il['Longitude'] = il['Longitude'].map('{:.4f}'.format)
il = il[['State Name','County Name', 'Date Local','Arithmetic Mean','Address','Average Ambient Temperature','Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
il['Date Local'] = pd.to_datetime(pd.to_datetime(il['Date Local'], format="%Y-%m-%d"))
il['Year'] = il['Date Local'].dt.year
il['Month'] = il['Date Local'].dt.month
il['Day'] = il['Date Local'].dt.day
df['Sulfate PM2.5 LC'] = df['Sulfate PM2.5 LC'].clip(lower=0)
df['Total Nitrate PM2.5 LC'] = df['Total Nitrate PM2.5 LC'].clip(lower=0)
df['Arithmetic Mean'] = df['Arithmetic Mean'].clip(lower=0)
il = il.pivot_table(index=['State Name','County Name','Year', 'Month', 'Day'], values=['Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC', 'Arithmetic Mean', 'Average Ambient Temperature']).reset_index()
il = pd.merge(no3v, il,how='right', on=['State Name','County Name', 'Year', 'Month', 'Day'])
il = pd.merge(il, rh,how='left',on=['State Name','County Name','Year', 'Month', 'Day'])

md = pd.read_csv('Research/EPA data/PM2.5_NO3_mass_MD.csv')
md['Latitude'] = md['Latitude'].map('{:.4f}'.format)
md['Longitude'] = md['Longitude'].map('{:.4f}'.format)
md = md[['State Name','County Name', 'Date Local','Arithmetic Mean','Address','Average Ambient Temperature','Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
md['Date Local'] = pd.to_datetime(pd.to_datetime(md['Date Local'], format="%Y-%m-%d"))
md['Year'] = md['Date Local'].dt.year
md['Month'] = md['Date Local'].dt.month
md['Day'] = md['Date Local'].dt.day
df['Sulfate PM2.5 LC'] = df['Sulfate PM2.5 LC'].clip(lower=0)
df['Total Nitrate PM2.5 LC'] = df['Total Nitrate PM2.5 LC'].clip(lower=0)
df['Arithmetic Mean'] = df['Arithmetic Mean'].clip(lower=0)
md = md.pivot_table(index=['State Name','County Name', 'Year', 'Month', 'Day'], values=['Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC', 'Arithmetic Mean', 'Average Ambient Temperature']).reset_index()
md = pd.merge(no3v, md,how='right', on=[ 'State Name','County Name', 'Year', 'Month', 'Day'])
md = pd.merge(md, rh,how='left',on=['State Name','County Name','Year', 'Month', 'Day'])

df1= ca[ca['County Name']=='Kern']
df2= ca[ca['County Name']=='Tulare']
df3= ca[ca['County Name']=='Fresno']
df4= ca[ca['County Name']=='Los Angeles']
df5= il[il['County Name']=='Cook']
df6= md[md['County Name']=='Baltimore']

df1 = df1[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df2 = df2[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df3 = df3[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df4 = df4[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df5 = df5[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]
df6 = df6[['Year', 'NO3_v','Arithmetic Mean', 'Average Ambient Temperature', 'RH', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC']]

fig, ax = plt.subplots()
##take last 6 columns and plot correlation
dfs = [df1, df2, df3, df4, df5, df6]
for df in dfs:
    df.columns = ['Year', '\u0394NO3','PM2.5', 'Temp', 'RH', 'SO4', 'NO3']

df1 = df1.iloc[:,-6:]
corr = df1.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Kern County')
plt.show()

fig, ax = plt.subplots()
df2 = df2.iloc[:,-6:]
corr = df2.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Tulare County')
plt.show()

fig, ax = plt.subplots()
df3 = df3.iloc[:,-6:]
corr = df3.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Fresno County')
plt.show()

fig, ax = plt.subplots()
df4 = df4.iloc[:,-6:]
corr = df4.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()

sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Los Angeles County')
plt.show()

fig, ax = plt.subplots()
df5 = df5.iloc[:,-6:]
corr = df5.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Cook County')
plt.show()

fig, ax = plt.subplots()
df6 = df6.iloc[:,-6:]
corr = df6.corr(method = 'spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Baltimore County')
plt.show()
