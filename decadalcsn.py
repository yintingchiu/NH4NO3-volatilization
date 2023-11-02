#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:46:13 2023

@author: yintingchiu
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import seaborn as sns

plt.rcParams['figure.dpi'] = 500
df = pd.read_csv('Research/EPA data/PM2.5_NO3_mass_CA.csv')
df['Sulfate PM2.5 LC'] = df['Sulfate PM2.5 LC'].clip(lower=0)
df['Total Nitrate PM2.5 LC'] = df['Total Nitrate PM2.5 LC'].clip(lower=0)
df['Arithmetic Mean'] = df['Arithmetic Mean'].clip(lower=0)
print(df.columns)
df = df[['Latitude', 'Longitude', 'Date Local','Arithmetic Mean','Address','Average Ambient Temperature',
         'OC1 PM2.5 LC','Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC', 'County Name']]
df['Date Local'] = pd.to_datetime(pd.to_datetime(df['Date Local'], format="%Y-%m-%d"))
df['Year'] = df['Date Local'].dt.year
df['Month'] = df['Date Local'].dt.month
df['Day'] = df['Date Local'].dt.day

fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(6,6))

#fig.suptitle('PM2.5 Trends')
fig.tight_layout(pad=1.5)

df1= df[df['County Name']=='Kern']
df1 = df1[df1['Year']>=2001]
df1 = df1.pivot_table(values=['Arithmetic Mean', 'Average Ambient Temperature', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC'], index=['Year'])
df1 = df1.reset_index()
df1 = df1.interpolate(method='polynomial', order=2)
x = df1['Year'].values
y = df1['Arithmetic Mean'].values
y1 = df1['Sulfate PM2.5 LC'].values
y2 = df1['Total Nitrate PM2.5 LC'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y_Spline = make_interp_spline(x,y)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
X_Y2_Spline = make_interp_spline(x,y2)
Y2_ = X_Y2_Spline(X_)
ax[0,0].plot(X_, Y_, color='black')
ax[0,0].plot(X_, Y1_, color='red')
ax[0,0].plot(X_, Y2_, color='blue')
#ax[0,0].plot(X_, Y3_, color='orange')
a, b = np.polyfit(x, y, 1)
print(a, b)
ax[0,0].plot(x, a*x+b, color='black', linestyle='--')
a, b = np.polyfit(x, y1, 1)
print(a, b)
ax[0,0].plot(x, a*x+b, color='red', alpha=0.3, linestyle='--')
a, b = np.polyfit(x, y2, 1)
print(a, b)
ax[0,0].plot(x, a*x+b, color='blue', alpha=0.3, linestyle='--')
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)
ax[0,0].set_title("Kern County", size=10)


df = pd.read_csv('Research/EPA data/PM2.5_NO3_mass_CA.csv')
df['Sulfate PM2.5 LC'] = df['Sulfate PM2.5 LC'].clip(lower=0)
df['Total Nitrate PM2.5 LC'] = df['Total Nitrate PM2.5 LC'].clip(lower=0)
df['Arithmetic Mean'] = df['Arithmetic Mean'].clip(lower=0)
df['Date Local'] = pd.to_datetime(pd.to_datetime(df['Date Local'], format="%Y-%m-%d"))
df['Year'] = df['Date Local'].dt.year
df['Month'] = df['Date Local'].dt.month
df['Day'] = df['Date Local'].dt.day
df2 = df[df['County Name']=='Tulare']
df2 = df2.pivot_table(values=['Arithmetic Mean', 'Average Ambient Temperature', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC'], index=['Year'])
df2 = df2.reset_index()
df2 = df2.dropna()
x = df2['Year'].values
y = df2['Arithmetic Mean'].values
y1 = df2['Sulfate PM2.5 LC'].values
y2 = df2['Total Nitrate PM2.5 LC'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y_Spline = make_interp_spline(x,y)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
X_Y2_Spline = make_interp_spline(x,y2)
Y2_ = X_Y2_Spline(X_)
ax[1,0].plot(X_, Y_, color='black')
ax[1,0].plot(X_, Y1_, color='red')
ax[1,0].plot(X_, Y2_, color='blue')
a, b = np.polyfit(x, y, 1)
print(a, b)
ax[1,0].plot(x, a*x+b, color='black', linestyle='--')
a, b = np.polyfit(x, y1, 1)
print(a, b)
ax[1,0].plot(x, a*x+b, color='red', alpha=0.3, linestyle='--')
a, b = np.polyfit(x, y2, 1)
print(a,b)
ax[1,0].plot(x, a*x+b, color='blue', alpha=0.3, linestyle='--')
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)
ax[1,0].set_title("Tulare County", size=10)


df3 = df[df['County Name']=='Fresno']
df3 = df3[df3['Year']>=2001]
df3 = df3.pivot_table(values=['Arithmetic Mean', 'Average Ambient Temperature', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC'], index=['Year'  ])
df3 = df3.reset_index()
df3 = df3.dropna()
x = df3['Year'].values
y = df3['Arithmetic Mean'].values
y1 = df3['Sulfate PM2.5 LC'].values
y2 = df3['Total Nitrate PM2.5 LC'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y_Spline = make_interp_spline(x,y)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
X_Y2_Spline = make_interp_spline(x,y2)
Y2_ = X_Y2_Spline(X_)
ax[0,1].plot(X_, Y_, color='black')
ax[0,1].plot(X_, Y1_, color='red')
ax[0,1].plot(X_, Y2_, color='blue')
a, b = np.polyfit(x, y, 1)
print(a, b)
ax[0,1].plot(x, a*x+b, color='black', linestyle='--')
a, b = np.polyfit(x, y1, 1)
print(a,b)
ax[0,1].plot(x, a*x+b, color='red', alpha=0.3, linestyle='--')
a, b = np.polyfit(x, y2, 1)
print(a,b)
ax[0,1].plot(x, a*x+b, color='blue', alpha=0.3, linestyle='--')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].set_title("Fresno County", size=10)

df4 = df[df['County Name']=='Los Angeles']
df4 = df4.pivot_table(values=['Arithmetic Mean', 'Average Ambient Temperature', 'Sulfate PM2.5 LC', 'Total Nitrate PM2.5 LC'], index=['Year'  ])
df4 = df4.reset_index()
df4 = df4.dropna()
x = df4['Year'].values
y = df4['Arithmetic Mean'].values
y1 = df4['Sulfate PM2.5 LC'].values
y2 = df4['Total Nitrate PM2.5 LC'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y_Spline = make_interp_spline(x,y)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
X_Y2_Spline = make_interp_spline(x,y2)
Y2_ = X_Y2_Spline(X_)
ax[1,1].plot(X_, Y_, color='black')
ax[1,1].plot(X_, Y1_, color='red')
ax[1,1].plot(X_, Y2_, color='blue')
a, b = np.polyfit(x, y, 1)
print(a)
ax[1,1].plot(x, a*x+b, color='black', linestyle='--')
a, b = np.polyfit(x, y1, 1)
print(a)
ax[1,1].plot(x, a*x+b, color='red', alpha=0.3, linestyle='--')
a, b = np.polyfit(x, y2, 1)
print(a)
ax[1,1].plot(x, a*x+b, color='blue', alpha=0.3, linestyle='--')
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
ax[1,1].set_title("Los Angeles County", size=10)





