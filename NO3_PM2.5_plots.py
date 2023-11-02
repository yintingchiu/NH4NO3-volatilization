#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:26:33 2023

@author: yintingchiu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline

plt.rc_context({'axes.edgecolor':'grey', 'xtick.color':'grey', 'ytick.color':'grey', 'figure.facecolor':'grey'})
plt.rcParams['figure.dpi'] = 700

site = pd.read_csv('Research/Nitrate/Nitrate Loss py/hourly_t_losangeles.csv')
site['sfc_temp'] = site['Sample Measurement']
site['sfc_temp'] = ((site['Sample Measurement']-32) *5/9) + 273
#print(site.loc[:,'sfc_temp'].mean())
site = site[['Latitude', 'Longitude', 'Date Local','Time Local','State Name', 'County Name','sfc_temp']]
site['Date Local'] = pd.to_datetime(pd.to_datetime(site['Date Local'], format="%Y-%m-%d"))
site['Year'] = site['Date Local'].dt.year
site['Month'] = site['Date Local'].dt.month
site['Day'] = site['Date Local'].dt.day
site['K'] = ((site['sfc_temp'])**(-6.025))*np.exp(118.87)*(np.exp(-24084/((site['sfc_temp']))))
rh = pd.read_csv('Research/EPA data/hourly_rh_losangeles.csv')
rh = rh[['Latitude', 'Longitude', 'Date Local', 'Time Local','Sample Measurement']]
rh = rh.rename(columns={'Sample Measurement':'RH'})
rh['Date Local'] = pd.to_datetime(pd.to_datetime(rh['Date Local'], format="%Y-%m-%d"))
rh['Year'] = rh['Date Local'].dt.year
rh['Month'] = rh['Date Local'].dt.month
rh['Day'] = rh['Date Local'].dt.day
rh = rh.drop(columns='Date Local')
rh['Season'] = rh['Month']%12 // 3 + 1
rh['Season'] = rh['Season'].replace({1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'})

rh = rh.pivot_table(index=['Year', 'Month', 'Day', 'Time Local', 'Season'], values='RH', aggfunc=np.median)
rh = rh.reset_index()
site = pd.merge(site, rh, on=['Year','Month','Day', 'Time Local'])
site['sfc_temp'] = site['sfc_temp'].astype(float)
site['DRH%'] = 61.8*np.exp((16254.84/8.3145)*(4.298*((1/site['sfc_temp'])-(1/298))-((-3.623e-2)*np.log(site['sfc_temp']/298))-(7.853e-5*(site['sfc_temp']-298))))
#site['DRH%'] =  62*np.exp(824.3908017*((1/(site['sfc_temp']))-(1/298)))
site['ln P1'] = -135.94 + (8763/site['sfc_temp']) + (19.12*np.log(site['sfc_temp']))
site['ln P2'] = -122.65 + (9969/site['sfc_temp']) + (16.22*np.log(site['sfc_temp']))
site['ln P3'] = -182.61 + (13875/site['sfc_temp']) + (24.46*np.log(site['sfc_temp']))
site['P1'] = np.exp(site['ln P1'])
site['P2'] = np.exp(site['ln P2'])
site['P3'] = np.exp(site['ln P3'])
site['K* factor'] = (site['P1'] - (site['P2']*(1-(site['RH']/100)))+(site['P3']*(1-(site['RH']/100))**2))*(1-(site['RH']/100)**1.75)
#when K is above the DRH
site['K'] = np.where(site['RH'] > site['DRH%'], site['K']*site['K* factor'], site['K'])


#square root K
site['K'] = site['K'].clip(lower=0)
site['sK'] = np.sqrt(site['K'])


site = site.pivot_table(index=['Date Local','State Name',
                               'County Name','Year', 'Month', 'Day','Season'], values=['sfc_temp', 'sK'],
                        aggfunc={'sfc_temp':np.mean, 'sK':np.sum}).reset_index()
#volatilized nitrate
site['NO3_v'] = (745.7/site['sfc_temp'])*(site['sK'])*1/24
site = site.pivot_table(index=['Date Local', 'State Name', 'County Name', 'Year', 'Month', 'Day', 'Season'], values=['NO3_v', 'sfc_temp'])
site = site.reset_index()
nitrate = pd.read_csv('Research/Regression Model/PM2.5_NO3_mass_CA.csv')
nitrate['Date Local'] = pd.to_datetime(pd.to_datetime(nitrate['Date Local']))
nitrate['Year'] = nitrate['Date Local'].dt.year
nitrate['Month'] = nitrate['Date Local'].dt.month
nitrate['Day'] = nitrate['Date Local'].dt.day
nitrate = nitrate[nitrate['County Name']=='Los Angeles']
nitrate = nitrate[['Date Local','Arithmetic Mean','Method Name','State Name', 'County Name','Address',
                   'Average Ambient Temperature','Total Nitrate PM2.5 LC', 'Year', 'Month','Day',
                   'Latitude', 'Longitude']]
#nitrate = nitrate.dropna()
nitrate['Total Nitrate PM2.5 LC'] = nitrate['Total Nitrate PM2.5 LC'].clip(lower=0)
##finding value counts
#nitrate['monthyear'] = nitrate['Date Local'].dt.to_period('M')
#nitrate = nitrate[['Date Local', 'Arithmetic Mean', 'monthyear', 'Address']].dropna()
#print(nitrate['Address'].unique())
#values = nitrate['monthyear'].value_counts().reset_index()
#values = values.rename(columns={'monthyear':'counts', 'index':'monthyear'})
#values['monthyear'] = values['monthyear'].astype(str)
#values['monthyear'] = pd.to_datetime(values['monthyear'])
#values['month'] = values['monthyear'].dt.month
#values['year'] = values['monthyear'].dt.year
#values = values.set_index(['year', 'month'])
#values = values[['counts']].unstack('month')
#values.to_excel('Research/Manuscript Drafts/no3_data.xlsx')



###
nitrate = nitrate.rename(columns={'Arithmetic Mean':'PM2.5'})
nitrate = nitrate.pivot_table(index=['State Name', 'County Name', 'Year', 'Month', 'Day'],
                              values=['PM2.5', 'Average Ambient Temperature', 'Total Nitrate PM2.5 LC' ]).reset_index()
nitrate = nitrate.dropna()
site = pd.merge(site, nitrate, on=['State Name', 'County Name','Year', 'Month','Day'])
#dont need lat long anymore, average for whole county
site['sfc_temp'] = site['sfc_temp'] - 273.15
#site = site.dropna()
#site = site[site['Total Nitrate PM2.5 LC']>=0]
site.loc[site['NO3_v']> site['Total Nitrate PM2.5 LC'], 'NO3_v'] = site['Total Nitrate PM2.5 LC']
site['Total PM2.5'] = site['PM2.5'] + site['NO3_v']
site['NO3_v/PM2.5'] = (site['NO3_v']*100/site['Total PM2.5'])
site['NO3_v/NO3'] = site['NO3_v']*100/ site['Total Nitrate PM2.5 LC']

#site.to_csv('Research/Manuscript Drafts/Code for Nitrate_PM/Los Angeles_pm.csv')


#plot PM2.5
fig, ax= plt.subplots()
pmtotal = site.dropna().pivot_table(index='Year', values=['Total PM2.5', 'PM2.5'])
pmtotal = pmtotal.reset_index()
x = pmtotal['Year'].values
y = pmtotal['PM2.5'].values
y1 = pmtotal['Total PM2.5'].values
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
# Plotting the Graph
plt.plot(X_, Y_, color='blue')
ax.fill_between(X_, Y_, Y1_, alpha=0.3, color='blue')
plt.plot(X_, Y1_, color='blue', alpha=0.3)
#ax.fill_between(X_, Y_, Y1_, alpha=0.4, color='green')
a, b = np.polyfit(x, y, 1)
print(a,b)
plt.plot(x, a*x+b, color='blue', linestyle='--')
a, b = np.polyfit(x, y1, 1)
print(a,b)
plt.plot(x, a*x+b, color='blue', alpha=0.3, linestyle='--')
plt.legend(['Reported PM2.5', 'Volatilized Nitrate','Ambient Burden'])
#plt.legend(['Reported PM2.5', 'Estimated Nitrate Volatilization'], loc='lower right')
#plt.tick_params(axis='both', which='both', length=0)
plt.xticks(range(2001,2023,2))
plt.yticks(range(0,35,5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title("Los Angeles County")
plt.xlabel("Year")
plt.ylabel("PM2.5 Concentration ${\mu}g m^{-3}$")
plt.show()

fig, ax= plt.subplots()
fig.text(0.5, 0.04, 'Year', ha='center')
#plot Volatlized NO3/PM2.5

vno3pm = site.dropna().pivot_table(index='Year', values=['NO3_v','Total PM2.5'])
vno3pm = vno3pm.reset_index()
vno3pm['NO3_v/PM2.5'] = vno3pm['NO3_v']/vno3pm['Total PM2.5']
x = vno3pm['Year'].values
y = vno3pm['NO3_v/PM2.5'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y_Spline = make_interp_spline(x,y)
Y_ = X_Y1_Spline(X_)
plt.plot(X_, Y1_, color='indigo')
plt.xticks(range(2001,2023,2))
plt.yticks(range(0,35,5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('%')
plt.title("Percentage NO3 Volatilized from PM2.5 in Kern County")

fig, ax = plt.subplots(2,2)
fig.suptitle('Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')
fig.text(0, 0.25, 'Percentage Mass Volatilized (%)', ha='center', rotation='vertical')
no3vseason = pd.pivot_table(site, index=['Year', 'Season'],values=['NO3_v/PM2.5'])
no3vseason = no3vseason.reset_index()
winter = no3vseason[no3vseason['Season']=='Winter']
spring = no3vseason[no3vseason['Season']=='Spring']
summer = no3vseason[no3vseason['Season']=='Summer']
fall = no3vseason[no3vseason['Season']=='Fall']

x = winter['Year'].values
y1 = winter['NO3_v/PM2.5'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y1_, color='purple')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

x = spring['Year'].values
y1 = spring['NO3_v/PM2.5'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,1].plot(x, a*x+b, color='olive', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y1_, color='olive')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].set_title("Spring")

x = summer['Year'].values
y1 = summer['NO3_v/PM2.5'].values
X_ = np.linspace(x.min(), x.max(), 500)
#pch = pchip(x, y1) 
#xx=np.linspace(x[0], x[-1], 500) 
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,0].plot(x, a*x+b, color='pink', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y1_, color='pink')
#ax[1,0].plot(xx, pch(xx), color='pink')
ax[1, 0].set_title("Summer")
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)
ax[1,0].set_ylim(ymin=0)


x = fall['Year'].values
y1 = fall['NO3_v/PM2.5'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,1].plot(x, a*x+b, color='orange', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y1_, color='orange')
ax[1,1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
plt.setp(ax, yticks=[0,10,20,30, 40], xticks=[2000,2005,2010,2015,2020])

fig, ax = plt.subplots(2,2)
fig.suptitle('Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')
fig.text(0, 0.25, 'PM2.5 Concentration ${\mu}g m^{-3}$', ha='center', rotation='vertical')

pmseason = site.dropna().pivot_table(index=['Year', 'Season'],values=['PM2.5', 'Total PM2.5'])
pmseason = pmseason.reset_index()
winter = pmseason[pmseason['Season']=='Winter']
spring = pmseason[pmseason['Season']=='Spring']
summer = pmseason[pmseason['Season']=='Summer']
fall = pmseason[pmseason['Season']=='Fall']

x = winter['Year'].values
y = winter['PM2.5'].values
y1 = winter['Total PM2.5']
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y_, color='purple')
a, b = np.polyfit(x, y1, 1)
ax[0,0].plot(x, a*x+b, color='indigo', linestyle='--', alpha=0.5)
ax[0,0].fill_between(X_, Y_, Y1_, alpha=0.3, color='indigo')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

x = spring['Year'].values
y = spring['PM2.5'].values
y1 = spring['Total PM2.5']
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[0,1].plot(x, a*x+b, color='green', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y_, color='green')
a, b = np.polyfit(x, y1, 1)
ax[0,1].plot(x, a*x+b, color='olive', linestyle='--', alpha=0.5)
ax[0,1].fill_between(X_, Y_, Y1_, alpha=0.3, color='olive')
ax[0,1].set_title("Spring")
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

x = summer['Year'].values
y = summer['PM2.5'].values
y1 = summer['Total PM2.5']
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[1,0].plot(x, a*x+b, color='red', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y_, color='red')
a, b = np.polyfit(x, y1, 1)
ax[1,0].plot(x, a*x+b, color='pink', linestyle='--', alpha=0.5)
ax[1,0].fill_between(X_, Y_, Y1_, alpha=0.3, color='pink')
ax[1, 0].set_title("Summer")
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

x = fall['Year'].values
y = fall['PM2.5'].values
y1 = fall['Total PM2.5']
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[1,1].plot(x, a*x+b, color='brown', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y_, color='brown')
a, b = np.polyfit(x, y1, 1)
ax[1,1].plot(x, a*x+b, color='orange', linestyle='--', alpha=0.5)
ax[1,1].fill_between(X_, Y_, Y1_, alpha=0.3, color='orange')
ax[1,1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

plt.setp(ax, yticks=[0,20,40,60], xticks=[2000,2005,2010,2015,2020])

#%NO3V/NO3
fig, ax = plt.subplots(2,2)
fig.suptitle('Percentage NO3 Volatilized from NO3 in Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')

no3vseasonp = pd.pivot_table(site, index=['Year', 'Season'],values=['NO3_v/NO3'])
no3vseasonp = no3vseasonp.reset_index()
winter = no3vseasonp[no3vseasonp['Season']=='Winter']
spring = no3vseasonp[no3vseasonp['Season']=='Spring']
summer = no3vseasonp[no3vseasonp['Season']=='Summer']
fall = no3vseasonp[no3vseasonp['Season']=='Fall']

x = winter['Year'].values
y1 = winter['NO3_v/NO3'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y1_, color='purple')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)


x = spring['Year'].values
y1 = spring['NO3_v/NO3'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,1].plot(x, a*x+b, color='olive', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y1_, color='olive')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].set_title("Spring")
x = summer['Year'].values
y1 = summer['NO3_v/NO3'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,0].plot(x, a*x+b, color='pink', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y1_, color='pink')
ax[1, 0].set_title("Summer")
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

x = fall['Year'].values
y1 = fall['NO3_v/NO3'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,1].plot(x, a*x+b, color='orange', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y1_, color='orange')
ax[1, 1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
plt.setp(ax, yticks=[0,25,50,75,100], xticks=[2000,2005,2010,2015,2020])

fig, ax = plt.subplots(2,2)
fig.suptitle('Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')
fig.text(0, 0.25, 'Nitrate Concentration ${\mu}g m^{-3}$', ha='center', rotation='vertical')

nitrate['Season'] = nitrate['Month']%12 // 3 + 1
nitrate['Season'] = nitrate['Season'].replace({1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'})
no3season = pd.pivot_table(nitrate, index=['Year', 'Season'],values=['Total Nitrate PM2.5 LC'])
no3season = no3season.reset_index()
winter = no3season[no3season['Season']=='Winter']
spring = no3season[no3season['Season']=='Spring']
summer = no3season[no3season['Season']=='Summer']
fall = no3season[no3season['Season']=='Fall']

x = winter['Year'].values
y = winter['Total Nitrate PM2.5 LC'].values
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y_, color='purple')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

x = spring['Year'].values
y = spring['Total Nitrate PM2.5 LC'].values
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[0,1].plot(x, a*x+b, color='green', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y_, color='green')
ax[0,1].set_title("Spring")
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

x = summer['Year'].values
y = summer['Total Nitrate PM2.5 LC'].values
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[1,0].plot(x, a*x+b, color='red', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y_, color='red')
ax[1, 0].set_title("Summer")
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

x = fall['Year'].values
y = fall['Total Nitrate PM2.5 LC'].values
X_Y_Spline = make_interp_spline(x,y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
a, b = np.polyfit(x, y, 1)
ax[1,1].plot(x, a*x+b, color='brown', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y_, color='brown')
ax[1,1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

plt.setp(ax, yticks=[0,5,10], xticks=[2000,2005,2010,2015,2020])

#temperature
fig, ax = plt.subplots(2,2)
fig.suptitle('Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')
fig.text(0, 0.4, 'Temperature (˚C)', ha='center', rotation=90)

temp = site.pivot_table(index=['Year', 'Season'], values='sfc_temp')
temp = temp.reset_index()
winter = temp[temp['Season']=='Winter']
spring = temp[temp['Season']=='Spring']
summer = temp[temp['Season']=='Summer']
fall = temp[temp['Season']=='Fall']

x = winter['Year'].values
y1 = winter['sfc_temp'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y1_, color='purple')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)


x = spring['Year'].values
y1 = spring['sfc_temp'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,1].plot(x, a*x+b, color='olive', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y1_, color='olive')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].set_title("Spring")
x = summer['Year'].values
y1 = summer['sfc_temp'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,0].plot(x, a*x+b, color='pink', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y1_, color='pink')
ax[1, 0].set_title("Summer")
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

x = fall['Year'].values
y1 = fall['sfc_temp'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,1].plot(x, a*x+b, color='orange', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y1_, color='orange')
ax[1, 1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
plt.setp(ax, yticks=[0,10,20,30], xticks=[2000,2005,2010,2015,2020])


#nitrate as percent pm2.5
fig, ax = plt.subplots(2,2)
fig.suptitle('Nitrate as % PM2.5 Kern County')
fig.tight_layout(pad=1.5)
fig.text(0.5, 0, 'Year', ha='center')

site['npm25'] = site['Total Nitrate PM2.5 LC']*100/site['PM2.5']
npm25 = site.pivot_table(index=['Year', 'Season'], values='npm25')
npm25 = npm25.reset_index()
winter = npm25[npm25['Season']=='Winter']
spring = npm25[npm25['Season']=='Spring']
summer = npm25[npm25['Season']=='Summer']
fall = npm25[npm25['Season']=='Fall']

x = winter['Year'].values
y1 = winter['npm25'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,0].plot(x, a*x+b, color='purple', linestyle='--', alpha=0.5)
ax[0,0].plot(X_, Y1_, color='purple')
ax[0, 0].set_title("Winter")
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

x = spring['Year'].values
y1 = spring['npm25'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[0,1].plot(x, a*x+b, color='olive', linestyle='--', alpha=0.5)
ax[0,1].plot(X_, Y1_, color='olive')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].set_title("Spring")
x = summer['Year'].values
y1 = summer['npm25'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,0].plot(x, a*x+b, color='pink', linestyle='--', alpha=0.5)
ax[1,0].plot(X_, Y1_, color='pink')
ax[1, 0].set_title("Summer")    
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

x = fall['Year'].values
y1 = fall['npm25'].values
X_ = np.linspace(x.min(), x.max(), 500)
X_Y1_Spline = make_interp_spline(x,y1)
Y1_ = X_Y1_Spline(X_)
a, b = np.polyfit(x, y1, 1)
ax[1,1].plot(x, a*x+b, color='orange', linestyle='--', alpha=0.5)
ax[1,1].plot(X_, Y1_, color='orange')
ax[1, 1].set_title("Fall")
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
plt.setp(ax, yticks=[0,25,50,75,100], xticks=[2000,2005,2010,2015,2020])
