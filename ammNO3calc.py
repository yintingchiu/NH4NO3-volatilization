#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:27:28 2023

@author: yintingchiu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#file with hourly temperature values
site = pd.read_csv('Research/Nitrate/Nitrate Loss py/hourly_t_kern.csv')
#convert temperature to kelvin units
site['sfc_temp'] = site['Sample Measurement']
site['sfc_temp'] = ((site['Sample Measurement']-32) *5/9) + 273
site = site[['Latitude', 'Longitude', 'Date Local','Time Local','State Name', 'County Name','sfc_temp']]
#convert date to datetime format, Year Month & Day
site['Date Local'] = pd.to_datetime(pd.to_datetime(site['Date Local'], format="%Y-%m-%d"))
site['Year'] = site['Date Local'].dt.year
site['Month'] = site['Date Local'].dt.month
site['Day'] = site['Date Local'].dt.day
#calculate dissocation constant from each hourly value
site['K'] = ((site['sfc_temp'])**(-6.025))*np.exp(118.87)*(np.exp(-24084/((site['sfc_temp']))))

#file with hourly RH values
rh = pd.read_csv('Research/EPA data/hourly_rh_kern.csv')
rh = rh[['Latitude', 'Longitude', 'Date Local', 'Time Local','Sample Measurement']]
rh = rh.rename(columns={'Sample Measurement':'RH'})
#change datetime to Year Month & day
rh['Date Local'] = pd.to_datetime(pd.to_datetime(rh['Date Local'], format="%Y-%m-%d"))
rh['Year'] = rh['Date Local'].dt.year
rh['Month'] = rh['Date Local'].dt.month
rh['Day'] = rh['Date Local'].dt.day
rh = rh.drop(columns='Date Local')
#label seasons
rh['Season'] = rh['Month']%12 // 3 + 1
rh['Season'] = rh['Season'].replace({1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'})

#take median RH value for each Year month day and hour (without lat/long because sites dont perfectly match)
rh = rh.pivot_table(index=['Year', 'Month', 'Day', 'Time Local', 'Season'], values='RH', aggfunc=np.median)
rh = rh.reset_index()

#merge hourly temperature and rh
site = pd.merge(site, rh, on=['Year','Month','Day', 'Time Local'])
site['sfc_temp'] = site['sfc_temp'].astype(float)
#calculate deliquescence RH
site['DRH%'] =  62*np.exp(824.3908017*((1/(site['sfc_temp']))-(1/298)))
#calculate factor to multiply K values for conditions above DRH
site['ln P1'] = -135.94 + (8763/site['sfc_temp']) + (19.12*np.log(site['sfc_temp']))
site['ln P2'] = -122.65 + (9969/site['sfc_temp']) + (16.22*np.log(site['sfc_temp']))
site['ln P3'] = -182.61 + (13875/site['sfc_temp']) + (24.46*np.log(site['sfc_temp']))
site['P1'] = np.exp(site['ln P1'])
site['P2'] = np.exp(site['ln P2'])
site['P3'] = np.exp(site['ln P3'])
site['K* factor'] = (site['P1'] - (site['P2']*(1-(site['RH']/100)))+(site['P3']*(1-(site['RH']/100))**2))*(1-(site['RH']/100)**1.75)
#when K is above the DRH, multiply by K*
site['K'] = np.where(site['RH'] > site['DRH%'], site['K']*site['K* factor'], site['K'])

#square root K
site['K'] = site['K'].clip(lower=0)
site['sK'] = np.sqrt(site['K'])
#sum sqrtK for a day 
site = site.pivot_table(index=['Date Local','State Name',
                               'County Name','Year', 'Month', 'Day','Season'], values=['sfc_temp', 'sK'],
                        aggfunc={'sfc_temp':np.mean, 'sK':np.sum}).reset_index()
#volatilized nitrate
site['NO3_v'] = (745.7/site['sfc_temp'])*(site['sK'])*1/24
site = site.pivot_table(index=['Date Local', 'State Name', 'County Name', 'Year', 'Month', 'Day', 'Season'], values=['NO3_v', 'sfc_temp'])
site = site.reset_index()
#file from EPA CSN generated 
nitrate = pd.read_csv('Research/Regression Model/PM2.5_NO3_mass_CA.csv')
nitrate['Date Local'] = pd.to_datetime(pd.to_datetime(nitrate['Date Local']))
nitrate['Year'] = nitrate['Date Local'].dt.year
nitrate['Month'] = nitrate['Date Local'].dt.month
nitrate['Day'] = nitrate['Date Local'].dt.day
nitrate = nitrate[nitrate['County Name']=='Kern']
#relevant columns
nitrate = nitrate[['Date Local','Arithmetic Mean','Method Name','State Name', 'County Name','Address',
                   'Average Ambient Temperature','Total Nitrate PM2.5 LC', 'Year', 'Month','Day',
                   'Latitude', 'Longitude']]
#make sure nitrate values are not negative
nitrate['Total Nitrate PM2.5 LC'] = nitrate['Total Nitrate PM2.5 LC'].clip(lower=0)

#Make sure PM2.5 values are not negative
nitrate = nitrate.rename(columns={'Arithmetic Mean':'PM2.5'})
nitrate['PM2.5'] = nitrate['PM2.5'].clip(lower=0)

#since nitrate is measured every 3 days, we take the average temperature and PM2.5 for those 3 days
nitrate = nitrate.pivot_table(index=['State Name', 'County Name', 'Year', 'Month', 'Day'],
                              values=['PM2.5', 'Average Ambient Temperature', 'Total Nitrate PM2.5 LC' ]).reset_index()
nitrate = nitrate.dropna()

#dont need lat long anymore, average for whole county
site = pd.merge(site, nitrate, on=['State Name', 'County Name','Year', 'Month','Day'])
#convert t back to units in C
site['sfc_temp'] = site['sfc_temp'] - 273.15
#make sure NO3v does not exceed total nitrate PM2.5 LC that is measured from nylon filters
site.loc[site['NO3_v']> site['Total Nitrate PM2.5 LC'], 'NO3_v'] = site['Total Nitrate PM2.5 LC']

#calculate PM2.5 + NO3_v for total PM2.5
site['Total PM2.5'] = site['PM2.5'] + site['NO3_v']
#fraction of volatilized nitrate in PM2.5
site['NO3_v/PM2.5'] = (site['NO3_v']*100/site['Total PM2.5'])
#fraction of volatilized nitrate in nitrate
site['NO3_v/NO3'] = site['NO3_v']*100/ site['Total Nitrate PM2.5 LC']

