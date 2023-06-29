# NH4NO3-volatilization

This is a project that calculates ammonium nitrate volatilization in PM2.5 filters using data obtained from the USEPA.

## nitrate loss calculation
Hourly temperature and relative humidity data are obtained from the USEPA (https://aqs.epa.gov/aqsweb/airdata/download_files.html).
Alternatively, the ECMWF Reanalysis v5 (ERA5) model (https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) provides hourly surface temperature data and dew point data, which you can calculate RH using the following equations: 
<p align="center">
RH = $100\frac{e_{s,T_d}}{e_{s,T}}$ 
</p>

$T$ = temperature in Celsius at pressure P

$T_{d}$ = dew point temperature in Celsius at pressure P
<p align="center">
$e_s = \frac{\exp(34.494-\frac{4924.99}{t+237.1})}{(t+105)^{1.57}}$
</p>
Particulate nitrate measured every 3 or 6 days, and daily PM2.5 mass concentrations are obtained from the USEPA (https://aqs.epa.gov/aqsweb/airdata/download_files.html).

Nitrate volatilization (NO3_v) is calculated from hourly surface and dewpoint temperature values, and compared with pNO3 measured by the CSN. 
###### Huang, J., 2018: A Simple Accurate Formula for Calculating Saturation Vapor Pressure of Water and Ice. J. Appl. Meteor. Climatol., 57, 1265–1272

## ammNO3calc.py
this code calculates Kern county PM2.5 nitrate losses
the PM2.5_NO3_mass_CA.csv zip file has been uploaded for use in ammNO3calc.py
hourly temperature, rh files for Kern county are also needed to run this code

