import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt    
from scipy import stats
import pandas as pd
import csv
import datetime

def buildIncidence(file,province,hr):

    case_data = pd.read_csv(file)

    case_data = case_data.loc[(case_data['health_region']==hr) & (case_data['province']==province)]

    case_data['date'] = pd.to_datetime(case_data['date_report'], dayfirst=True)

    case_data = case_data.sort_values(by=['date'])

    total_cases_per_day = case_data[['date','cumulative_cases','cases']]

    province = province.replace(' ','')
    hr = hr.replace(' ','')

    total_cases_per_day.to_csv(province+hr+'Data.csv',index=False,header=True)

def main():

    data = pd.read_csv('PopulationSizes.csv')
    provinces = data['HR']


    for index, row in data.iterrows():
        pr = row['Province']
        hr = row['HR']
        buildIncidence('timeseries_hr.csv',pr,hr)
    