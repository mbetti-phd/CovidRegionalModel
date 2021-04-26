#TODO: MAKE MORE USER FRIENDLY


# These is all the packages we need to be able to run the code
# If you get errors here, python3 -m pip install *packagename*
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report
from scipy.integrate import odeint
import pandas as pd
import datetime
import os
import json
import csv
import random

import uncertainties.unumpy as unp
import uncertainties as unc

def getStartDate(date,incidence):

    #Function gets the start date from the data

    if incidence > 1:
        return date.year, date.month, date.day-5
    else:
        return date.year, date.month, date.day

def mitigation(t,m,A,n,m1,t0):

    #print(np.exp(-m*t)+A*(t)**n*np.exp(-m1*(t)))
    return np.exp(-m*t)+t0*A*t**n*np.exp(-m1*t)

def f(y, t, paras):
    
    #These is the model equations

    Im = y[0]
    Is = y[1]
    Ic = y[2]
    R = y[3]
    

    try:

        r = paras['r'].value
        p = paras['p'].value
        N = paras['N'].value
        R0 = paras['R0'].value
        k = paras['k'].value
        m = paras['m'].value
        n = paras['n'].value
        m1 = paras['m1'].value
        A = paras['A'].value
        t0 = paras['t0'].value
        ps = 0.1
    except:
        print('error')

    S = (1-R/N)
    M = mitigation(t,m,A,n,m1,t0)
    # the model equations
    f0 = (1-ps)*R0*(k+(1-k)*M)*S*(Im + p*Is) - Im
    f1 = ps*R0*(k+(1-k)*M)*S*(Im + p*Is) - Is
    f2 = r*(1-ps)*R0*(k+(1-k)*M)*S*(Im + p*Is) + ps*R0*(k+(1-k)*M)*S*(Im + p*Is) #r*Im/10 + ps*R0*(k+(1-k)*M)*S*(Im + p*Is)#
    f3 = R0*(k+(1-k)*M)*S*(Im + p*Is)

    #print([f0,f1,f2,f3])

    return [f0, f1, f2, f3]


def g(t, x0, paras):
    
    #Helper function
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(paras, t, data1, data2):

    #This is where we define our residual, the thing we're trying to minimize.

    x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value, paras['x10'].value+paras['x20'].value+paras['x40'].value
    model = g(t, x0, paras)
    
    dx = np.zeros((len(t),4))

    i = 0
    dt = t[2]-t[1]
    for time in t:
        dxt = f(model[i,:],time,paras)
        dx[i,:] = dxt
        i += 1
    # you only have data for one of your variables
    x2_model = model[:, 2]
    return np.asarray([x2_model-data1, (dx[:,2]/10 - data2)]).ravel()

def progress_bar(iter,total):

    #Helper function

    per = 100*iter/total
    print(' %.2f %% Complete'% per,end='\r', flush=True)

def sensitivity(provinces,i, pop_size, is_gamma):

    #Main code. This pulls in the data, samples data, runs a fit and checks if its good.

    predict = 1#random.randint(1,7)

    folder = 'Fits0421'

    c_N = 1e8
    for province in provinces:

        #with open('CensusData_adjusted_sizes_2020.json') as file:
        #    prov_dict = json.load(file)
        prov_size = pop_size

        province = province.replace(' ','')

        D = 10
        G = 0
        if province == 'Ontario':
            G = 0
        
        # measured data
        data = pd.read_csv(province+'Data.csv',parse_dates=True)
        data['date'] = pd.to_datetime(data['date'], dayfirst=True)
        y, m, d = getStartDate(data['date'][0],data['cumulative_cases'][0])
        start_date1 = datetime.datetime(2020,12,31)
        today = str(datetime.datetime.today().date())
        start_date = datetime.datetime(2020,12,31)
        #if province in ['Ontario','BritishColumbia']:
        #    start_date = datetime.datetime(2020,3,15)
        #if province in ['Canada']:
        #    start_date = datetime.datetime(2020,3,15)
        
        ind = data.index[data['date']==start_date]
        ind = ind[0]
        days_since = (data['date'] - start_date1).dt.days
        days_since = days_since/D+G/D
        data['cumulative_cases'] = data['cumulative_cases']
        x = days_since.values
        y = data['cumulative_cases'].values
        z = data['cases'].values

        t_measured = x[:]
        x2_measured = y[:]
        z_measured = z[:]

        sample_set = np.linspace(ind,len(x)-predict,len(x)-predict-ind+1)

        sample_set = sample_set.tolist()
        #print(sample_set)
        sample_set = random.sample(sample_set,int(len(sample_set)/random.randint(1,2)))
        #sample_set.extend([len(x)-8, len(x)-7,len(x)-3,len(x)-2,len(x)-1])

        sample_set.sort()
        sample_set = [int(x) for x in sample_set]
        x = x[sample_set]
        y = y[sample_set]
        z = z[sample_set]

        #x = x[ind:]
        #y = y[ind:]
        #z = z[ind:]

        #if predict in range(1,20):

        #    x = np.delete(x,sample_set)
        #    y = np.delete(y,sample_set)
        #    z = np.delete(z,sample_set)

        sizer = len(y)

        # initial conditions
        x10 = (y[0]+1)
        x20 = 0.5*(y[0]+1)
        x30 = y[0]
        x40 = 10
        y0 = [x10, x20,x30, x40]#,x40,x40]

        LAMBDA = 0.5
        if abs(y[len(y)-1] - y[len(y)-2]) < 20:
            LAMBDA = 0

        p0 = random.random()
        r0 = random.random()
        R00 = 2+ 1.5*random.random()
        k0 = random.random()
        m0 = random.random()
        n0 = random.randint(1,25)
        X40 = random.random()/5
        X10 = random.randint(2,20)
        xx = random.randint(1, 100)
        xx1 = random.randint(1,100)
        xtra_sev = random.randint(2,10)
        # set parameters including bounds; you can also fix parameters (use vary=False)
        params = Parameters()
        params.add('x40',value = x30 + xx, min = x30/10, max = X40*prov_size)
        params.add('x10',value = x30+xx1, min = x30/2, max = X10*x30+100)
        params.add('x20', value=x30+1, min = x30/2, max = x30+xtra_sev)
        params.add('x30', value = x30, vary = False)
        params.add('p', value=p0, min = 0, max = 1)
        params.add('r', value=r0, min=0.0, max=1)
        #if abs(y[len(y)-1] - y[len(y)-2])< 10:
        params.add('N', value=prov_size, vary=False)
        #else:
        #params.add('N', value=0.75*prov_size, min = 0*prov_size, max = 0.8*prov_size)
        params.add('R0', value=R00, min = 0, max = 3.5)
        params.add('k', value=k0, min=0, max=1)
        params.add('m', value=m0, min=0, max=1)
        params.add('n', value=n0, min=0)
        params.add('m1',value=m0, min=0)
        params.add('A',value=1e-10,min=0,max=10)
        if is_gamma:
            params.add('t0',value=1,vary=False)
        else:
            params.add('t0',value=0,vary=False)
        #params.add('ps', value=0.1, min=0.001, max=0.5)
    
        # fit model
        try:
            result = minimize(residual, params, args=(x, y, z), method='leastsq',xtol=1e-13,ftol=1e-13,max_nfev=1500)  # leastsq nelder
        except:
            print('bad run')
            return 0, 0, False
        px = np.linspace(x[0], x[-1], 366)

        y0_f = [result.params['x10'].value,result.params['x20'].value,result.params['x30'].value,result.params['x10'].value+result.params['x20'].value+result.params['x40'].value]

        data_fitted = g(px, y0_f, result.params)

        #print(np.mean(abs(result.residual)), abs(result.residual[-1]))

        if abs(np.mean(result.residual))<500 and abs(result.residual[-1])<350:
            flag = True

        else:
            return 0,0,False

        plt.scatter(x, y, marker='o', color='xkcd:navy blue', label='measured data', s=20)
        #plt.scatter(data['date'][len(t_measured)-predict:], data['numtotal'][len(x2_measured)-predict:], marker='o', color='xkcd:red', label='measured data', s=20)
        #plt.scatter(x,y, marker='o', color='tab:blue', label='measured data', s=50)
        if flag:
            plt.plot(px, data_fitted[:366,1], '-', linewidth=2, color='tab:blue', label='severe cases')
            plt.plot(px, data_fitted[:366,0], '-', linewidth=2, color='tab:red', label='mild cases')
            plt.plot(px, data_fitted[:366,2], '-', linewidth=2, color='tab:green', label='known cases')
            plt.plot(px, data_fitted[:366,3], '-', linewidth=4, color='tab:purple', label='total cases')

        plt.savefig(folder+'/'+province+'/'+str(i)+'.png', dpi=300,bbox_inches='tight')
            #px = np.linspace(x[0], x[-1]+0.4, 366)
        plt.clf()
            #y0_f = [result.params['x10'].value,result.params['x20'].value,result.params['x30'].value,result.params['x10'].value+result.params['x20'].value+result.params['x40'].value]
            
            #data_fitted = g(px, y0_f, result.params)
            
        return result, x[0], flag


def main():
# Here is the main driver function

    folder = 'Fits0421'

    regions = pd.read_csv('PopulationSizes.csv')

    for index, row in regions.iterrows():

        if row['Skip']=='Yes':
            continue

        province = row['Province']+row['HR']

        is_gamma = False

        if row['Mitigation'] == 'Gamma':
            is_gamma = True

        print(province)

        pop_size = row['Population']

        try:
            os.makedirs(folder+'/'+province.replace(' ','')+'/')
        except:
            print('unable to make directory')
        files = os.listdir(folder+'/'+province.replace(' ','')+'/')
        i = 0
        # if len(files) == 0:

        #     i = 0

        # else:
        #     try:
        #         files.remove('.DS_Store')
        #     except:
        #         continue

        #    i = len(files)/2 - 1
            
        #if province == 'Toronto':
        #        i = 490
        #progress_bar(i,1000)
        while i <  row['Samples']:

            r, t0, flag = sensitivity([province],i,pop_size,is_gamma)

            if not flag:
                continue
            
            params = r.params

            trueR0 = params['R0'].value*(params['k'].value+(1-params['k'].value)*np.exp(-params['m'].value*t0))

            with open(folder+'/'+province.replace(' ','')+'/'+str(i)+'.txt', "w") as file:
                file.write(fit_report(r))
            with open(folder+'/'+province.replace(' ','')+'/'+str(i)+'.txt', "a+") as file:
                file.write('\n trueR0: '+str(trueR0))
            progress_bar(i,row['Samples'])
            i += 1

#main()