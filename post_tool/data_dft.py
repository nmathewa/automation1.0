#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:41:08 2021

@author: nma
"""

""" 5 second interval start time : 14/03/21 06:20"""

from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sbs
import glob


out_file = "/home/nma/hdd/GIT/automation1.0/post_tool/19_03_21_13_13_00.out"

read_data = pd.read_fwf(files[1],sep=None,header=None)[2:][0]

#%%
files = glob.glob("/home/nma/hdd/GIT/automation1.0/post_tool/19_*")

fin_data = []
tim_list = []
vals = []
dtype = []
time_ = []
nw_time = []
t_index = []
temp_vals,pre_vals,hum_vals = [],[],[]

import datetime as dt
for jj in range(len(files)):
    read_data = pd.read_fwf(files[jj],sep=None,header=None)[0]
    mon,dat,hr,mm = int(files[jj][46:47]),int(files[jj][42:44]),int(files[jj][51:53]),int(files[jj][54:56])
    tim_list += [datetime(2021,mon,dat,hr,mm,0)]
    for ii in range(0,len(read_data)):
        time_ += [tim_list[jj] + dt.timedelta(seconds = ii*5)]
        dtype += [read_data[ii][:3]] 
        if dtype[ii] == 'TEM' and dtype[ii] == 'PRE' and dtype[ii] == 'HUM':
            time_ += [tim_list[jj] + dt.timedelta(seconds = ii*5)]
#%%
            temp_vals += [read_data[ii][4:]]
        if dtype[ii] == 'PRE' :
            pre_vals += [read_data[ii][4:]]   
        if dtype[ii] == 'HUM' :
            hum_vals += [read_data[ii][4:]]
        
#fin_data.reset_index(inplace=True,drop=True)


#%%
for ii in range(0,len(fin_data)):
    dtype += [fin_data[ii][:3]] 
    if dtype[ii] == 'TEM' :
        temp_vals += [fin_data[ii][4:]]
    if dtype[ii] == 'PRE' :
        pre_vals += [fin_data[ii][4:]]   
    if dtype[ii] == 'HUM' :
        hum_vals += [fin_data[ii][4:]]

temp,pre,hum = np.array(temp_vals),np.array(pre_vals),np.array(hum_vals)

if len(temp) != len(pre) != len(hum):
    mm = int((len(temp) + len(pre) + len(hum))/3)
    temp,pre,hum = temp[:mm],pre[:mm],hum[:mm]

fin_dft = pd.DataFrame({'temp':temp,'pre':pre,'hum':hum})
#%%
for ii in range(len(fin_dft)):
    time_ += [time1 + dt.timedelta(seconds = ii*5)]
fin_dft['Date'] = time_
fin_dft.set_index('Date',inplace=True)

#sbs.lineplot(y='temp',x='Date',data=fin_dft)

#%%
files = glob.glob("/home/nma/hdd/GIT/automation1.0/post_tool/19_*")

def int_dft(infile):
    read_data = pd.read_fwf(infile,sep=None,header=None)[2:][0]
    mon,dt,hr,mm = int(infile[46:47]),int(infile[42:44]),int(infile[51:53]),int(infile[54:56])
    time1 = datetime(2021,mon,dt,hr,mm,0)
    dtype = []
    time_ = []
    temp_vals,pre_vals,hum_vals = [],[],[]
    for ii in range(0,len(read_data)):
        dtype += [read_data[ii][:3]]
        if dtype[ii] == 'TEM' :
            temp_vals += [read_data[ii][4:]]
        if dtype[ii] == 'PRE' :
            pre_vals += [read_data[ii][4:]]   
        if dtype[ii] == 'HUM' :
            hum_vals += [read_data[ii][4:]]
            
    temp,pre,hum = np.array(temp_vals),np.array(pre_vals),np.array(hum_vals)

    fin_dft = pd.DataFrame({'temp':temp,'pre':pre,'hum':hum})
    for jj in range(len(fin_dft)):
        time_ += [time1 + dt.timedelta(seconds = jj*5)]
    
    fin_dft['Date'] = time_
    fin_dft.set_index('Date',inplace=True)
    return fin_dft
#%%
fir_file = int_dft(files[0])