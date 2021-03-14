#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:41:08 2021

@author: nma
"""

""" 5 second interval start time : 14/03/21 06:20"""

import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sbs

out_file = "/home/nma/hdd/GIT/automation1.0/post_tool/test.out"

read_data = pd.read_fwf(out_file,sep=None,header=None)[2:][0]

#%%
read_data.reset_index(inplace=True,drop=True)
vals = []
time1 = dt.datetime(2021,3,14,18,20,0)
dtype = []
time_ = []
temp_vals,pre_vals,hum_vals = [],[],[]
for ii in range(0,len(read_data)):
    dtype += [read_data[ii][:3]]
    if dtype[ii] == 'TEM' :
        time_ += [time1 + dt.timedelta(0,ii*5)]
        temp_vals += [read_data[ii][4:]]
    if dtype[ii] == 'PRE' :
        pre_vals += [read_data[ii][4:]]
    
    if dtype[ii] == 'HUM' :
        hum_vals += [read_data[ii][4:]]

temp,pre,hum = np.array(temp_vals),np.array(pre_vals),np.array(hum_vals)

fin_dft = pd.DataFrame({'date':time_,'temp':temp,'pre':pre,'hum':hum})

fin_dft.set_index('date',inplace=True)

sbs.lineplot(y='temp',x='date',data=fin_dft)