# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:55:38 2019

@author: devang
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats as st1
import math
def z_val(inp1,inp2):
    ''' I have assumed the null hypothesis to be that there is not interdependency amongst the features'''
    l1=[]
    l2=[]
    i=0
    max_1=max(inp1)
    min_1=min(inp1)
    max_2=max(inp2)
    min_2=min(inp2)
    while i<len(inp1):
        res1=(inp1[i]-min_1)/(max_1-min_1)
        res2=(inp2[i]-min_2)/(max_2-min_2)
        l1.append(res1)
        l2.append(res2)
        i=i+1
    inp1=l1
    inp2=l2
    inp1_mean=st.mean(inp1)
    inp2_mean=st.mean(inp2)
    ''' we have now normalised the values'''
    test_1=inp1[500:900]
    test_2=inp2[500:900]
    ''' we have taken 500 samples from the population'''
    sam1_mean=st.mean(test_1)
    sam2_mean=st.mean(test_2)
    sd_1=st.stdev(inp1)
    sd_2=st.stdev(inp2)
    z_val=((sam1_mean-sam2_mean)-(inp1_mean-inp2_mean))/(math.sqrt(((sd_1*sd_1)/500)+((sd_2*sd_2)/500)))
    p_val=st1.norm.sf(abs(z_val))
    
    print(p_val)



def mean(input1,t):
    i=0
    mean_list=[]
    while i<24:
        x=st.mean(list(input1[0:,i]))
        mean_list.append(x)
        i=i+1
    ''' now  we have the list of the mean values as per the time'''
    ''' now we will be plotting the values so as to get the gist of the concerntration'''
    label=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    plt.bar(label,mean_list)
    plt.show()
    plt.close()

def corr(input1,input2):
    i=0
    j=0
    output_list=[]
    while i<24:
        res1=list(input1[0:,i])
        res2=list(input2[0:,i])
        sol=np.corrcoef(res1,res2)
        output_list.append(sol)
        i=i+1
def norm(mat):
    i=0
    j=0
    while i<24:
        max_val=max(list(mat[0:,i]))
        min_val=min(list(mat[0:,i]))
        print(max_val)
        j=0
        while j<390:
            
             mat[j,i]=(mat[j,i]-min_val)/(max_val-min_val)
            
             j=j+1
        i=i+1
    return mat    
            


def AQ(data_set,label):
    
    
    time_slots=data_set['Time']
    time_slots=pd.unique(time_slots)
    time_slots=list(time_slots)
    time_slots.sort()

    i=0
    temp_list=[]
    final_list_1=[]
    while i<24:
         j=0
         temp_list=[]
         while j<9356:
        
            if data_set['Time'][j]==time_slots[i]:
                temp_list.append(data_set[str(label)][j])
            j=j+1    
         final_list_1.append(temp_list)
         i=i+1
    return final_list_1,time_slots





def converter(mat,a):
    mat=pd.DataFrame(mat)
    mat=np.array(mat)
    mat=np.transpose(mat)
    print(mat.shape)
    return mat






def g_plot(final_1,t,final_y):
    final_1=converter(final_1,t)
    final_1=norm(final_1)
    final_y=converter(final_y,t)
    final_y=norm(final_y)
    mean(final_1,t)
    i=0
    j=0
    while i<24:
     j=0
     
     while j<390:
        plt.scatter(final_1[j,i],final_y[j,i],color='red')
        j=j+1
     i=i+1
     
     plt.xlabel('Time Period of the Day')
     plt.ylabel('The normalised Range of of Values')
     plt.show()
     plt.close()
        


data_set=pd.read_csv('AQ.csv')
data_set=data_set.drop(columns=['Date'])
data_set=data_set.dropna()
data_set=data_set.drop(data_set.index[7713])
data_set=data_set.reset_index()
data_set=data_set.replace(-200,0)
labels=data_set.columns


''' now we have the labels '''
''' now we will move towards segregrating the data'''



final_1,t=AQ(data_set,labels[2])
final_2,t=AQ(data_set,labels[3])
final_3,t=AQ(data_set,labels[4])
final_4,t=AQ(data_set,labels[5])
final_5,t=AQ(data_set,labels[6])
final_6,t=AQ(data_set,labels[7])
final_7,t=AQ(data_set,labels[8])
final_8,t=AQ(data_set,labels[9])
final_9,t=AQ(data_set,labels[10])
final_10,t=AQ(data_set,labels[11])
final_11,t=AQ(data_set,labels[12])
final_12,t=AQ(data_set,labels[13])
final_13,t=AQ(data_set,labels[14])

''' plots regarding the output values'''
g_plot(final_2,t,final_12)
print('From The plotting we can say that we have clustered cloud realtion bewteen the input and output')
print('As far As the Bar Graph Is Concerned, we can very well say that at 8:00 The Concerntration CO is max')
g_plot(final_3,t,final_12)
print('From the Bar graph We can Easily Interpret that data gathered my sensor is maximum at 20:00')

g_plot(final_5,t,final_12)
print('We have scattered Realtionship between the Input and Output for all the time domains')
g_plot(final_6,t,final_12)
print('The maximum Level of PT08(NHMC) Sensor is at 8:00 in the morning')
g_plot(final_7,t,final_12)
print('The maximum Range of NOx is between 00:00 t0 03:00 and 12:00 to 14:00')
g_plot(final_8,t,final_12)
print('The maximum value of NO2(GT) is at 19:00 and as per the scatter plots the range is on the higherside(Close to the avg value)')

g_plot(final_10,t,final_12)
print('On the Overall basis the scatter plots tell that the concerntration of PT08.S5 lies in the central domain of max value')
g_plot(final_11,t,final_12)

g_plot(final_12,t,final_12)
''' plots with the output have been fininshed'''
''' plots between concerntrarion and analyser'''

g_plot(final_3,t,final_2)
print('From all the hourly based plots we can say that the value picked up by the analyser and level of')
print('CO is following a linear Relationship')

g_plot(final_4,t,final_5)
print('From the above hourly plots we can infer that they follow the linear Relationship with each other(highly Coorelated terms)')

g_plot(final_10,t,final_9)
print('These values also follow a bit linear relation but they are quite more scattered')

''' now we will be looking towards the various value of the correlation'''
data_temp=pd.DataFrame(data_set)
print(np.corrcoef(data_temp['CO(GT)'],data_temp['RH']))
print(np.corrcoef(data_temp['NMHC(GT)'],data_temp['RH']))
print(np.corrcoef(data_temp['C6H6(GT)'],data_temp['RH']))
print(np.corrcoef(data_temp['PT08.S2(NMHC)'],data_temp['RH']))
print(np.corrcoef(data_temp['PT08.S5(O3)'],data_temp['RH']))
print(np.corrcoef(data_temp['PT08.S3(NOx)'],data_temp['RH']))
print(np.corrcoef(data_temp['T'],data_temp['RH']))
print(np.corrcoef(data_temp['RH'],data_temp['RH']))
print(np.corrcoef(data_temp['AH'],data_temp['RH']))


''' now we will move towards the hypothesis part'''
''' since we need to design the model, we will be working for the two sapmled p value testing'''
''' this value will be determined using the z value test'''

z_val(list(data_temp['CO(GT)']),list(data_temp['RH']))
''' in this case we cannot reject the null hypo'''
z_val(list(data_temp['NMHC(GT)']),list(data_temp['RH']))
''' in this case also we can reject the null hypotheses'''
z_val(list(data_temp['C6H6(GT)']),list(data_temp['RH']))
''' in this case also we can reject the null hypothesis'''
z_val(list(data_temp['PT08.S2(NMHC)']),list(data_temp['RH']))
''' in this case we can reject the null hypothesis'''
z_val(list(data_temp['T']),list(data_temp['RH']))
''' we will accept this parameter'''
z_val(list(data_temp['AH']),list(data_temp['RH']))
''' in  this case we will reject the null hypothesis and thus accept this parameter'''
z_val(list(data_temp['PT08.S3(NOx)']),list(data_temp['RH']))
''' we will not reject the null hypotheses in this case too'''
z_val(list(data_temp['NOx(GT)']),list(data_temp['RH']))
''' we will be accepting this value too'''

z_val(list(data_temp['PT08.S5(O3)']),list(data_temp['AH']))
''' we can reject the null hypothesis'''
z_val(list(data_temp['PT08.S1(CO)']),list(data_temp['AH']))
'''we can reject the null hypothesis'''
z_val(list(data_temp['PT08.S4(NO2)']),list(data_temp['AH']))
''' we can reject the null hypothesis'''

z_val(list(data_temp['NO2(GT)']),list(data_temp['RH']))
''' we will accept this parameter'''




y=data_temp['RH']
time=data_temp['Time']
x=data_temp.drop(columns=['index','CO(GT)','PT08.S3(NOx)','RH','Time'])
from sklearn import preprocessing
temp=x.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(temp)
x = pd.DataFrame(x_scaled)

time=np.array(time)
time=time.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
categorical=OneHotEncoder()
time_cat=categorical.fit(time)
time_cat=categorical.transform(time).toarray()
time_cat=pd.DataFrame(time_cat)

x_final=[time_cat,x]
x_final=pd.concat(x_final,axis=1)
i=0
max1=max(y)
min1=min(y)
y_final=[]
while i<len(y):
    y_final.append((y[i]-min1)/(max1-min1))
    i=i+1


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final,y_final, test_size=0.33)



''' now we will be using various machine learning models'''

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)
print(lin.score(x_test,y_test))


