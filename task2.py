from turtle import color
from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('task2.csv')
data = data.drop(['name'],axis = 1)
x = data.iloc[:,0]
y = data.iloc[:,1]
#####################################################
rate = 0.0001
prev_size = 1
stop_learn = 1e-20
iters = 0 
cur_m = 50
cur_c = 100
m = lambda m : m
c = lambda m : m
def costs():
        for i in range(len(x)) :
            total = 0
            y_pred = cur_m*x[i] + cur_c
            cur_mse = (( y_pred*x[i] - y[i])**2)/(2*len(x))
            total += cur_mse    
        return(total)
cur_cost = costs()
cost_list = []
iterss = []
coefs = []
intercepts = []
prev_cost = cur_cost
while prev_size > stop_learn or cur_cost < prev_cost: 
    prev_cost = cur_cost
    prev_m = cur_m
    prev_c = cur_c
    cur_m = cur_m - rate*m(prev_m)
    cur_c = cur_c - rate*c(prev_c)
    costs()
    cur_cost = costs()
    prev_size = prev_cost-cur_cost
    iters += 1
    cost_list.append(cur_cost)
    iterss.append(iters)
    coefs.append(cur_m)
    intercepts.append(cur_c)
    # print('iteration ',iters,'\n minimum error is ',cur_cost)
    
# print('after ', iters,'iterations and  learning rate of ',rate, ',','\n the best coefficient of prediction line is :', cur_m , '\n and the best intercept of prediction line is :', cur_c )
#cost_listt =pd.DataFrame(cost_list)
#cost_listt.to_csv('outs.csv')
Y_last_pred = cur_m*x + cur_c
# plt.scatter(x,y)
# plt.plot([min(x),max(x)] , [min(y),max(y)] , [min(Y_last_pred),max(Y_last_pred)],color = 'red')
# plt.show()
results = pd.DataFrame(list(zip(iterss,cost_list,coefs,intercepts)),columns= ['iter-No','cost of current prediction line','coeficient of current prediction line','intercept of current prediction line'])
results.to_csv('task 2 results.csv')