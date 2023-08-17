# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:29:59 2020

@author: mh iyer

Linear Regression with Two variables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get cost given parameter and ground truth values
# hypothesis = candidate*x
# cost = sum_over(square_of(h(x) - y))

# done in two ways:
# traditional
def computeCost_traditional(theta,x,y):
    # initialize total_cost
    total_cost = 0
    # compute h(x)
    h_x = 0
    # compute the cost by looping over each example

    for i in range(len(y)):
        # compute current h(x)
        for index in range(0,len(theta)):
            h_x+=theta[index]*x[i][index]
        # get euclidean distance of h(x) from the corresponding ground truth of y
        total_cost+=(h_x - y[i])**2
        cost = (1/(2*len(y))+0.0)*total_cost
    return cost

# using numpy optimized calculations for speed
def computeCost(theta, x, y):
    cost = (1/(2*len(y))) * np.sum((np.matmul(x,theta) - y)**2, axis = 0)
    return(cost)
    

# gradient descent
def gradientDescent_multi(X,y,theta,alpha,iterations):
    # initialize some parameters
    m = len(y)
    J_history = []
    
    for iteration in range(iterations):
    
        # calculate error
        error = np.matmul(X,theta) - y
        # loop through the thetas
        for index in range(len(theta)):
            theta[index] = theta[index] - (alpha/m)*np.sum(np.matmul(error, X[:,index]), axis = 0)
        # store values of cost
        J_history.append(computeCost(theta, X, y))
    
    return theta, J_history
    
# get value of y for a value of x
def get_value(x,theta):
    value = np.matmul(np.array(x), theta)
    return(value)


# feature normalization - input a pandas dataframe, get a normalized dataframe(each column/feature is normalized)
#loop through the features, subtract the mean, and divide by the standard devation of the feature 
def feature_normalization(df, features):
    df_normalized = df.copy()
    for column in features:
        orig = df[column]
        # subtract mean
        mean_subtracted = orig - np.average(orig)
        # divide by standard deviation
        sd = np.std(mean_subtracted)
        normalized = [x/sd for x in mean_subtracted]
        # update the new dataframe
        df_normalized[column] = normalized
    return df_normalized
    
# main routine
if __name__ == "__main__":
    
    # open text file
    df = pd.read_csv('ex1data2.txt', sep =',',header=None)
    
    # rename the columns, according to the programming assignment
    df = df.rename(columns = {0:'Size', 1:'Bedrooms',2:'Price'})

    # features 
    features = ['Size','Bedrooms']
    # get feature normalized- pandas dataframe
    df_normalized = feature_normalization(df,features)
  
    # initialize variables according to the steps laid out in the assignment- also add a 1 that corresponds to the x0 term
    X=np.ones([len(df_normalized),len(features)+1])
    X[:,1:3] = np.array([[df_normalized.iloc[j][i] for i in features] for j in range(len(df_normalized))])
    
    y=df_normalized['Price']
    m=len(y)
    
   

    # initialize some parameters for linear regression
    alpha = 0.1
    iterations = 400
    theta = np.zeros(3,)

    
    # compute the cost for theta = [0,0]
    starting_cost = computeCost(theta, X, y )
    print('starting cost is : ',starting_cost)

    # perform gradient descent
    theta, J_history = gradientDescent_multi(X,y,theta,alpha,iterations)
        
    # print out values of theta
    print('Theta obtained:',theta)


    # plot the data
    fig, axs = plt.subplots(2,1,sharex=False)
    
    # plot data
    axs[0].scatter(df_normalized['Size'],y,c='red',marker='x')
    axs[0].set(xlabel='Size of House', ylabel='Price')
    axs[0].set_title('Variation of Price with Size of the House')
    # plot a line using calculated theta
    df['Calculated Price'] = [np.matmul(v,theta) for v in X]
    axs[0].plot(df_normalized['Size'],df['Calculated Price'], c='blue')    

    # Plot the cost
    axs[1].scatter(range(iterations), J_history, c='green')
    axs[1].set(xlabel = 'Iterations', ylabel='J(theta)')
    axs[1].set_title('Cost at each Iteration')
    fig.subplots_adjust(hspace=1)
