# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:29:59 2020

@author: mh iyer

Linear Regression with One variable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
def gradientDescent(X,y,theta,alpha,iterations):
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
    
# main routine
if __name__ == "__main__":
    
    # open text file
    df = pd.read_csv('ex1data1.txt', sep =',',header=None)
    
    # rename the columns, according to the programming assignment
    df = df.rename(columns = {0:'Population', 1:'Profit'})
    
    # initialize variables according to the steps laid out in the assignment
    x = df['Population']
    y = df['Profit']

    # the number of samples
    m = len(y)    
   
    
    # initialize some parameters for linear regression
    alpha = 0.01
    iterations = 1500
    theta = np.zeros(2,)
    
    # prepare X - for each x, concatenate a 1 to the left of it- this indicates x0
    X = np.array([[1,x1] for x1 in x])
    
    # compute the cost for theta = [0,0]
    starting_cost = computeCost(theta, X, y )
    print('starting cost is : ',starting_cost)

    # perform gradient descent
    theta, J_history = gradientDescent(X,y,theta,alpha,iterations)
        
    # print out values of theta
    print('Theta obtained:',theta)


    # plot the data
    fig, axs = plt.subplots(2,1,sharex=False)
    
    # plot profit vs poulation, and the fitted line
    axs[0].scatter(x,y,c='red',marker='x')
    axs[0].set(xlabel='Population of City in 10,000s', ylabel='Profit in 10,000s')
    axs[0].set(xlim = [4,24] ,ylim =[-5,25] )
    axs[0].set_title('Variation of Profits with Population')
    # plot a line using calculated theta
    df['Calculated Profits'] = [np.matmul(v,theta) for v in X]
    axs[0].plot(df['Population'],df['Calculated Profits'], c='blue')    

    # Plot the cost
    axs[1].scatter(range(iterations), J_history, c='green')
    axs[1].set(xlabel = 'Iterations', ylabel='J(theta)')
    axs[1].set_title('Cost at each Iteration')
    fig.subplots_adjust(hspace=1)
    
    # visualize 3d surface plot
    
    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100);
    theta1_vals = np.linspace(-1, 4, 100);
    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])
    
    # Fill out J_vals
    for i  in range(len(theta0_vals)):
        for j  in range(len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]]  
            J_vals[i,j] = computeCost(t, X, y)

    fig = plt.figure()
    ax=plt.axes(projection='3d')
    ax.set(xlabel='Theta0',ylabel='Theta1')
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals,cmap='cool',lw=0)
    # plot computed theta0,theta1
    ax.scatter3D(theta[0], theta[1], computeCost(theta,X,y),s=100,c='r')
    plt.show()
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # plot profits for some values of population
    pop1 = 3.5
    print('Expected Profit for population ',int(pop1*10000), ':$', np.round(get_value([1,pop1], theta)*10000,2))
    pop2 = 7
    print('Expected Profit for population ',int(pop2*10000),':$', np.round(get_value([1,pop2], theta)*10000,2))
    