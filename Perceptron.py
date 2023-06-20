# Lodz University of Technology
# Artificial Intelligence Fundamentals - Assignment 2
# Ahmet Galip Sengun - 904261
# 13.10.2022

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns


mode1 = st.number_input('Class 1: Number of modes')
size1 = st.number_input('Class1: Number of samples per mode')

mode2 = st.number_input('Class 2: Number of modes')
size2 = st.number_input('Class2: Number of samples per mode')

lr = st.number_input('Learning rate', value=(0.01))
epoch = st.number_input('Number of epochs', value=(500))


activation = st.selectbox(
    'Choose an activation function',
    ('sig', 'step_func', 'relu', 'leaky_relu', 'sign', 'sin', 'tan'))



def generate_data(size, n_modes = 1):
    
    scale = np.random.rand()
    loc = np.random.uniform(-5, 5, size=2)
    data1 = np.random.normal(scale = scale, loc = loc, size = (size, 2))
    
    if n_modes > 1:
        for i in range(n_modes-1):
            scale = np.random.rand()
            loc = np.random.uniform(-5, 5, size=2)
            arr = np.random.normal(scale = scale, loc = loc, size = (size, 2))
            
            data1 = np.concatenate((data1, arr), axis = 0)

    return data1

def step_func(x):
        return 1.0 if (x > 0) else 0.0
    
def sig(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x>0 else 0.01*x

def sign(x):
    if x>0 :
        return 1
    elif x==1 :
        return 0
    else:
        return -1
    
def sin(x):
    return np.sin(x)

def tan(x):
    return np.tanh(x)

def perceptron(X, y, lr=0.01, epochs=500, activation = "sig"):
    
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n+1,1))
    
    # Empty list to store how many examples were 
    
    # Training.
    for epoch in range(epochs):
        
        # looping for every example.
        for idx, x_i in enumerate(X):
            
            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            
            # Calculating prediction/hypothesis.
            if activation == "sig": y_hat = sig(np.dot(x_i.T, theta))
            if activation == "step_func": y_hat = step_func(np.dot(x_i.T, theta))
            if activation == "relu": y_hat = relu(np.dot(x_i.T, theta))
            if activation == "leaky_relu": y_hat = relu(np.dot(x_i.T, theta))
            if activation == "sign": y_hat = sign(np.dot(x_i.T, theta))
            if activation == "sin": y_hat = sin(np.dot(x_i.T, theta))
            if activation == "tan": y_hat = tan(np.dot(x_i.T, theta))

            
            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += lr*((y[idx] - y_hat)*x_i)
            
        
    return theta

def plot_decision_boundary(X, theta):
    
    fig, ax = plt.subplots()
    # X --> Inputs
    # theta --> parameters
    
    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c
    sns.scatterplot(x=df.X, y=df.Y, hue=df.Label)
    plt.plot(x1, x2, 'y-')
    st.pyplot(fig)

if st.button('Generate graph'):
    if (size1 and size2 and mode1 and mode2):
        data1 = generate_data(int(size1), n_modes = int(mode1))
        data2 = generate_data(int(size2), n_modes = int(mode2))
        
        z1 = np.zeros((len(data1),), dtype = np.int64)
        z2 = np.ones((len(data2),), dtype = np.int64)
        
        data = np.concatenate((data1, data2), axis=0)
        target = np.concatenate((z1, z2), axis=0)
        
        df1 = pd.DataFrame(data1, columns =['X', 'Y']) 
        df1['Label'] = 0
        
        df2 = pd.DataFrame(data2, columns =['X', 'Y']) 
        df2['Label'] = 1
        
        df = pd.concat([df1, df2])
        X = df[["X", "Y"]]
        y = df[['Label']]
        
        theta = perceptron(data, target, lr, epoch, activation=activation)
        plot_decision_boundary(data, theta)