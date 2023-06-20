# Lodz University of Technology
# Artificial Intelligence Fundamentals - Assignment 3
# Ahmet Galip Sengun - 904261
# 02.2022

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

c1, c2 = st.columns(2)
with c1:
    mode1 = st.text_input("Class 1: Number of modes",value=1)
with c2:
    size1 = st.text_input("Class1: Number of samples per mode",value=100)
    
c3, c4 = st.columns(2)
with c3:
    mode2 = st.text_input("Class 2: Number of modes",value=1)
with c4:
    size2 = st.text_input("Class2: Number of samples per mode",value=100)
    
c5, c6, c7, c8 = st.columns(4)
with c5:
    lr = st.text_input("learning rate",value = 0.1)
with c6:
    batch_size = st.text_input("batch size",value=32)
with c7:
    epoch = st.text_input("epoch",value=1000)
with c8:
    hidden_shape = st.text_input("hidden layer shape")
    hidden_shape = [int(i) for i in hidden_shape.split(',')]


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

# Define neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Saves the weights of the neural network to self.weights
        # Randomly initializes the weights with a normal distribution
        # layer_sizes is a list that includes the sizes of all layers in the network
        # Iterates over layer_sizes, creates a weight matrix for each layer, and appends it to self.weights
        self.weights = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def forward(self, X):
        # Performs a forward pass through the neural network
        # Given an input X, it calculates the activations of each layer and returns them as a list
        # activations[0] is the input layer and activations[-1] is the output layer
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i])
            a = self.sigmoid(z)
            activations.append(a)
        return activations
        
    def backward(self, X, y, activations):
        # Performs backpropagation through the neural network and updates the weights
        # X is the input, y is the target output, and activations is a list of activations computed by the forward pass
        # It first calculates the error at the output layer
        # Then, it calculates the deltas for each layer in the network
        # It then updates the weights using the deltas and activations
        error = activations[-1] - y
        deltas = [error * self.sigmoid_derivative(activations[-1])]
        for i in range(len(self.weights)-1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            layer = activations[i]
            delta = deltas[i]
            self.weights[i] -= self.learning_rate * np.dot(layer.T, delta)
        
    def train(self, X, y, epochs, batch_size, learning_rate):
        self.learning_rate = learning_rate  # set the learning rate
        for epoch in range(epochs):  # loop through the number of epochs
            for i in range(0, len(X), batch_size):  # loop through the training data in batches
                X_batch = X[i:i+batch_size]  # get the input batch
                y_batch = y[i:i+batch_size]  # get the label batch
                activations = self.forward(X_batch)  # compute activations
                self.backward(X_batch, y_batch, activations)  # update weights using backpropagation
            if (epoch+1) % 10 == 0:  # every 10 epochs
                loss = self.loss(X, y)  # calculate loss on the whole training set
                #print(f"Epoch {epoch+1}/{epochs}, loss={loss:.4f}")  # print the epoch number and loss
    
    def predict(self, X):
        return np.round(self.forward(X)[-1])  # make predictions using the output of the last layer
        
    def loss(self, X, y):
        y_pred = self.forward(X)[-1]  # get the output of the last layer
        return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))  # calculate binary cross-entropy loss


    
if st.button('Train'):
        if (size1 and size2 and mode1 and mode2):
            data1 = generate_data(int(size1), n_modes = int(mode1))
            data2 = generate_data(int(size2), n_modes = int(mode2))
            z1 = np.zeros((len(data1),), dtype = np.int64)
            z2 = np.ones((len(data2),), dtype = np.int64)
            
            df1 = pd.DataFrame(data1, columns =['x', 'y']) 
            df1['class'] = 0
            df2 = pd.DataFrame(data2, columns =['x', 'y']) 
            df2['class'] = 1
            df = pd.concat([df1, df2])
            
            X = df[["x","y"]].values
            y = df['class'].values.reshape((-1, 1))
            
            # Initialize and train neural network
            nn = NeuralNetwork(input_size=2, hidden_sizes=hidden_shape, output_size=1)
            nn.train(X, y, epochs=int(epoch), batch_size=int(batch_size), learning_rate=float(lr))
            
            # Evaluate model on test set
            y_pred = nn.predict(X)
            accuracy = np.mean(y_pred == y)
            st.write('Accuracy: {:.2f}%'.format(accuracy * 100))
            
            # Visualize data and decision boundary using meshgrid
            # Generate meshgrid
            h = 0.01
            x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
            y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            X_mesh = np.c_[xx.ravel(), yy.ravel()]
            
            # Plot data
            fig, ax = plt.subplots()
            ax.scatter(df[df['class'] == 0]['x'], df[df['class'] == 0]['y'], c='red', label='Class 0')
            ax.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], c='blue', label='Class 1')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            
            # Plot decision boundary
            Z_mesh = nn.predict(X_mesh)
            Z_mesh = Z_mesh.reshape(xx.shape)
            ax.contourf(xx, yy, Z_mesh, cmap=plt.cm.Spectral, alpha=0.4)
            
            st.pyplot(fig)
            

    

    
    
    
    
