# Lodz University of Technology
# Artificial Intelligence Fundamentals - Assignment 1
# Ahmet Galip Sengun - 904261
# 13.10.2022

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

mode1 = st.number_input('Class 1: Number of modes')
size1 = st.number_input('Class1: Number of samples per mode')

mode2 = st.number_input('Class 2: Number of modes')
size2 = st.number_input('Class2: Number of samples per mode')

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

if st.button('Generate graph'):
    if (size1 and size2 and mode1 and mode2):
        data1 = generate_data(int(size1), n_modes = int(mode1))
        data2 = generate_data(int(size2), n_modes = int(mode2))
        
        fig, ax = plt.subplots()
        ax.scatter(*zip(*data1))
        ax.scatter(*zip(*data2))
        
        st.pyplot(fig)