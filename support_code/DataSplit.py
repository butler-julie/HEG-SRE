#################################################
# Data Split
# Julie Butler Hartley
# Version 1.0.0
# Date Created: July 25, 2021
# Last Modified: July 25, 2021
#
# A collection of three different methods to split an overall
# data set into training and test sets.  Allows for splitting
# three lists.arrays the same way (the input data, the output
# data, and some other related array).
#################################################

#############################
# IMPORTS
#############################
# To generate random numbers
from random import randrange
# For linspace
import numpy as np

#############################
# SPLIT EVEN
#############################
def split_even(x, y, g, split=0.60):
    """
        Inputs:
            x (a list or array): the input data set
            y (a list or array): the output data set
            g (a list or array): an auxillary data set that needs to
                be split as well
            split (a float): the fraction of the data set to be
                kept in the training set.  Default is 0.60
        Returns:
            x_train, y_train, g_train (lists): the training sets for each
                of the given data sets
            x_test, y_test, g_test (lists): the test sets for each
                of the given data sets
        Splits the three given data sets into training and test data sets by
        evenly distributing the training data over the data range and using
        the rest of the data set as the test data.
    """
    # Create lists to hold training and test data sets
    x_train = list()
    y_train = list()
    g_train = list()
    x_test = list(x)
    y_test = list(y)
    g_test = list(g)

    # Create the indices that will be the data in the training set
    train_size = int(split * len(x))
    indices = np.linspace(0, len(x)-1, train_size)
    indices = [int(i) for i in indices]

    # For each index in the total data set, place the elements in the
    # correct lists (training or test)
    for i in range(0, len(x)):
        if i in indices:
            x_train.append(x[i])
            y_train.append(y[i])
            g_train.append(g[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
            g_test.append(g[i])

    # Return the training and test data   
    return x_train, x_test, y_train, y_test, g_train, g_test
 
#############################
# SPLIT RANDOM
#############################
def split_random(x, y, g, split=0.60):
    """
        Inputs:
            x (a list or array): the input data set
            y (a list or array): the output data set
            g (a list or array): an auxillary data set that needs to
                be split as well
            split (a float): the fraction of the data set to be
                kept in the training set.  Default is 0.60
        Returns:
            x_train, y_train, g_train (lists): the training sets for each
                of the given data sets
            x_test, y_test, g_test (lists): the test sets for each
                of the given data sets
        Splits the three given data sets into training and test data sets 
        randomly.

        Modified from code found at: 
        https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
    """
    # Create lists to hold the training and test data sets
    x_train = list()
    y_train = list()
    g_train = list()
    x_test = list(x)
    y_test = list(y)
    g_test = list(g)

    # The number of points in the training data
    train_size = split * len(x)

    # Randomly assign each value in the data sets to either training or test
    while len(x_train) < train_size:
        index = randrange(len(x_test))
        x_train.append(x_test.pop(index))
        y_train.append(y_test.pop(index))
        g_train.append(g_test.pop(index))

    # Return the training and test data sets
    return x_train, x_test, y_train, y_test, g_train, g_test

#############################
# SPLIT EXTRAPOLATE
#############################
def split_extrapolate (x, y, g, split=0.60):
    """
        Inputs:
            x (a list or array): the input data set
            y (a list or array): the output data set
            g (a list or array): an auxillary data set that needs to
                be split as well
            split (a float): the fraction of the data set to be
                kept in the training set.  Default is 0.60
        Returns:
            x_train, y_train, g_train (lists): the training sets for each
                of the given data sets
            x_test, y_test, g_test (lists): the test sets for each
                of the given data sets
        Splits the three given data sets into training and test data sets 
        making the first x points of each data set the training data (where z
        is the size specified by the split parameter).  The remainder of each data
        set is the test data.
    """  
    # Define the index where the split will occur  
    dim = int(split*len(x))

    # Split the data sets into training and test
    x_train = x[:dim]
    x_test = x[dim:]
    y_train = y[:dim]
    y_test = y[dim:]
    g_train = g[:dim]
    g_test = g[dim:]

    # Return the training and test data sets
    return x_train, x_test, y_train, y_test, g_train, g_test    