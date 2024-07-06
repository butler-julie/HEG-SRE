##################################################
# Bayesian Extrapolation
# Part of Library: Sequential Regression Extrapolation
# Author: Julie Butler
# Date Created: March 2, 2023
# Last Modified: March 2, 2023
#
# Performs SRE using either Bayesian Ridge Regression or Gaussian Processes.  Also
# performs an uncertainity analysis on the results.
##################################################

##############################
##         OUTLINE          ##
##############################
# bayesian_ridge_regression_extrapolation: uses SRE formatting for CC and MBPT correlation energies to 
#   extrapolation to the converged CC correlation energy.  The machine learning algorith is Bayesian
#   ridge regression implemented by Scikit-Learn
# gaussian_process_extrapolation: uses SRE formatting for CC and MBPT correlation energies to 
#   extrapolation to the converged CC correlation energy.  The machine learning algorith is a 
#   gaussian process implemented by Scikit-Learn

##############################
##          IMPORTS         ##
##############################
# THIRD PARTY IMPORTS
# For numpy arrays
import numpy as np
# Import Bayesian ridge regression implementation
from sklearn.linear_model import BayesianRidge
# Import GP implementation and the needed kernels
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, WhiteKernel

# LOCAL IMPORTS
# For percent error calculations
from Analysis import ErrorAnalysis
# For SRE method
from Extrapolate import *
# For sequential data formatting method
from Support import *
# For methods to import and format the data set
from DataImportAndFormat import *


#############################################
## BAYESIAN RIDGE REGRESSION EXTRAPOLATION ##
############################################
def bayesian_ridge_regression_extrapolation (numbers, filenames, seq=3, start_dim=5,dim=15, isDisplay=False, isOld=False, sep=" "):
    """
    Inputs:
        numbers (a list of ints): the number of particles in each calculation to be performed
        filenames (a list of strings): the filenames for each data set to be analyzed
        seq (an int): the length of the sequence for the sequential regression extrapolation format.  Default 
            value is 3.
        start_dim (an int): the index to start the training data.  Default value is 5.
        dim (an int): one more than the index to end the training data.  Default value is 15.
        isDisplay (a boolean): True case prints information to the console about every analysis performed.  
            Default value is False.
        isOld (a boolean): True cases uses a method written for the old method of formatting data.  Default value 
            is False.
        sep (a string): the delimiter in the data files.  Default value is " ".
    Returns:
        data (a 2D list): the first column is the fully calculated converged CC correlation energy and the
            second column is the ML prediction.  There is a row for each analysis performed.
        std (a 1D list): the uncertainity on the prediction of the converged CC correlation energy for each 
            analyze that is performed.
        errs (a 1D list): the percent error for each analysis that is performed
    Uses SRE formatting for CC and MBPT correlation energies to extrapolation to the converged CC 
    correlation energy.  The machine learning algorith is Bayesian ridge regression implemented by Scikit-Learn.
    """
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    # Check to make sure certain requirements are met and force some arguments to ints to avoid
    # later errors
    assert len(numbers) == len(filenames)
    seq = int(seq)
    start_dim = int(start_dim)
    dim = int(dim)
    
    # Create an instance of the error analysis class.  Used to calculate percent error.
    ea = ErrorAnalysis()
    
    # Set the number of training points and the length of the 
    # sequence for the SRE algorithm
    data = []
    errs = []
    std = []
    
    # Loop through every number of particles given
    for i in range(len(numbers)):
        if isDisplay:
            print("***************************************************")
            print("Number of Neutrons:",numbers[i])
        # Import and format the data using the appropriate method
        if isOld:
            states,mbpt,cc = import_and_split_columns_old(filenames[i],sep=sep)
        else:
            states, mbpt, cc, mbpt_times, cc_times = import_and_split_columns(filenames[i], sep=sep)
            
        # Seperate the final values (the test data), and the training data
        final_cc = get_70(cc)
        final_mbpt = get_70(mbpt)
        cc_seq = get_seq(cc)
        mbpt_seq=get_seq(mbpt)

        # Separate the specified points for the training data, without the first one
        # which would yield 0/0 with division
        training_data = cc_seq[start_dim:dim]/mbpt_seq[start_dim:dim]
        training_data = training_data.tolist()
        
        # Format the training data in sequential series formatting
        x_train, y_train = format_sequential_data (training_data, seq=seq)
       
        # Set up kernel function 
        kernel = ConstantKernel()*RationalQuadratic()+WhiteKernel()#3.94%

        # Set up Gaussian Process algorithm and train it
        R = BayesianRidge(n_iter=10000,tol=1e-15)
        R.fit(x_train, y_train)

        # Feed the trained GP algorithm and data into the SRE method
        ypred,ystd = sequential_extrapolate_sklearn(R, training_data, 50, seq=seq)

        # Separate the last prediction as the final converged slope
        final_slope = ypred[-1]
        
        # Data analysis on the results
        if isDisplay:
            print("ERROR BETWEEN SRE AND TRUE CONVERGED VALUES:",np.abs((final_cc/final_mbpt)- final_slope))
            print("CONVERGENCE OF PREDICTION:", np.abs(ypred[-2]- final_slope))
            print()

        # Predict the CC correlation energy at the long range point
        cc_prediction = final_slope*final_mbpt
        # Calculate the uncertainity on the CC correlation energy prediction
        slope_uncertainity = ystd[-1]
        cc_prediction_uncertainity = slope_uncertainity*final_mbpt
        
        if isDisplay:
            print("PERCENT ERROR FOR EXTRAPOLATION:", ea.percent_error(cc_prediction, final_cc))
            print("Predicted CCD Correlation Energy:", final_slope*final_mbpt)
            print("Actual CCD Correlation Energy:", final_cc)
        
        # Save the neccessary data
        data.append([final_cc, cc_prediction])
        errs.append(ea.percent_error(cc_prediction, final_cc))
        std.append(cc_prediction_uncertainity)

        if isDisplay:
            print("********************************************************************")
    
    if isDisplay:        
        print("AVERAGE PERCENT ERROR:", np.avg(errs))
        print("MAXIMUM PERCENT ERROR:", np.max(errs))
        print("AVERAGE UNCERTAINITY:", np.avg(std))

    return data, std, errs
    
    
####################################
## GAUSSIAN PROCESS EXTRAPOLATION ##
####################################
def gaussian_process_extrapolation (numbers, filenames, n=1, seq=3, start_dim=5,dim=15, isDisplay=False, isOld=False, sep=" "):
    """
    Inputs:
        numbers (a list of ints): the number of particles in each calculation to be performed
        filenames (a list of strings): the filenames for each data set to be analyzed
        n (an int or float): the power to raise the standard deviation of the training data to as the alpha
            value in the Gaussian Process algorithm.  Default value is 1.
        seq (an int): the length of the sequence for the sequential regression extrapolation format.  Default 
            value is 3.
        start_dim (an int): the index to start the training data.  Default value is 5.
        dim (an int): one more than the index to end the training data.  Default value is 15.
        isDisplay (a boolean): True case prints information to the console about every analysis performed.  
            Default value is False.
        isOld (a boolean): True cases uses a method written for the old method of formatting data.  Default value 
            is False.
        sep (a string): the delimiter in the data files.  Default value is " ".
    Returns:
        data (a 2D list): the first column is the fully calculated converged CC correlation energy and the
            second column is the ML prediction.  There is a row for each analysis performed.
        std (a 1D list): the uncertainity on the prediction of the converged CC correlation energy for each 
            analyze that is performed.
        errs (a 1D list): the percent error for each analysis that is performed
    Uses SRE formatting for CC and MBPT correlation energies to extrapolation to the converged CC 
    correlation energy.  The machine learning algorith is a gaussian process implemented by Scikit-Learn.
    The kernel is set to ConstantKernel()*RationalQuadratic() + WhiteKernel() and the alpha value of the GP
    algorithm is set to be the standard deviation of the training data raised to the power specified in the 
    arguments.
    """
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    # Check to make sure certain requirements are met and force some arguments to ints to avoid
    # later errors
    assert len(numbers) == len(filenames)
    seq = int(seq)
    start_dim = int(start_dim)
    dim = int(dim)
    
    # Create an instance of the error analysis class.  Used to calculate percent error.
    ea = ErrorAnalysis()
    
    # Set the number of training points and the length of the 
    # sequence for the SRE algorithm
    data = []
    errs = []
    std = []
    
    # Loop through every number of particles given
    for i in range(len(numbers)):
        if isDisplay:
            print("***************************************************")
            print("Number of Neutrons:",numbers[i])
        # Import and format the data using the appropriate method
        if isOld:
            states,mbpt,cc = import_and_split_columns_old(filenames[i],sep=sep)
        else:
            states, mbpt, cc, mbpt_times, cc_times = import_and_split_columns(filenames[i], sep=sep)
            
        # Seperate the final values (the test data), and the training data
        final_cc = get_70(cc)
        final_mbpt = get_70(mbpt)
        cc_seq = get_seq(cc)
        mbpt_seq=get_seq(mbpt)

        # Separate the specified points for the training data, without the first one
        # which would yield 0/0 with division
        training_data = cc_seq[start_dim:dim]/mbpt_seq[start_dim:dim]
        training_data = training_data.tolist()
        
        # Format the training data in sequential series formatting
        x_train, y_train = format_sequential_data (training_data, seq=seq)
       
        # Set up kernel function 
        kernel = ConstantKernel()*RationalQuadratic()+WhiteKernel()#3.94%

        # Set up Gaussian Process algorithm and train it
        R = GP(kernel=kernel, alpha=np.std(training_data)**n,n_restarts_optimizer=5)
        R.fit(x_train, y_train)

        # Feed the trained GP algorithm and data into the SRE method
        ypred,ystd = sequential_extrapolate_sklearn(R, training_data, 50, seq=seq)

        # Separate the last prediction as the final converged slope
        final_slope = ypred[-1]
        
        # Data analysis on the results
        if isDisplay:
            print("ERROR BETWEEN SRE AND TRUE CONVERGED VALUES:",np.abs((final_cc/final_mbpt)- final_slope))
            print("CONVERGENCE OF PREDICTION:", np.abs(ypred[-2]- final_slope))
            print()

        # Predict the CC correlation energy at the long range point
        cc_prediction = final_slope*final_mbpt
        
        # Calculate the uncertainity on the CC correlation energy prediction
        slope_uncertainity = ystd[-1]
        cc_prediction_uncertainity = slope_uncertainity*final_mbpt
        
        if isDisplay:
            print("PERCENT ERROR FOR EXTRAPOLATION:", ea.percent_error(cc_prediction, final_cc))
            print("Predicted CCD Correlation Energy:", final_slope*final_mbpt)
            print("Actual CCD Correlation Energy:", final_cc)
        
        # Save the neccessary data
        data.append([final_cc, cc_prediction])
        errs.append(ea.percent_error(cc_prediction, final_cc))
        std.append(cc_prediction_uncertainity)

        if isDisplay:
            print("********************************************************************")
    
    if isDisplay:        
        print("AVERAGE PERCENT ERROR:", np.avg(errs))
        print("MAXIMUM PERCENT ERROR:", np.max(errs))
        print("AVERAGE UNCERTAINITY:", np.avg(std))

    return data, std, errs