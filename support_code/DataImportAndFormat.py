##################################################
# Data Import and Format
# Part of Library: Sequential Regression Extrapolation
# Author: Julie Butler
# Date Created: March 1, 2023
# Last Modified: March 1, 2023
#
# A collection of functions to import, format, and access the data
# associated with the author's thesis research and codes.
##################################################

##############################
##          OUTLINE         ##
##############################
# generate_filenames: generates the list of filenames for the data and the list of
#   particles numbers for the electron gas and pure neutron matter (x2 for symmetric
#   nuclear matter)
# import_and_split_columns: imports the data at a given file name and splits the columns
#   into states (the number of shells), CC correlation energy, MBPT correlation energy,
#   CC calculation time, and MBPT calculation time.  Allows data to be separated with any
#   character.
# import_and_split_columns_old: imports the data at a given file name and splits the columns
#   into states (the number of shells), CC correlation energy, MBPT correlation energy. Assumes
#   the data is separated only by commas.
# get_seq: returns the first 25 entries in the array it is passed, corresponding to the data for 
#   calculations up to 24 open shells, including the calculation at 0 open shells
# get_plus_25: returns the point in the passed array corresponding the the calculation at 25
#   open shells
# get_50: returns the point in the passed array corresponding the the calculation at 50 shells
# get_plus_50: returns the point in the passed array corresponding the the calculation at 40
#   open shells
# get_65: returns the point in the passed array corresponding the the calculation at 65 shells
# get_40: returns the point in the passed array corresponding the the calculation at 40 shells
# get_70: returns the point in the passed array corresponding the the calculation at 70 shells


##############################
##          IMPORTS         ##
##############################
# THIRD PARTY IMPORTS
# For NumPy arrays
import numpy as np
# For importing the data file
import pandas as pd


##############################
## IMPORT AND SPLIT COLUMNS ##
##############################
def generate_filenames (prefix):
    """
    generates the list of filenames for the data and the list of particles numbers for the electron 
    gas and pure neutron matter (x2 for symmetric nuclear matter)
    """
    filenames = np.array([prefix+"N_2.csv",prefix+"N_14.csv",prefix+"N_38.csv",prefix+"N_54.csv",prefix+"N_66.csv",
                      prefix+"N_114.csv",prefix+"N_162.csv",prefix+"N_186.csv",prefix+"N_246.csv",prefix+"N_294.csv",
                      prefix+"N_342.csv",prefix+"N_358.csv",prefix+"N_406.csv",prefix+"N_502.csv",prefix+"N_514.csv"])
    numbers = [2, 14, 38, 54, 66, 114, 162, 186, 246, 294, 342, 358,406,502,514]
    return filenames, numbers


##############################
## IMPORT AND SPLIT COLUMNS ##
##############################
def import_and_split_columns (data_path,sep=','):
    """
    imports the data at a given file name and splits the columns into states 
    (the number of shells), CC correlation energy, MBPT correlation energy, CC calculation 
    time, and MBPT calculation time.  Allows data to be separated with any character.
    """
    data = pd.read_csv(data_path,sep=sep)
    # Separate the sequential data into arrays
    states = np.asarray(data["states"])
    mbpt = np.asarray(data["mbpt"])
    cc = np.asarray(data["cc"])
    cc_times = np.asarray(data["cc_times"])
    mbpt_times = np.asarray(data["mbpt_times"])
    return states, mbpt, cc, mbpt_times, cc_times


##################################
## IMPORT AND SPLIT COLUMNS OLD ##
##################################    
def import_and_split_columns_old (data_path):
    """
    imports the data at a given file name and splits the columns into states 
    (the number of shells), CC correlation energy, MBPT correlation energy. Assumes
    the data is separated only by commas.
    """
    data = pd.read_csv(data_path)
    # Separate the sequential data into arrays
    states = np.asarray(data["states"])
    mbpt = np.asarray(data["mbpt"])
    cc = np.asarray(data["cc"])
    return states, mbpt, cc 


##############################
##         GET SEQ          ##
##############################
def get_seq (data):
    """
    returns the first 25 entries in the array it is passed, corresponding to the data for 
    calculations up to 24 open shells, including the calculation at 0 open shells
    """
    return data[:25]


##############################
##       GET PLUS 25        ##
##############################
def get_plus_25 (data):
    """
    returns the point in the passed array corresponding the the calculation at 25
    open shells
    """
    return data[-6]


##############################
##          GET 50          ##
##############################
def get_50 (data):
    """
    returns the point in the passed array corresponding the the calculation at 50 shells
    """
    return data[-5]    
    
    
##############################
##       GET PLUS 50        ##
##############################
def get_plus_50 (data):
    """
    returns the point in the passed array corresponding the the calculation at 50
    open shells
    """
    return data[-4]
    
    
##############################
##          GET 65          ##
##############################
def get_65 (data):
    """
    returns the point in the passed array corresponding the the calculation at 65 shells
    """
    return data[-3]
    
    
##############################
##          GET 40          ##
##############################
def get_40 (data):
    """
    returns the point in the passed array corresponding the the calculation at 40 shells
    """    
    return data[-2]
    
    
##############################
##          GET 70          ##
##############################
def get_70 (data):
    """
    returns the point in the passed array corresponding the the calculation at 70 shells
    """    
    return data[-1]