##################################################
# Support
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 1.0.0
# Date Created: February 28, 2021
# Last Modified: March 2, 2022
# #################################################

##############################
##          OUTLINE         ##
##############################
# format_sequential_data: formats a data set in sequential format to be used
#   in a sequential regression extrapolation
# get_states_dict: returns a dictionary which converts the number of shells
#   to the number of single particle states
##############################
##          IMPORTS         ##
##############################
# THIRD-PARTY IMPORTS
# For NumPy arrays
import numpy as np

##############################
##  FORMAT SEQUENTIAL DATA  ##
##############################
def format_sequential_data (y, seq=2):
    """
        Inputs:
            y (a list or NumPy array): the y values of a data set
            seq (an int): the length of the sequence.  Default value is 2
        Returns:
            inputs (a list): the inputs for a machine learning model using 
                sequential data formatting
            outputs (a list): the outputs for a machine learning model using
                sequential data formatting              
        Formats a given list or array in sequential formatting using the 
        given sequence lenght.  Default sequence length is two.

        Explanation of sequential formatting:
        Typically data points of the form (x,y) are used to train a machine
        learning model.  This teaches the model the relationship between the
        x data and the y data in the training range.  This model works well 
        for interpolation, but not so well for extrapolation.  A better data
        model for extrapolation would be one that learns the patterns in the y
        data to better guess what y value should come next.  Therefore, this 
        method formats the data in a sequential pattern so that the points are
        of the form ((y1, y2, ..., yn), yn+1) where n is the lenght of the 
        sequence (seq).
    """
    # Make sure seq is an int
    assert isinstance(seq, int)
    # Set up the input and output lists
    inputs = []
    outputs = []
    # Cycle through the whole y list/array and separate the points into 
    # sequential format
    for i in range(0, len(y)-seq):
        inputs.append(y[i:i+seq])
        outputs.append(y[i+seq])
    # Return the input and output lists.  NOTE: the data type of the return 
    # values is LIST
    return inputs, outputs  

##############################
##      GET STATES DICT     ##
##############################    
def get_states_dict ():
    """
    Inputs:
        None.
    Returns:
        states_dict (a dictionary): the conversion between shell number and number of 
            single particle states.
    Dictionary thats converts between number of shells to number of single particle states
    for the first 178 shells.  
    """
    states_dict = {
        1:2, 2:14, 3:38, 4:54, 5:66, 6:114, 7:162, 8:186, 9:246, 10:294, 
        11:342, 12:358, 13:406, 14:502, 15:514, 16:610, 17:682, 18:730, 
        19:778, 20:874, 21:922, 22:970, 23:1030, 24:1174, 25:1238, 26:1382, 
        27:1478, 28:1502,29:1598, 30:1694, 31:1790, 32:1850, 33:1898, 
        34:2042, 35:2090, 36:2282, 37:2378, 38:2426, 39:2474, 40:2618, 
        41:2714, 42:2730, 43:2838, 44:3006, 45:3102, 46:3150, 47:3294, 
        48:3486, 49:3582, 50:3678, 51:3726, 52:3870, 53:4014, 54:4206, 
        55:4218, 56:4410, 57:4602, 58:4650, 59:4746,  60:4938, 61:5034, 
        62:5106, 63:5202, 64:5442, 65:5554, 66:5602, 67:5794, 68:5890,
        69:5938, 70:6142, 71:6238, 72:6382, 73:6478, 74:6574, 75:6814,
        76:6862, 77:7150, 78:7390, 79:7486, 80:7582, 81:7774, 82:7822,
        83:7918, 84:8134, 85:8278, 86:8338, 87:8674, 88:8770, 89:8914,
        90:9106 ,91:9250, 92:9394, 93:9458, 94:9602, 95:9890, 96:10082,
        97:10274, 98:10370, 99:10514, 100:10754, 101:10898, 102:10994,
        103:11138, 104:11330, 105:11378, 106:11618, 107:11810, 108:11834,
        109:12074, 110:12122, 111:12266, 112:12362, 113:12458, 114:12698,
        115:12794, 116:12938, 117:13034, 118:13130, 119:13226, 120:13322,
        121:13418, 122:13466, 123:13610, 124:13802, 125:13818, 126:14058,
        127:14202, 128:14298, 129:14490, 130:14586, 131:14682, 132:14778,
        133:14970, 134:15042, 135:15090, 136:15186, 137:15378, 138:15522,
        139:15618, 140:15714, 141:15762, 142:15810, 143:15906, 144:16002,
        145:16050, 146:16098, 147:16146, 148:16242, 149:16386, 150:16482,
        151:16674, 152:16722, 153:16818, 154:16914, 155:16930, 156:17026,
        157:17122, 158:17218, 159:17338, 160:17386, 161:17434, 162:17530,
        163:17626, 164:17674, 165:17770, 166:17818, 167:17914, 168:17962,
        169:18010, 170:18058, 171:18154, 172:18202, 173:18218, 174:18314,
        175:18362, 176:18410, 177:18458, 178:18506}
        
    return states_dict