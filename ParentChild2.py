#Version: v0.1
#Date Last Updated: 1-12-2025

#DELETE THIS comment
#  Follow the coding standards listed in coding_standards.pdf 
#  Delete the sections in this template if not used

#%% MODULE BEGINS
module_name_gl = 'Parent-Child2'

'''
Version: <***>

Description:
    See if there is a specific time where the most orders are made and the income per
    hour is the highest.
    This will help us see if there is a specific time where we can have more employees


Authors:
    Elizabeth, Henry, and Davidson

Date Created     :  <4-12-2025>
Date Last Updated:  <4-12-2025>

Doc:
    <***>

Notes:
    Cannot commit from visual studio, so copy and pasting code. Need to see if Henery's
    works with this because there seems to be variation between our versions.
    Very rough draft, struggling with very vauge instructions. Probably need to see if math is 
    actually true. Median is coming out to 0, seems fishy. 
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pck
import datetime as dt

#custom imports


#other imports


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here
class Inventory():
    pass

class InventoryAnalytics(Inventory):

    def __init__(self, filename):
        self.filename = filename

    def readPickle(self):
        try:
            df_pick = pd.read_pickle(self.filename)
            return df_pick
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File {self.filename} not found in InventoryAnalysis.readPickle() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None
        

            
#Function definitions Start Here
def main():
    pass
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.")
    
    test = InventoryAnalytics('test_pickle.pkl')
    testResult = test.readPickle()
    print(testResult)

