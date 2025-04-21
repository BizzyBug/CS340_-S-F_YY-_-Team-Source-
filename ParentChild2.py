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
        
    def Probability(self, **kwargs):
        df = self.readPickle()

        joint_df = pd.Series(True, index=df.index)

        for col, val in kwargs.items():
            if col not in df.columns:
                with open(f'Doc/log_file.txt', 'a') as file:
                    file.write(f"Column {col} not found in DataFrame.\n")
                    file.write(f'{dt.datetime.now()}\n\n')
                continue

            if callable(val):
                    joint_df &= val(df[col])
                    b_count = df[col].apply(val).sum()
            else:
                    joint_df &= df[col] == val
                    b_count = (df[col] == val).sum()


        self.JointCount = joint_df.sum()

        if b_count > 0:
            conditional_probability = round(self.JointCount / b_count,2)
        else:
            conditional_probability = 0 

        self.JointCount = joint_df.sum()
        self.joint_probability = round(self.JointCount / len(df),2)

        mean = round(joint_df.mean(),2)
        median = round(joint_df.median(),2)
        std = round(joint_df.std(),2)

        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f"Joint Count: {self.JointCount}\n")
            file.write(f"Joint Probability: {self.joint_probability}\n")
            file.write(f"Conditional Probability: {conditional_probability}\n")
            file.write(f"Mean: {mean}\n")
            file.write(f"Median: {median}\n")
            file.write(f"Standard Deviation: {std}\n")

    def Vector_Ops(self, col1, col2):
        df = self.readPickle()

        if col1 not in df.columns or col2 not in df.columns:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"Column {col1} or {col2} not found in DataFrame.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None
        
        vector1 = df[col1].values
        vector2 = df[col2].values
        vectors = df[[col1, col2]].values
        
        def get_position_vector(self,index)
            position_vector = vectors[index]
            
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

