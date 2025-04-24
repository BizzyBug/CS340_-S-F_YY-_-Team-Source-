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
        event_results = {}
        for col, val in kwargs.items():
            if col not in df.columns:
                with open(f'Doc/log_file.txt', 'a') as file:
                    file.write(f"Column {col} not found in DataFrame.\n")
                    file.write(f'{dt.datetime.now()}\n\n')
                continue
        
                               
            if callable(val):
                condition = val(df[col])
        
            else:
                condition = df[col] == val
            
            joint_df = joint_df & condition 

            event_results[col] = condition

        
        event1_col, event1_condition = list(kwargs.items())[0]
        event2_col, event2_condition = list(kwargs.items())[1] 

    
        if callable(event1_condition):
            event1 = df[event1_condition(df[event1_col])]
        else:
            event1 = df[df[event1_col] == event1_condition]

        if callable(event2_condition):
            event2 = df[event2_condition(df[event2_col])]
        else:
            event2 = df[df[event2_col] == event2_condition]
    
        
        JointCount = joint_df.sum()

        Prob_A = len(event1) / len(df)
        Prob_B = len(event2) / len(df)

        prob_A_and_B = JointCount / len(df)
        prob_A_given_B = prob_A_and_B / Prob_B
        prob_B_Given_A = prob_A_and_B / Prob_A

        mean1 = round(event1[event1_col].mean(numeric_only = True),2)
        mean2 = round(event2[event2_col].mean(),2)
        median1 = round(event1[event1_col].median(),2)
        median2 = round(event2[event2_col].median(),2)
        std1 = round(event1[event1_col].std(),2)
        std2 = round(event2[event2_col].std(),2)
    
        lists = list(kwargs.items())
        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f'Event A: {lists[0]}\n')
            file.write(f'Event B: {lists[1]}\n')
            file.write(f"Probability of A: {Prob_A}\n")
            file.write(f"Probability of B: {Prob_B}\n")
            file.write(f"Mean of A: {mean1}\n")
            file.write(f"Mean of B: {mean2}\n")
            file.write(f"Median of A: {median1}\n")
            file.write(f"Median of B: {median2}\n")
            file.write(f"Standard Deviation of A: {std1}\n")
            file.write(f"Standard Deviation of B: {std2}\n")
            file.write(f"Probability of A given B: {prob_A_given_B}\n")
            file.write(f"Probability of B given A: {prob_B_Given_A}\n")
            file.write(f"Probability of A and B: {prob_A_and_B}\n")

        

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
        
        def get_position_vector(index):
            position_vector = vectors[index]
            return position_vector
        
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        unit_vector1 = vector1 / magnitude1 if magnitude1 != 0 else vector1
        unit_vector2 = vector2 / magnitude2 if magnitude2 != 0 else vector2

        dot_product = np.dot(vector1, vector2)
        
        projection1 = (dot_product / magnitude2**2) * vector2 if magnitude2 != 0 else vector2
        projection2 = (dot_product / magnitude1**2) * vector1 if magnitude1 != 0 else vector1
        angle_cal = np.arccos(dot_product / (magnitude1 * magnitude2)) if magnitude1 != 0 and magnitude2 != 0 else 0
        angle = np.degrees(angle_cal)
        
        
        def orthogonal():
            nonlocal vector1, vector2
            return np.isclose(np.dot(vector1,vector2), 0)
        
        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f"Vector 1: {col1}\n")
            file.write(f"Vector 2: {col2}\n")
            file.write(f"Vector 1: {vector1}\n")
            file.write(f"Vector 2: {vector2}\n")
            file.write(f"Position Vector: {get_position_vector(0)}\n")
            file.write(f"Magnitude of Vector 1: {magnitude1}\n")
            file.write(f"Magnitude of Vector 2: {magnitude2}\n")
            file.write(f"Unit Vector 1: {unit_vector1}\n")
            file.write(f"Unit Vector 2: {unit_vector2}\n")
            file.write(f"Dot Product: {dot_product}\n")
            file.write(f"Projection of Vector 1 on Vector 2: {projection1}\n")
            file.write(f"Projection of Vector 2 on Vector 1: {projection2}\n")
            file.write(f"Angle between Vectors (degrees): {angle}\n")
            file.write(f"Are the vectors orthogonal? {'Yes' if orthogonal() else 'No'}\n")
    

            
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
    count_test = test.Probability(Party_size=2, Arrival_Time = lambda x: x>= 12)
    test.Vector_Ops('Party_size', 'Bill_Amount')
