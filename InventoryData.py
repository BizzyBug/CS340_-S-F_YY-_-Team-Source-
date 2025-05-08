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
    Cannot do commits from visual studio, there is a discrepencey between my version and the GitHub one. Have been copy and pasting everything to and from GitHub. 
    Need to see if it runs on Henery's version. I am still missing some requirments like the *args, and converting a mxn numpy array into a data frame.
    I think everything so far is in order, still trying to relate this to a goal in a real life resturant situation.
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pck
import datetime as dt
import numpy as np
import math

#custom imports


#other imports


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here

outputFilepath = 'output'

#Class definitions Start Here
class Inventory():
    def readData(self):
        try:
                my_data = np.genfromtxt(self.filename, delimiter=',')
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File {self.filename} not found in InventoryAnalysis.readPickle() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None
        
        return my_data
    
    def convert_to_df(self, array):
        new_df = pd.DataFrame(array[1:], columns=['Party_size', 'Arrival_Time', 'Duration_of_Stay', 'Bill_Amount'])
        new_df.to_pickle('df_pickle.pkl')

    def __init__(self, filename):
        self.filename = filename
        self.convert_to_df(self.readData())
        try:
            self.df = pd.read_pickle('df_pickle.pkl')
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File df_pickle.pkl not found in Inventory.__init__() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')


class InventoryAnalytics(Inventory):
    def __init__(self, filename):
        super().__init__(filename)
          
    def get_events(self, **kwargs):
        event1 = None
        condition_str = ''
        event_list = []
        condition_list = []
        '''
        joint_df = pd.Series(True, index=df.index)
        '''
        for col, condition in kwargs.items():
            if col not in self.df.columns:
                with open(f'Doc/log_file.txt', 'a') as file:
                    file.write(f"Column {col} not found in DataFrame.\n")
                    file.write(f'{dt.datetime.now()}\n\n')
                continue

            if callable(condition):
                event1 = condition(self.df[col])
                condition_str = 'placeholder'
            elif isinstance(condition, str) and condition.startswith("eval:"):
                try:
                    expr = condition.replace("eval:", "")
                    event1 = eval(f"self.df[col] {expr}")    
                    condition_str = expr 
                except Exception as e:
                    with open(f'Doc/log_file.txt', 'a') as file:
                        file.write(f"Eval error for column {col}: {e}\n")
                        file.write(f'{dt.datetime.now()}\n\n')
                    return None
            else:
                event1 = self.df[col] == condition
                condition_str = f'= {condition}'

            event_list.append(event1)
            condition_list.append(condition_str)
        return event_list, condition_list
    
    def plot_violin(self):
        for col in self.df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.violinplot(y=self.df[col])
            plt.title(f'Violin Plot of {col}')
            filename = f'Violin Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
            plt.close()
            
    def plot_box(self):
        for col in self.df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.boxplot(y=self.df[col])
            plt.title(f'Box Plot of {col}')
            filename = f'Box Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
            plt.close()

    def plot_scatter(self, x_col, y_col):
        if x_col in self.df.columns and y_col in self.df.columns:
            plt.figure()
            sns.scatterplot(data=self.df, x=x_col, y=y_col)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            filename = f'Scatter Plot {x_col} vs {y_col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
            plt.close()

    def get_probabilities(self, event1, event2, *args):

        prob_event1 = round(event1.sum() / len(self.df), 2)
        prob_event2 = round(event2.sum() / len(self.df), 2)

        event1_list = event1.to_list()
        event2_list = event2.to_list()

        joint_events = [False] * len(event1)

        for i, val in enumerate(event1_list):
            if val and event2_list[i]:
                joint_events[i] = True

        joint_events = pd.Series(joint_events)

        prob_event1_and_event2 = round(joint_events.sum() / len(self.df), 2)
        
        prob_event1_given_event2 = round(prob_event1_and_event2 / prob_event2, 2)
        prob_event2_given_event1 = round(prob_event1_and_event2 / prob_event1, 2)
        
        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f'Event A: {event1.name} {args[0]}\n')
            file.write(f'Event B: {event2.name} {args[1]}\n')
            file.write(f"Probability of A: {prob_event1}\n")
            file.write(f"Probability of B: {prob_event2}\n")
            file.write(f"Probability of A given B: {prob_event1_given_event2}\n")
            file.write(f"Probability of B given A: {prob_event2_given_event1}\n")
            file.write(f"Probability of A and B: {prob_event1_and_event2}\n")

        
    def get_data_stats(self, col):
        mean = round(self.df[col].mean(numeric_only = True),2)
        median = round(self.df[col].median(),2)
        std = round(self.df[col].std(),2)

        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f'{col} mean: {mean}\n')
            file.write(f'{col} median: {median}\n')
            file.write(f'{col} std: {std}\n')

    def get_all_column_stats(self):
        for col in self.df:
            self.get_data_stats(col)

    def Vector_Ops(self, col1, col2):

        if col1 not in self.df.columns or col2 not in self.df.columns:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"Column {col1} or {col2} not found in DataFrame.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None
        
        vector1 = self.df[col1].values
        vector2 = self.df[col2].values
        vectors = self.df[[col1, col2]].values
        
        def get_position_vector(index):
            nonlocal vectors
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

    
    def unique_vals(self, col1, col2, col3, col4):

        if col1 not in self.df.columns or col2 not in self.df.columns or col3 not in self.df.columns or col4 not in self.df.columns:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"Column {col1}, {col2}, {col3} or {col4} not found in DataFrame.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None

        
        categories1 = self.df[col1].unique()
        categories2 = self.df[col2].unique()
        categories3 = self.df[col3].unique()
        categories4 = self.df[col4].unique()

        combinations = self.df.drop_duplicates(subset=[col1,col2, col3, col4])
        permutations = math.factorial(len(combinations))

        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f"Categories in {col1}: {categories1}\n")
            file.write(f"Categories in {col2}: {categories2}\n")
            file.write(f"Categories in {col3}: {categories3}\n")
            file.write(f"Categories in {col4}: {categories4}\n")
            file.write(f"Number of unique combinations: {len(combinations)}\n")
            file.write(f"Number of unique permutations: {permutations}\n")






        


#Function definitions Start Here
def main():
    pass
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.")
    
    test = InventoryAnalytics('Input/data.csv')

    test.plot_box()
    test.plot_scatter('Party_size', 'Bill_Amount')
    test.plot_violin()

    events, conditions = test.get_events(Party_size=2, Arrival_Time = 'eval:>= 12')
    
    probabilities = test.get_probabilities(events[0], events[1], conditions[0], conditions[1])

    test.get_all_column_stats()

    test.get_events(Price=lambda x: x > 50, Quantity="eval:< 100")

    test.Vector_Ops('Party_size', 'Bill_Amount')
    test.unique_vals('Party_size', 'Arrival_Time', 'Duration_of_Stay', 'Bill_Amount')



