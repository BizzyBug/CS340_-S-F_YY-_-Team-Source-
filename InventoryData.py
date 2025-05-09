module_name_gl = 'Parent-Child2'

'''
Version: <1.0>

Description:
    See if there is a specific time where the most orders are made and the income per
    hour is the highest.
    This will help us see if there is a specific time where we can have more employees.


Authors:
    <Henry Hazlett, Elizabeth Valenti, Davidson Rock>

Date Created     :  <4-12-2025>
Date Last Updated:  <5-8-2025>

Doc:

Notes:

'''

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pck
import datetime as dt
import numpy as np
import math


outputFilepath_gl = 'output'

class Inventory():
    #DESCRIPTION readData()
    #Purpose: Read a given .csv file and store it in a Numpy array.
    #Input: None
    #Output: npArray my_data
    def readData(self):
        #Attempt to open the file and throw an error if file not found.
        try:
                my_data = np.genfromtxt(self.filename, delimiter=',')
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File {self.filename} not found in InventoryAnalysis.readPickle() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            #
            return None
        #
        
        return my_data
    #
    
    #DESCRIPTION convert_to_df()
    #Purpose: Convert a Numpy mxn array to a dataframe.
    #Input: npArray array
    #Output: None
    def convert_to_df(self, array):
        new_df = pd.DataFrame(array[1:], columns=['Party_size', 'Arrival_Time', 'Duration_of_Stay', 'Bill_Amount'])
        new_df.to_pickle('df_pickle.pkl')
    #

    def __init__(self, filename):
        self.filename = filename
        self.convert_to_df(self.readData())
        #Attempt to open the file and throw an error if file not found.
        try:
            self.df = pd.read_pickle('df_pickle.pkl')
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File df_pickle.pkl not found in Inventory.__init__() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            #
        #
    #
#


class InventoryAnalytics(Inventory):
    def __init__(self, filename):
        super().__init__(filename)
    #
          
    #DESCRIPTION getEvetns()
    #Purpose: Takes a number of conditions passed as kw arguments and applies them to the Dataframe instance variable. Returns the resulting dataframes in a list and a list of the conditions as strings.
    #Input: String or int or lambda **kwargs
    #Output: Dataframe[] event_list, String[] condition_list
    def get_events(self, **kwargs):
        event1 = None
        condition_str = ''
        event_list = []
        condition_list = []

        #For each column specified in the coditions, search for values which satisfy the specified condition.
        for col, condition in kwargs.items():
            #Check if column exists in Dataframe, if not skip that column.
            if col not in self.df.columns:
                with open(f'Doc/log_file.txt', 'a') as file:
                    file.write(f"Column {col} not found in DataFrame.\n")
                    file.write(f'{dt.datetime.now()}\n\n')
                #
                continue
            #

            #Check if the condition is a callable function and apply it to the column if so.
            if callable(condition):
                event1 = condition(self.df[col])
                condition_str = 'placeholder'
            #Check if the condition is a string starting with 'eval:' and apply the eval to the column if so.
            elif isinstance(condition, str) and condition.startswith("eval:"):
                try:
                    expr = condition.replace("eval:", "")
                    event1 = eval(f"self.df[col] {expr}")    
                    condition_str = expr 
                except Exception as e:
                    with open(f'Doc/log_file.txt', 'a') as file:
                        file.write(f"Eval error for column {col}: {e}\n")
                        file.write(f'{dt.datetime.now()}\n\n')
                    #
                    return None
                #
            else:
                event1 = self.df[col] == condition
                condition_str = f'= {condition}'
            #

            event_list.append(event1)
            condition_list.append(condition_str)
        #
        return event_list, condition_list
    #
    
    #DESCRIPTION plot_violin()
    #Purpose: Generate violin plots displaying the data from the dataframe instance variable.
    #Input: None
    #Output: None
    def plot_violin(self):
        #For every column in the Dataframe, perform the necessary methods to generate the plot.
        for col in self.df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.violinplot(y=self.df[col])
            plt.title(f'Violin Plot of {col}')
            filename = f'Violin Plot of {col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #

    #DESCRIPTION plot_box()
    #Purpose: Generate box plots displaying the data from the dataframe instance variable.
    #Input: None
    #Output: None    
    def plot_box(self):
        #For every column in the Dataframe, perform the necessary methods to generate the plot.
        for col in self.df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.boxplot(y=self.df[col])
            plt.title(f'Box Plot of {col}')
            filename = f'Box Plot of {col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #

    #DESCRIPTION plot_scatter()
    #Purpose: Generate a scatter plot displaying the data from two columns of the dataframe instance variable.
    #Input: None
    #Output: None
    def plot_scatter(self, x_col, y_col):
        #Check if both columns exist in Dataframe and if so, perform the necessary methods to generate the plot.
        if x_col in self.df.columns and y_col in self.df.columns:
            plt.figure()
            sns.scatterplot(data=self.df, x=x_col, y=y_col)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            filename = f'Scatter Plot {x_col} vs {y_col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #

    #DESCRIPTION get_probabilities()
    #Purpose: Takes two sets of possible rows from a Dataframe along with the conditions applied to derive them and calculates the probabilities of those outcomes from the entire sample.
    #Input: Dataframe column event1, Dataframe column event2, String *args
    #Output: None
    def get_probabilities(self, event1, event2, *args):

        prob_event1 = round(event1.sum() / len(self.df), 2)
        prob_event2 = round(event2.sum() / len(self.df), 2)

        event1_list = event1.to_list()
        event2_list = event2.to_list()

        joint_events = [False for i in event1]

        #Generate a list of Booleans which is True at every index where both event1_list and event2_list are True.
        for i, val in enumerate(event1_list):
            if val and event2_list[i]:
                joint_events[i] = True
            #
        #

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
        #
    #

    #DESCRIPTION get_data_stats()
    #Purpose: Calculate the mean, median, and standard deviation of a given column of a Dataframe.
    #Input: Dataframe column col
    #Output: None
    def get_data_stats(self, col):
        mean = round(self.df[col].mean(numeric_only = True),2)
        median = round(self.df[col].median(),2)
        std = round(self.df[col].std(),2)

        with open(f'Output/WrittenOutput', 'a') as file:
            file.write(f'{col} mean: {mean}\n')
            file.write(f'{col} median: {median}\n')
            file.write(f'{col} std: {std}\n')
        #
    #

    #DESCRIPTION get_all_column_stats()
    #Purpose: Calculate the mean, median, and standard deviation of all columns in a Dataframe.
    #Input: Dataframe column col
    #Output: None
    def get_all_column_stats(self):
        #For every column in the Dataframe instance variable call the get_data_stats method.
        for col in self.df:
            self.get_data_stats(col)
        #
    #

    #DESCRIPTION Vector_Ops()
    #Purpose: Performs various vector operations using values from two given columns of the Dataframe instance variable.
    #Input: Dataframe column col1, Dataframe col2
    #Output: None
    def Vector_Ops(self, col1, col2):
        ##Check if column exists in Dataframe, if not abort function.
        if col1 not in self.df.columns or col2 not in self.df.columns:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"Column {col1} or {col2} not found in DataFrame.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            #
            return None
        #
        
        vector1 = self.df[col1].values
        vector2 = self.df[col2].values
        vectors = self.df[[col1, col2]].values
        
        #DESCRIPTION get_position_vector()
        #Purpose: Calculate the position vector of the vector at the given index in the list of vectors.
        #Input: int index
        #Output: Series position_vector
        def get_position_vector(index):
            nonlocal vectors
            position_vector = vectors[index]
            return position_vector
        #
        
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        unit_vector1 = vector1 / magnitude1 if magnitude1 != 0 else vector1
        unit_vector2 = vector2 / magnitude2 if magnitude2 != 0 else vector2

        dot_product = np.dot(vector1, vector2)
        
        projection1 = (dot_product / magnitude2**2) * vector2 if magnitude2 != 0 else vector2
        projection2 = (dot_product / magnitude1**2) * vector1 if magnitude1 != 0 else vector1
        angle_cal = np.arccos(dot_product / (magnitude1 * magnitude2)) if magnitude1 != 0 and magnitude2 != 0 else 0
        angle = np.degrees(angle_cal)
        
        #DESCRIPTION orthogonal()
        #Purpose: Determine if two vectors are orthogonal.
        #Input: None
        #Output: Boolean np.isclose(np.dot(vector1,vector2), 0)
        def orthogonal():
            nonlocal vector1, vector2
            return np.isclose(np.dot(vector1,vector2), 0)
        #
        
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
        #
    #
#

    #DESCRIPTION unique_vals()
    #Purpose: Finds every unique value in every column of the Dataframe instance variable. Also determines the total number of possible combinations and permutations.
    #Input: String col1, String col2, String col3, String col1
    #Output: None
    def unique_vals(self, col1, col2, col3, col4):
        ##Check if columns exist in Dataframe, if not abort function.
        if col1 not in self.df.columns or col2 not in self.df.columns or col3 not in self.df.columns or col4 not in self.df.columns:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"Column {col1}, {col2}, {col3} or {col4} not found in DataFrame.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            #
            return None
        #
        
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
        #
    #
#

def main():
    pass
#

if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.", flush=True)
    
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
#



