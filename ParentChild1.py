#Version: v0.1
#Date Last Updated: 1-12-2025

module_name_gl = 'write the file name w/o file extension'

'''
Versiion: v0.1

Description:
    Import resturant data from a csv file and get a visulization of the data. 
    See if the resturant has the potential to handle more expenses; in the form of labor.
    
Authors:
    Elizabeth, Henry, and Davidson

Date Created     :  4-6-2024
Date Last Updated:  4-14-2025

Doc:
    <***>

Notes:
   
'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime as dt
import UI


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Global declarations Start Here

outputFilepath = 'output'


class CustomerTrends:

    def __init__(self, df):
        self.__df = df

    def plot_histograms(self):
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            self.__df[col].hist()
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            filename = f'Histogram Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
        
    def plot_line(self):
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            plt.plot(self.__df[col])
            plt.title(f'Line Plot of {col}')
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.grid(True)
            filename = f'Line Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)

    def query_data(self, column, value):
        return self.__df[self.__df[column] == value]

class CustomerAnalytics(CustomerTrends): 
   
    def __init__(self, filename):
        self.filename = filename
        super().__init__(self.read_csv())
    
    def read_csv(self):
        try:
            with open(f'Input/{self.filename}', 'r') as file:
                self.__df = pd.read_csv(file)
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File {self.filename} not found in DataAnalysis.read_csv() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            return None
        
        return self.__df
   

    def plot_violin(self):
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.violinplot(y=self.__df[col])
            plt.title(f'Violin Plot of {col}')
            filename = f'Violin Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
            
    def plot_box(self):
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.boxplot(y=self.__df[col])
            plt.title(f'Box Plot of {col}')
            filename = f'Box Plot of {col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)

    def plot_scatter(self, x_col, y_col):
        if x_col in self.__df.columns and y_col in self.__df.columns:
            plt.figure()
            sns.scatterplot(data=self.__df, x=x_col, y=y_col)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            filename = f'Scatter Plot {x_col} vs {y_col}.png'
            filepath = os.path.join(outputFilepath,filename)
            plt.savefig(filepath)
            
    def advanced_query(self, conditions: dict):
        task = pd.Series([True] * len(self.__df))
        for col, (op, val) in conditions.items():
            if op == '==':
                task &= self.__df[col] == val
            elif op == '>=':
                task &= self.__df[col] >= val
            elif op == '<=':
                task &= self.__df[col] <= val
            elif op == '>':
                task &= self.__df[col] > val
            elif op == '<':
                task &= self.__df[col] < val
            elif op == '!=':
                task &= self.__df[col] != val
            else:
                raise ValueError(f"Unsupported operator: {op}")
        return self.__df[task]
           


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.")
    
    samples = CustomerAnalytics(UI.UserInput.getInput())
    samples.plot_violin()
    samples.plot_box()
    samples.plot_scatter('Party_size', 'Bill_Amount')
    result1 = samples.advanced_query({'Bill_Amount':('>=', 100)})
    print(result1)
    samples.plot_histograms()
    samples.plot_line()
    result2 = samples.query_data('Party_size', 2)
    print(result2)

    samples._CustomerAnalytics__df.to_pickle('test_pickle.pkl')
