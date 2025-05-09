module_name_gl = 'write the file name w/o file extension'

'''
Version: <1.0>

Description:
    Import resturant data from a csv file and get a visulization of the data. 
    See if the resturant has the potential to handle more expenses; in the form of labor.
    
Authors:
    <Henry Hazlett, Elizabeth Valenti, Davidson Rock>

Date Created     :  4-6-2024
Date Last Updated:  5-8-2025

Doc:

Notes:
   
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime as dt
import UI


outputFilepath_gl = 'output'


class CustomerTrends:
    def __init__(self, df):
        self.__df = df
    #

    #DESCRIPTION plot_histograms()
    #Purpose: Generate histograms displaying the data from the dataframe instance variable.
    #Input: None
    #Output: None
    def plot_histograms(self):
        #For every column in the Dataframe, perform the necessary methods to generate the plot.
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            self.__df[col].hist()
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            filename = f'Histogram Plot of {col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #
    
    #DESCRIPTION plot_line()
    #Purpose: Generate line plots displaying the data from the dataframe instance variable.
    #Input: None
    #Output: None
    def plot_line(self):
        #For every column in the Dataframe, perform the necessary methods to generate the plot.
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            plt.plot(self.__df[col])
            plt.title(f'Line Plot of {col}')
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.grid(True)
            filename = f'Line Plot of {col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #

    #DESCRIPTION query_data()
    #Purpose: Return all the rows of the Dataframe where the value of a certain column is equal to a certain value.
    #Input: string column, string value
    #Output: Dataframe self.__df[self.__df[column] == value]
    def query_data(self, column, value):
        return self.__df[self.__df[column] == value]
    #

class CustomerAnalytics(CustomerTrends): 
   
    def __init__(self, filename = 'data.csv'):
        self.filename = filename
        super().__init__(self.read_csv())
    #
    
    #DESCRIPTION read_csv()
    #Purpose: Read a given .csv file and store it in a Dataframe instance variable.
    #Input: None
    #Output: Dataframe self.__df
    def read_csv(self):
        #Attempt to open the file and throw an error if file not found.
        try:
            with open(f'Input/{self.filename}', 'r') as file:
                self.__df = pd.read_csv(file)
            #
        except FileNotFoundError:
            with open(f'Doc/log_file.txt', 'a') as file:
                file.write(f"File {self.filename} not found in DataAnalysis.read_csv() function.\n")
                file.write(f'{dt.datetime.now()}\n\n')
            #
            return None
        #
        
        return self.__df
    #
   
    #DESCRIPTION plot_violin()
    #Purpose: Generate violin plots displaying the data from the dataframe instance variable.
    #Input: None
    #Output: None
    def plot_violin(self):
        #For every column in the Dataframe, perform the necessary methods to generate the plot.
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.violinplot(y=self.__df[col])
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
        for col in self.__df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.boxplot(y=self.__df[col])
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
        if x_col in self.__df.columns and y_col in self.__df.columns:
            plt.figure()
            sns.scatterplot(data=self.__df, x=x_col, y=y_col)
            plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            filename = f'Scatter Plot {x_col} vs {y_col}.png'
            filepath = os.path.join(outputFilepath_gl,filename)
            plt.savefig(filepath)
            plt.close()
        #
    #
            
    #DESCRIPTION advanced_query()
    #Purpose: Return all the rows of the Dataframe where the value of a certain column satisfies the given condition.
    #Input: Dictionary conditions
    #Output: Dataframe self.__df[task]
    def advanced_query(self, conditions: dict):
        task = pd.Series([True] * len(self.__df))

        #For each column specified in the coditions, search for values which satisfy the specified condition.
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
            #
        #
        return self.__df[task]
    #
#
           

if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.")
    
    samples = CustomerAnalytics(UI.UserInput.getInput())
    samples.plot_violin()
    samples.plot_box()
    samples.plot_scatter('Party_size', 'Bill_Amount')
    result1 = samples.advanced_query({'Bill_Amount':('>=', 100)})
    print(result1, flush=True)
    samples.plot_histograms()
    samples.plot_line()
    result2 = samples.query_data('Party_size', 2)
    print(result2, flush=True)
#
