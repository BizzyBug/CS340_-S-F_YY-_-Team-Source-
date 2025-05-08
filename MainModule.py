#Version: v0.1
#Date Last Updated: 1-12-2025

#DELETE THIS comment
#  Follow the coding standards listed in coding_standards.pdf 
#  Delete the sections in this template if not used

#%% MODULE BEGINS
module_name_gl = 'write the file name w/o file extension'

'''
Version: <***>

Description:
    <***>

Authors:
    <***>

Date Created     :  <***>
Date Last Updated:  <***>

Doc:
    <***>

Notes:
    cannot figure why the module ParentChild1 cannot be found. 
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os

import CustomerData
import InventoryData
import UI


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here
def f():
    samples = CustomerData.CustomerAnalytics(UI.UserInput.getInput())
    samples.plot_violin()
    samples.plot_box()
    samples.plot_scatter('Party_size', 'Bill_Amount')
    result1 = samples.advanced_query({'Bill_Amount':('>=', 100)})
    print(result1)
    samples.plot_histograms()
    samples.plot_line()
    result2 = samples.query_data('Party_size', 2)
    print(result2)

    test = InventoryData.InventoryAnalytics('Input/data.csv')

    test.plot_box()
    test.plot_scatter('Party_size', 'Bill_Amount')
    test.plot_violin()

    events, conditions = test.get_events(Party_size=2, Arrival_Time = 'eval:>= 12')
    
    test.get_probabilities(events[0], events[1], conditions[0], conditions[1])

    test.get_all_column_stats()

    test.get_events(Price=lambda x: x > 50, Quantity="eval:< 100")
    
    test.Vector_Ops('Party_size', 'Bill_Amount')
    test.unique_vals('Party_size', 'Arrival_Time', 'Duration_of_Stay', 'Bill_Amount')

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    #print(f"\"{module_name_gl}\" module begins.")
    
    f()

