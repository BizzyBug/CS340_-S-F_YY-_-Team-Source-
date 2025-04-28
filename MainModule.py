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

import ParentChild1
import ParentChild2


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here
def f():
    samples = ParentChild1.CustomerAnalytics('data.csv')
    df = samples.read_csv()
    samples.plot_violin()
    samples.plot_box()
    samples.plot_scatter('Party_size', 'Bill_Amount')
    result1 = samples.advanced_query({'Bill_Amount':('>=', 100)})
    print(result1)
    samples.plot_histograms()
    samples.plot_line()
    result2 = samples.query_data('Party_size',2)
    print(result2)

    log_test = ParentChild1.CustomerAnalytics('badfile.lol')
    test_df = log_test.read_csv()

    array = ParentChild2.Inventory.readData('Input/data.csv')
    ParentChild2.Inventory.convert_to_df(array)
    
    test = ParentChild2.InventoryAnalytics('test_pickle.pkl')
    testResult = test.readPickle()

    events, conditions = test.get_events(Party_size=2, Arrival_Time = 'eval:>= 12')
    
    probabilities = test.get_probabilities(events[0], events[1], conditions[0], conditions[1])

    test.get_all_column_stats(testResult)

    test.get_events(Price=lambda x: x > 50, Quantity="eval:< 100")
    test.Vector_Ops('Party_size', 'Bill_Amount')
    test.unique_vals('Party_size', 'Arrival_Time', 'Duration_of_Stay', 'Bill_Amount')

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    #print(f"\"{module_name_gl}\" module begins.")
    
    f()


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    #print(f"\"{module_name_gl}\" module begins.")
    
    f()
