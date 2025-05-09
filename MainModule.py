module_name_gl = 'write the file name w/o file extension'

'''
Version: <1.0>

Description:
    <***>

Authors:
    <Henry Hazlett, Elizabeth Valenti, Davidson Rock>

Date Created     :  <4-6-2025>
Date Last Updated:  <5-8-2025>

Doc:

Notes:
     
'''

if __name__ == "__main__":
   import os
#

import CustomerData
import InventoryData
import UI

#DESCRIPTION f()
#Purpose: Perform all methods from imported classes.
#Input: None
#Output: None
def f():
    samples = CustomerData.CustomerAnalytics(UI.UserInput.getInput('Please enter filename:\n'))
    samples.plot_violin()
    samples.plot_box()
    samples.plot_scatter('Party_size', 'Bill_Amount')
    result1 = samples.advanced_query({'Bill_Amount':('>=', 100)})
    print(result1, flush=True)
    samples.plot_histograms()
    samples.plot_line()
    result2 = samples.query_data('Party_size', 2)
    print(result2, flush=True)

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
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    f()
#

