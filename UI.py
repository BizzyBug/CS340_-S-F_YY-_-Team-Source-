module_name_gl = 'write the file name w/o file extension'

'''
Version: <1.0>

Description:
    Define user input functionality for use in other modules.

Authors:
    <Henry Hazlett, Elizabeth Valenti, Davidson Rock>

Date Created     :  4-6-2024
Date Last Updated:  5-8-2025

Doc:

Notes:

'''

if __name__ == "__main__":
   import os
#

#DESCRIPTION UserInput()
#Purpose: Get input from the user via the keyboard and include a prompt if necessary.
#Input: None
#Output: string input
class UserInput():
    def getInput(prompt=''):
        return input(prompt)
    #
#

def main():
    pass  
#

if __name__ == "__main__":
    
    print(f"\"{module_name_gl}\" module begins.", flush=True)
    
    main()
#
