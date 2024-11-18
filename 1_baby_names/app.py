import csv
"""
Baby Name Analyzer Program

This program allows users to analyze historical baby name data stored in a CSV file.
It provides two functionalities:
1. Name Comparison: Compare the popularity of two names based on their 
   cumulative frequency in the dataset.
2. Maximum Popularity: Determine the year in which a specific name had the 
   highest frequency.

The program expects a CSV file with the following structure:
- Column 0: Name
- Column 1: Year
- Column 2: Frequency (integer)

The user interacts with the program through a simple text-based menu.
"""

def name_compare(name_one, name_two, csv_file):
    """Function that compares two names and display which name was more popular.

    Args:
        name_one (str): name one
        name_two (str): name two
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        
        sum_name_one = 0
        sum_name_two = 0
        
        # Loop over each line in CSV file.
        for line in reader:
            # Condition to check if line matches first name
            if line[0].lower() == name_one.lower().strip():
                sum_name_one += int(line[2])
            # Condition to check if line matches second name    
            elif line[0].lower() == name_two.lower().strip():
                sum_name_two += int(line[2])
            
    if (sum_name_one == 0) and (sum_name_two == 0):
        print(f"Name {name_one.strip().capitalize()} and {name_two.strip().capitalize()} did not occur")
    else:    
        if sum_name_one > sum_name_two:
            print(f"{name_one.strip().capitalize()} was more popular than {name_two.strip().capitalize()} ({sum_name_one} to {sum_name_two}) !")
            
        elif sum_name_two > sum_name_one:

            print(f"{name_two.strip().capitalize()} was more popular than {name_one.strip().capitalize()} ({sum_name_two} to {sum_name_one}) !")
            
        else:
            print(f"{name_one} and {name_two} were equally popular  ({sum_name_two} to {sum_name_one}) !")
            

def max_popularity(name, csv_file):
    """Function that gets a name as input and prints which year that name had the most frequency.

    Args:
        name (str): name to search most frequent year
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
                
        year_freq = {}
        freq_list = []
        # Loop over each line in CSV file.
        for line in reader:
            # Check if current line name (index 0) matches input name
            if line[0].lower() == name.strip().lower():
                # If match, append the frequency (index 2) to the freq_list
                freq_list.append(int(line[2]))
                # If match, add key value pair to dictionary -> 'year' : freq
                year_freq[line[1]] = int(line[2])
                
    if len(year_freq) == 0:
        print('Name not found')
        
    else:
        # Sort dict by frequency (index 1) using lambda function to set key
        sorted_by_freq = dict(sorted(year_freq.items(),  key=lambda item: item[1]))
        
        year = list(sorted_by_freq)[len(sorted_by_freq) - 1]
        freq = list(sorted_by_freq.values())[len(sorted_by_freq) - 1]
        
        print(f'{name.strip().capitalize()} was the most popular in {year} with a frequency of {freq}')
    
def main():
    print('Welcome to the baby name analyzer!')
    
    csv_file = '/Users/lakotacamp/files/dev/proj/github/python-mini-programs/hw-1/usa_baby_names.csv'
    
    while True:   
        analysis_type = input('What analysis would you like to run (name comparison/maximum popularity)? ')
        
        if analysis_type.strip() == 'name comparison':
            name_one = input('Enter the first name to analyze: ')
            name_two = input('Enter the second name to analyze: ')
            name_compare(name_one, name_two, csv_file)
            
        elif analysis_type.strip() == 'maximum popularity':
            name = input('Enter the name to analyze: ')
            
            max_popularity(name, csv_file)
            
        else:
            print('Sorry, that type of analysis is not supported.')
        
        while True:
            run = input('Would you like to run another analysis (yes/no)? ')
            if (run.lower() == 'yes') or (run.lower() == 'no'):
                break
            else:
                print("Please enter 'yes' or 'no'.")
                continue
        
        if run.lower() == 'yes':
            continue

        elif run.lower() == 'no':
            break
    print('Thank you for using the baby name analyzer! Have a good day!')        
    
if __name__ == "__main__":
    main()