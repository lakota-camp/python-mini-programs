import csv, warnings
import sklearn.cluster, sklearn.metrics
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

'''
    This program performs K-Means clustering analysis on a dataset of wholesale customer data. 
    It includes functions to load and process the data from a CSV file, create and evaluate 
    K-Means models, calculate silhouette scores to determine the optimal number of clusters, 
    and visualize the clusters based on user-selected variables.
    
    Functions:
        load_csv(filename: str) -> list:
        process_data(data: list) -> dict:
            Processes and structures data for the K-Means model.
        create_k_means_model(x, n_clusters=3, random_state=None):
            Creates and fits a K-Means model to the data.
        calc_silhouette_scores(x) -> dict:
            Calculates silhouette `scores for different numbers of clusters to find the optimal number.
        display_clusters(x_one: list, x_two: list, labels):
            Displays a scatter plot of the clusters based on two user-selected variables.
        prompt_user() -> list:
            Prompts the user to select two variables for visualization.
        main():
            Main function to load data, process it, run K-Means analysis, and visualize the clusters.
'''

def load_csv(filename: str) -> list:
    """
    Loads CSV file and returns data excluding first line containing column names.
    Args:
        filename (str): filepath of csv

    Returns:
        list: data from csv as a list
    """
    try:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
            return data
    except FileNotFoundError as e:
        print(f"Error. File {filename} not found")
        print(e)
        return None

def process_data(data: list) -> dict:
    """
    Process and structure data for model with the given format:
    
        {
            "x": x,
            "x_lists": {
                "fresh": fresh_list,
                "milk": milk_list,
                "grocery": grocery_list,
                "detergents_paper": detergents_paper_list,
                "delicassen": delicassen_list
            }
        }
    
    x: 2D list of x variables
    x_list: dictionary of x variables as lists
    
    Args:
        data (list): raw data to be processed

    Returns:
        dict: structured data
    """
    x = []
    
    fresh_list = []
    milk_list = []
    grocery_list = []
    frozen_list = []
    detergents_paper_list = []
    delicassen_list = []
    
    if data:
        for line in data:
            fresh = int(line[2])
            milk = int(line[3])
            grocery = int(line[4])
            frozen = int(line[5])
            detergents_paper = int(line[6])
            delicassen = int(line[7])
            
            fresh_list.append(fresh)
            milk_list.append(milk)
            grocery_list.append(grocery)
            frozen_list.append(frozen)
            detergents_paper_list.append(detergents_paper)
            delicassen_list.append(delicassen)
            
            x.append([fresh, milk, grocery, frozen, detergents_paper, delicassen])
        
        return {
            "x": x,
            "x_lists": {
                "fresh": fresh_list,
                "milk": milk_list,
                "grocery": grocery_list,
                "detergents_paper": detergents_paper_list,
                "delicassen": delicassen_list
            }
        }
        
    else:
        print("Error. No data to process.")
        return None
        

def create_k_means_model(x, n_clusters=3, random_state=None):
    """
    Creates K Means model
    Args:
        x: 2D list of x variables
        n_clusters (int, optional): number of clusters to create. Defaults to 3.
        random_state (int, optional): randomness of output. Defaults to None.

    Returns:
        tuple: A tuple containing the cluster centers and the labels of each point.
    """
    try:
        k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
        k_means = k_means.fit(x)
        return k_means.cluster_centers_, k_means.labels_
    except Exception as e:
        print("Error creating K Means Model.")
        print(e)
        
    


def calc_silhouette_scores(x) -> dict:
    """
    Calculates silhouette scores and determines the best highest score and corresponding K Value.
    Args:
        x: 2D List of x variables

    Returns:
        dict: best silhouette and corresponding K Value
    """
    max_sil = 0
    k_value = 0
    
    for k in range(2, 11, 1):
        k_means = sklearn.cluster.KMeans(n_clusters=k)
        k_means = k_means.fit(x)
        sil = sklearn.metrics.silhouette_score(x, k_means.labels_)
        print(f"K = {k} silhouette score: {sil}")
        if sil > max_sil:
            max_sil = sil
            k_value = k
            
    print(f"\nOptimal number of clusters: {k_value}")
    
    return {
        "best_sil": float(max_sil),
        "k_value": k_value
    }

def display_clusters(x_one: list, x_two: list, labels):
    """
    Displays clusters from two list of x variables 
    
    Args:
        x_one (list): First list of x variables.
        x_two (list): Second list of x variables.
        labels (list): List of cluster labels corresponding to the x variables.
    """
    colors = []
    # Generate colors for clusters
    for label in labels:
        if label == 0:
            colors.append("red")
        elif label == 1:
            colors.append("blue")
        elif label == 2:
            colors.append("green")
        elif label == 3:
            colors.append("purple")
        elif label == 4:
            colors.append("yellow")
        else:
            colors.append("orange")
    
    # Display data with corresponding cluster colors
    plt.scatter(x_one, x_two, color=colors)
    plt.show

def prompt_user() -> list:
    """
    Prompts user for inputs for two sets of x variables to display as clusters on graph.
    Returns:
        list: two lists of x variables
    """
    while True:
        # List of valid inputs
        valid_inputs = ["fresh", "milk", "grocery", "frozen", "detergents_paper", "delicassen"]
        print(f"Inputs to choose from: {', '.join(input for input in valid_inputs)}")
        
        # Prompt user for input one
        var_one = input("Enter first variable to visualize (enter e to exit): ").strip().lower()
        # Exit function if 'e' entered
        if var_one == "e" or var_two == "e":
            return None
        # Check if input is valid from list
        if var_one not in valid_inputs:
            print(f"Invalid input. The input {var_one} is not a valid option.")
            print(f"Valid inputs are {', '.join(input for input in valid_inputs)}")
            continue
        
        # Prompt user for input two
        var_two = input("Enter second variable to visualize (enter e to exit): ").strip().lower()
        # Exit function if 'e' entered
        if var_two == "e":
            return None
        # Check if input is valid from list
        if var_two not in valid_inputs:
            print(f"Invalid input. The input {var_one} is not a valid option.")
            print(f"Valid inputs are {', '.join(input for input in valid_inputs)}")
            continue
        break
    return [var_one, var_two]
            
def main():
   # Load and process data 
   data = load_csv("./Wholesale customers data.csv")
   
   # Ensure data is present before processing
   if data:
       data = process_data(data)
   else:
       print("No data from csv.")
   
   # Ensure processed data is present before indexing x_list
   if data:
       x_lists = data["x_lists"]
   else:
       print("No processed data.")

   # Ensure list of x data is present before running K Means model
   if x_lists:
       # Run K-Means analysis   
       print("Running K-Means analysis...\n")
       # Calculate best silhouette score
       best_sil_scores = calc_silhouette_scores(x=data["x"])
       print() # For formatting purpose
       # Get labels and cluster centers from K Means model
       cluster_centers, labels = create_k_means_model(x=data["x"], n_clusters=best_sil_scores["k_value"], random_state=0)
       # Call prompt_user() function to get user input to display clusters   
       inputs = prompt_user()
       
       # Ensure inputs are present before displaying clusters graph
       if inputs:
           display_clusters(x_one=x_lists[inputs[0]], x_two=x_lists[inputs[1]], labels=labels)
   else:
       print("No data to run K-Means analysis.")
       
if __name__ == "__main__":
    main()

    