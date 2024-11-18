import sklearn.metrics, sklearn.model_selection, sklearn.tree, sklearn.neighbors, sklearn.neural_network
import csv, nltk, json, textblob
import matplotlib.pyplot as plt
nltk.download("stopwords")
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

# TODO : Plot data for each x feature to see trends

def import_csv_data(filename: str) -> list:
    print("\nImporting csv data...")
    with open(filename, "r", encoding="latin-1") as file:
        reader = csv.reader(file)
        next(reader)
        data = list(reader)
        print("Import complete!\n")
        return data

def format_data(data: list) -> dict:
    print("Data received to process and format....")

    title_list = []
    blurb_list = []
    requested_list = []
    description_list = []
    y = []
    for line in data:
        title = line[1]
        blurb = line[2]
        status = line[3]
        requested = line[4]
        description = line[5]
        if status == "Funding Successful":
            y.append(1)
        else:
            y.append(0)
        
        title_list.append(title)
        blurb_list.append(blurb)
        requested_list.append([float(requested)])
        description_list.append(description)
        
    print("Data successfully processed and formatted.\n")
    
    return {
        "title": title_list,
        "description": description_list,
        "blurb": blurb_list,
        "requested": requested_list,
        "y": y,            
    }

def vectorize_text(text: list, max_features=3000) -> list:
    print("Vectorizing text data...")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords.words("english"), max_features=max_features) # Stop word argument
    vectors = vectorizer.fit_transform(text)
    x = vectors.toarray()
    print("Vectorizing complete!\n")
    
    return x

def plot_data(data, x_feature):
    if x_feature == "description" or x_feature == "blurb" or x_feature == "title":
        x = vectorize_text(data[x_feature])
        # Consider an aggregate plot for text data
        plt.hist(x.sum(axis=1), bins=30, color='blue', alpha=0.7)
        plt.xlabel(f"Sum of vectorized {x_feature} features")
        plt.ylabel("Frequency")
    elif x_feature == "requested":
        x = [r[0] for r in data["requested"]]  # Flatten 'requested' list of lists
        y = data["y"]
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel(f"{x_feature}")
        plt.ylabel("Status")
    
    plt.title(f"{x_feature} vs Status")
    plt.show()
    
def main():
    filename = './kickstarter_data_full.csv'
    data = import_csv_data(filename=filename)
    data_formatted = format_data(data) 
    plot_data(data_formatted, "requested")

main()