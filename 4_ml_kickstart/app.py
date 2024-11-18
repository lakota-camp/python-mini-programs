import sklearn.metrics, sklearn.model_selection, sklearn.tree, sklearn.neighbors, sklearn.neural_network
import csv, nltk, json, textblob
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

"""
This program reads in a csv file containing Kickstarter data containing various startup's
and trains machine learning models to predict the success (received funding) 
of a Kickstarter project based on various features.
"""

# Load NLTK stopwords to optimize text vectorization
try:
    nltk.download("stopwords")
    nltk.download("punkt", quiet=True)
    
except Exception as e:
    print(f"Error downloading NLTK stopwords: {e}")

# Maximum number of features to include in vectorized output
MAX_FEATURES = 2000

# Feature Configuration for data preprocessing
FEATURE_CONFIG = {
    "title": {
        "type": "text",
        "vectorize": True
    },
    "description": {
        "type": "text",
        "vectorize": True
    },
    "blurb": {
        "type": "text",
        "vectorize": True 
    },
    "requested": {
        "type": "numeric",
        "vectorize": False
    },
    "description_length": {
        "type": "numeric",
        "vectorize": False
    },
    "description_sentiment": {
        "type": "numeric",
        "vectorize": False
    },
    "title_description": {
        "type": "text",
        "vectorize": True
    }
}

def import_csv_data(filename: str) -> list:
    """
    Function to import csv data and return it as a list.
    Args:
        filename (str): Filename of csv file

    Returns:
        list: List of data from csv file
    """
    # print("\nImporting csv data...")
    try:
        with open(filename, "r", encoding="latin-1") as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
            # print("Import complete!\n")
            return data
    except FileNotFoundError as e:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def vectorize_text(text: list, max_features=MAX_FEATURES) -> list:
    """
    Vectorizes a list of text data by converting it into a matrix of token counts.
    Args:
        text (list): List of strings, where each string represents a document to be vectorized.
        max_features (int, optional): Maximum number of features to include in the vectorized output. Defaults to MAX_FEATURES.

    Returns:
        numpy.ndarray: 2D array of shape (n_samples, max_features) containing the vectorized output.
    """
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords.words("english"), max_features=max_features) # Stop word argument
    vectors = vectorizer.fit_transform(text)
    return vectors.toarray()

def get_sentiment(text: str) -> list:
    """
    Function to get sentiment scores for text data.
    Args:
        text (str): text data

    Returns:
        list: list of sentiment scores
    """
    sentiment_scores = []
    for i in text:
        blob = textblob.TextBlob(i)
        sentiment_scores.append([blob.polarity, blob.subjectivity])
    return sentiment_scores

# Preprocess data
def preprocess_data(data: list, feature: str) -> dict:
    """
    Function to preprocess data for machine learning model.
    Args:
        data (list): list of data from csv file

    Returns:
        tuple: Tuple containing x (features) and y (labels) for model training.
    """    

    y = []
    x = []
    
    # Loop through data and append to x and y
    for line in data:
        # Initialize inner list for x
        x_inner = []
        
        # Extract data
        title = line[1]
        blurb = line[2]
        status = line[3]
        requested = line[4]
        description = line[5]
        
        # Append data based on arg feature
        if feature == "description":
             x.append(description)
        elif feature == "requested":
            x_inner.append(requested)
            x.append(x_inner)
        elif feature == "blurb":
            x_inner.append(blurb)
        elif feature == "title":
            x_inner.append(title)
        elif feature == "description_length":
            x_inner.append(len(description))
            x.append(x_inner)
        elif feature == "description_sentiment":
            sentiment = get_sentiment([description])[0]
            x_inner.append(sentiment)
            x.append(x_inner)
        elif feature == "title_description":
            x.append(f"{title} {description}")
    
        y.append(1 if status == "Funding Successful" else 0)
        
    # Vectorize text data if configured
    if FEATURE_CONFIG[feature]["vectorize"]:
        x = vectorize_text(x)
    
    return x, y

# Initialize models
def initialize_models(**kwargs) -> dict:
    """
    Initializes machine learning models for training.

    Args:
        decision_tree_params (dict, optional): Parameters for DecisionTreeClassifier.
        knn_params (dict, optional): Parameters for KNeighborsClassifier.
        neural_network_params (dict, optional): Parameters for MLPClassifier.

    Returns:
        dict: Dictionary of ML models: {model_name: model_instance}
    """
    # Optional parameters for models
    decision_tree_params = kwargs.get("decision_tree_params", {})
    knn_params = kwargs.get("knn_params", {"n_neighbors": 7})
    neural_network_params = kwargs.get("neural_network_params", {})
    
    models = {
        "Decision_Tree": sklearn.tree.DecisionTreeClassifier(**decision_tree_params),
        "KNN": sklearn.neighbors.KNeighborsClassifier(**knn_params),
        "Neural_Network": sklearn.neural_network.MLPClassifier(**neural_network_params)
    }
    return models

# Train Model
def train_model(clf, x_train, y_train):
    """
    Trains a given machine learning model on the training data.

    Args:
        clf: ML classifier instance (e.g. DecisionTreeClassifier, KNeighborsClassifier, etc.)
        x_train: 2D array of shape (n_samples, n_features) of training data
        y_train: 1D array of shape (n_samples,) of training labels

    Returns:
        Trained ML classifier (clf) with fit data
    """
    clf = clf.fit(x_train, y_train)
    return clf

# Evaluate Model
def evaluate_model(clf, x_test, y_test):
    """
    Evaluates a given machine learning model on the test data and returns the accuracy.

    Args:
        clf: Trained ML classifier instance
        x_test: 2D array of shape (n_samples, n_features) of testing input data
        y_test: 1D array of shape (n_samples,) of target labels for testing data

    Returns:
        float: Accuracy of the model on the test data, between 0.0 and 1.0
    """
    predictions = clf.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    return accuracy

# Main function to run the program
def main():
    # Configuration
    test_feature = "description"
    test_size = 0.20
    n_neighbors = 15
    
    # Import data
    data = import_csv_data('./kickstarter_data_full.csv')
    
    # Preprocess data
    x, y = preprocess_data(data=data, feature=test_feature)
    
    # Initialize models
    models = initialize_models(n_neighbors=n_neighbors)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
    
    print(f"\nUsing feature: {test_feature}\n")
    
    # Loop through: Train, evaluate models, and display results
    for model_name, model in models.items():
        clf = train_model(clf=model, x_train=x_train, y_train=y_train)
        accuracy = evaluate_model(clf=clf, x_test=x_test, y_test=y_test)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("-"*50)

if __name__ == "__main__":
    main()
    