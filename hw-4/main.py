import sklearn.metrics, sklearn.model_selection, sklearn.tree, sklearn.neighbors, sklearn.neural_network
import csv, nltk, json, textblob
nltk.download("stopwords")
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

# FIXME: Figure out optimized way to run auto correct on text
# TODO: Test different x variables - title, blurb, requested, etc.
# TODO: Figure out best way to structure x variables
# TODO: Test combinations of different x variables - e.g. title and description, requested and descriptions, etc.
# TODO: Test description len as x variable
# TODO: Test x variable word count 
# TODO: Test x variable sentiment analysis
# TODO: DT - Experiment with parameters like max_depth, min_samples_split, and criterion
# TODO: KNN - Test k values 1-20
# TODO: NN - Tune hidden layer sizes and increase the maximum number of iterations (using the max_iter parameter) might help if the NN isn’t converging.
# TODO: Use logging instead of print statements https://docs.python.org/3/library/logging.html
# TODO: Add caching for vectorized data

MAX_FEATURES = 1000

# Helper Function to concatenate two arrays
def concatenate_arrays(array_1: list, array_2: list) -> list:
    """
    Helper function to concatenate two arrays.
    Args:
        array_1 (list): Array 1
        array_2 (list): Array 2

    Returns:
        list: concatenated array
    """
    return [list(array) for array in zip(array_1, array_2)]

def auto_correct(text: str) -> str:
    """
    Helper function to auto correct text.
    Args:
        text (str): text to correct

    Returns:
        str: corrected text
    """
    blob = textblob.TextBlob(text)
    blob_correct = blob.correct()
    return blob_correct

def import_csv_data(filename: str) -> list:
    """
    Function to import csv data and return it as a list.
    Args:
        filename (str): filename of csv file

    Returns:
        list: list of data from csv file
    """
    # print("\nImporting csv data...")
    try: 
        with open(filename, "r", encoding="latin-1") as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
            return data
        
    except FileNotFoundError as e:
        print(f"Error: File '{filename}' was not found.")
        
    except Exception as e:
        print(f"Error: {e}")

def get_sentiment(text):
    blob = textblob.TextBlob(text)
    return {
        "x_description_sentiment": {
            "polarity": blob.polarity,
            "subjectivity": blob.subjectivity
        }
    }

def format_data(data: list) -> dict:
    """
    Function to format data into a dictionary with x and y variables.
    Args:
        data (list): list of data from csv file

    Returns:
        dict: dictionary with x and y variables
    """
    # print("Data received to process and format....")
    title_list = []
    blurb_list = []
    blurb_length_list = []
    requested_list = []
    description_list = []
    description_length_list = []
    description_polarity_list = []
    description_subjectivity_list = []
    description_sentiment_list = []
    y = []
    
    for line in data:
        title = line[1]
        blurb = line[2]
        status = line[3]
        requested = line[4]
        description = line[5]
        sentiment = get_sentiment(description)["x_description_sentiment"]
        
        if status == "Funding Successful":
            y.append(1)
        else:
            y.append(0)
        
        title_list.append(title)
        blurb_list.append(blurb)
        requested_list.append([float(requested)])
        description_list.append(description)
        # ?FIXME: return len(description) instead of [len(description)] ?
        description_length_list.append([len(description)])
        blurb_length_list.append([len(blurb)])
        description_polarity_list.append([sentiment["polarity"]])
        description_subjectivity_list.append([sentiment["subjectivity"]])
        description_sentiment_list.append([sentiment["polarity"], sentiment["subjectivity"]])
    # print(json.dumps(description_subjectivity_list, indent=4))
    # print("Data successfully processed and formatted.\n")
    
    dict = {
        "x_title": vectorize_text(title_list),
        "x_description": vectorize_text(description_list),
        "x_description_length": description_length_list,
        "x_description_polarity": description_polarity_list,
        "x_description_subjectivity": description_subjectivity_list,
        "x_description_sentiment": description_sentiment_list,
        "x_blurb": vectorize_text(blurb_list),
        "x_blurb_length": blurb_length_list,
        "x_requested": requested_list,
        "y": y,    
    }
    
    # Create to store static data to decrease amount of computations
    # with open("formatted_data.json", 'w') as file:
    #     pass
        
    return dict

def vectorize_text(text: list, max_features=MAX_FEATURES) -> list:
    """
    Function to vectorize text data
    Args:
        text (list): list of text data
        max_features (int, optional): max features for vectorizer. Defaults to 1000.

    Returns:
        list: _description_
    """
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords.words("english"), max_features=max_features) # Stop word argument
    vectors = vectorizer.fit_transform(text)
    x = vectors.toarray()
        
    return x

def build_model(model_type, x, y, test_size=0.20) -> dict:
    """
    Function to build a machine learning model and return the results.
    Args:
        model_type (func): sklearn model type: DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier()
        x (2D List): x variables
        y (2D List): y variables
        test_size (float, optional): test size for train_test_split. Defaults to 0.20.

    Returns:
        dict: _description_
    """
    # print("Building ML model...")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
    clf = model_type
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    # print("ML build complete\n")
    
    return {
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
    }
    
def ml_factory(data: dict, feature="description", **kwargs) -> dict:
    """
    Function to run Decision Tree, KNN, and Neural Network models on data.
    Args:
        data (dict): _description_
        feature (str, optional): _description_. Defaults to "description". Used to test different x variables.

    Raises:
        ValueError: if feature not found in data

    Returns:
        dict: Results from Decision Tree, KNN, and Neural Network models including metadata and accuracy.
    """
    test_size = kwargs.get("test_size", 0.20)
    dt_k_value = kwargs.get("k_value", 7)
    
    formatted_data = format_data(data)
    
    if feature not in formatted_data:
        raise ValueError(f"Feature '{feature} not found in data.")

    
    y = formatted_data["y"]
    x = formatted_data[feature]
    
    # print(f"x shape: {len(x)}, y shape: {len(y)}\n")
    
    # Decision Tree
    results_dt = build_model(model_type=sklearn.tree.DecisionTreeClassifier(), x=x, y=y, test_size=test_size)
    # KNN
    results_knn = build_model(model_type=sklearn.neighbors.KNeighborsClassifier(dt_k_value), x=x, y=y, test_size=test_size)
    # NN
    results_nn = build_model(model_type=sklearn.neural_network.MLPClassifier(), x=x, y=y, test_size=test_size)
    
    return {
        "metadata": {
            "x_test_feature": feature,
            "data_shape": f"x shape: {len(x)}, y shape: {len(y)}\n",
        },
        "data": {
            "results_dt": {
                "test_size": test_size,
                "k_value": dt_k_value,
                "accuracy": results_dt["accuracy"]
            },
            "results_knn": {
                "test_size": test_size,
                "accuracy": results_knn["accuracy"]
            },
            "results_nn": {
                "test_size": test_size,
                "accuracy": results_nn["accuracy"]
            },
        }
    }

def display_results(data: dict) -> None:
    """
    Functions displays metadata and results from each model in human-readable format.

    Args:
        data (dict): object of results from ML factory
    """
    metadata = data["metadata"]
    data = data["data"]
    
    print(f"\nTesting feature: {metadata["x_test_feature"]}")
    print(f"Data shape:")
    print(metadata["data_shape"])
    
    print("Decision Tree:")
    print(f"Test size: {data["results_dt"]["test_size"]}")
    print(f"K value: {data["results_dt"]["k_value"]}")
    print(f"Accuracy: {data["results_dt"]["accuracy"]}\n")
    
    print("KNN:")
    print(f"Test size: {data["results_knn"]["test_size"]}")
    print(f"Accuracy for KNN: {data["results_knn"]["accuracy"]}\n")
    
    print("NN:")
    print(f"Test size: {data["results_nn"]["test_size"]}")
    print(f"Accuracy for NN: {data["results_nn"]["accuracy"]}\n")
    print("-" * 50)

def main():
    """
    Main function to call other functions, run models, and test different params to fine tune.
    """
    filename = './kickstarter_data_full.csv'
    data = import_csv_data(filename=filename)
    DT_K_VALUE = 7
    TEST_SIZE = 0.30
    input_data_features_test = {
            "title": {
                "data": data,
                "feature": "x_title",
                "dt_test_size": TEST_SIZE,
                "knn_test_size": TEST_SIZE,
                "nn_test_size": TEST_SIZE,
                "dt_k_value": DT_K_VALUE, 
            },
            "description": {
                "text": {    
                    "data": data,
                    "feature": "x_description",
                    "dt_test_size": TEST_SIZE,
                    "knn_test_size": TEST_SIZE,
                    "nn_test_size": TEST_SIZE,
                    "dt_k_value": DT_K_VALUE, 
                },
                "length": {
                    "data": data,
                    "feature": "x_description_length",
                    "dt_test_size": TEST_SIZE,
                    "knn_test_size": TEST_SIZE,
                    "nn_test_size": TEST_SIZE,
                    "dt_k_value": DT_K_VALUE, 
                    },
                "sentiment": {
                        "data": data,
                        "feature": "x_description_sentiment",
                        "dt_test_size": TEST_SIZE,
                        "knn_test_size": TEST_SIZE,
                        "nn_test_size": TEST_SIZE,
                        "dt_k_value": DT_K_VALUE, 
                    },
                "requested": {
                    "data": data,
                        "feature": "x_requested",
                        "dt_test_size": TEST_SIZE,
                        "knn_test_size": TEST_SIZE,
                        "nn_test_size": TEST_SIZE,
                        "dt_k_value": DT_K_VALUE, 
                }    
            }
        }
    
    # for i in input_data_features_test:
    #     # print(json.dumps(i, indent=4))   
    #     ml_results = ml_factory(**i)
    #     display_results(ml_results)
    
    # formatted_data = format_data(data=data)
    title = input_data_features_test["title"]
    
    # print(title)
    ml_results = ml_factory(feature=title["feature"], data=title["data"])
    display_results(ml_results)
    print()
    # ml_results = ml_factory(feature="x_description_subjectivity", data=data)
    # display_results(ml_results)
    
    
if __name__ == "__main__":
    main()
    
"""
Sample size 0.20 for NN has best accuracy so far:
NN:
Test size: 0.2
Accuracy for NN: 0.6757829730921924
Max Features: 1000
"""