import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns

def getLabels():
    """ get amino acids' labels

        Return:
            list
                a list of amino acids' names
        """
    labels = pd.read_excel('Amino acid names.xlsx')
    labels = list(labels.iloc[:, 0])  # Assuming the names are in the first column

    return labels

def load_spectrum_files(folder):
    """ Load all .txt files in the specified folder and return as a numpy array. """
    data_list = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            data = np.loadtxt(file_path)
            data_list.append(data[:, 1])  # Assuming the second column contains the relevant data
    return np.array(data_list)

def makeDataset(labels):
    """make the train dataset and the test dataset

        Returns:
            X_train: numpy array
            X_test: numpy array
            y_train: numpy array
            y_test: numpy array
        """

    # create empty lists to load data
    X, y = [], []

    # load data
    for count, name in enumerate(labels):
        folder = os.path.join('dataset', name)
        data = load_spectrum_files(folder)

        # Assuming the 0.3-2.0 THz data is from row 20 to row 190
        spectra = data[:, 20:190]
        X.append(spectra)
        y.extend([count] * spectra.shape[0])

    X = np.vstack(X)
    y = np.array(y)

    # split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test

def GPCmodel():
    """train and test the Gaussian process classifier

        Returns:
            metrics_dict: dict
                A dictionary containing the accuracy, recall, F1, and precision for each class.
            training_time:float
                time to train the GPC model
            test_time:float
                test time
            fps:float
                processing speed of the GPC model
            conf_matrix: np.ndarray
                The confusion matrix
        """

    # get amino acids' labels
    labels = getLabels()
    # make train and test dataset
    X_train, X_test, y_train, y_test = makeDataset(labels)

    # standardize the data
    pipe = make_pipeline(StandardScaler())
    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # create the model
    model = GaussianProcessClassifier(multi_class='one_vs_one')

    # record the beginning time
    time_start = time.time()
    # fitting the model
    model.fit(X_train, y_train)
    # training time
    training_time = time.time() - time_start

    time_start = time.time()
    # predicted results
    predict = model.predict(X_test)
    # test time
    test_time = time.time() - time_start

    # processing speed
    fps = len(y_test) / test_time

    # confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, predict)

    # calculate precision, recall, and F1 score
    precision = metrics.precision_score(y_test, predict, average=None)
    recall = metrics.recall_score(y_test, predict, average=None)
    f1_score = metrics.f1_score(y_test, predict, average=None)
    accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Create a dictionary to store all metrics for each class
    metrics_dict = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score
    }

    return metrics_dict, training_time, test_time, fps, conf_matrix, labels

def plot_confusion_matrix(conf_matrix, labels):
    """Plot the confusion matrix."""
    plt.figure(figsize=(10, 8))
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xticks(rotation=45)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # run GPC model
    metrics_dict, training_time, test_time, fps, conf_matrix, labels = GPCmodel()

    # show the results
    print("Per-Class Metrics:")
    for i, label in enumerate(labels):
        print(f"\nClass: {label}")
        print(f"Accuracy: {metrics_dict['accuracy'][i]:.4f}")
        print(f"Recall: {metrics_dict['recall'][i]:.4f}")
        print(f"Precision: {metrics_dict['precision'][i]:.4f}")
        print(f"F1 Score: {metrics_dict['f1_score'][i]:.4f}")

    print("\nOverall Metrics:")
    print("Training Time:   ", training_time)
    print("Test Time: ", test_time)
    print("Processing Speed:", fps, " fps")

    # plot and save the confusion matrix
    plot_confusion_matrix(conf_matrix, labels)
