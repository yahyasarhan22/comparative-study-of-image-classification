#Yahya Sarhan 1221858                                         
import numpy as np
import os
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Method 1: Load and preprocess the images
def load_images(image_folder, image_size=(32, 32)):
    """
    Loads images from the dataset, resizes them to a specified size,
    flattens them into 1D vectors, and returns the feature vectors and labels.

    Parameters:
        image_folder (str): Path to the root dataset folder.
        image_size (tuple): Desired image size for resizing (default: 32x32).

    Returns:
        np.array: Flattened image data (features).
        np.array: Corresponding labels for each image.
    """
    images = []
    labels = []
    classes = os.listdir(image_folder)  # Get the list of classes (folders)

    for class_label in classes:
        class_folder = os.path.join(image_folder, class_label)

        if os.path.isdir(class_folder):  # Ensure it's a folder
            print(f"Processing class: {class_label}")

            # Loop through each image in the class folder
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)

                if os.path.isfile(image_path) and image_name.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        # Load, resize, and flatten the image
                        image = io.imread(image_path)
                        image_resized = transform.resize(image, image_size)
                        image_flattened = image_resized.flatten()

                        images.append(image_flattened)
                        labels.append(class_label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels)


# Method 2: Train a model (Naive Bayes, Decision Tree, or Neural Network)
def train_model(model_type, X_train, y_train):
    """
    Trains a classifier model (Naive Bayes, Decision Tree, or Neural Network) based on the model type.

    Parameters:
        model_type (str): Type of model to train ('naive_bayes', 'decision_tree', 'neural_network').
        X_train (np.array): Feature data for training.
        y_train (np.array): Labels for training.

    Returns:
        model: Trained model.
    """
    if model_type == 'naive_bayes':
        model = GaussianNB()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'naive_bayes', 'decision_tree', or 'neural_network'.")

    model.fit(X_train, y_train)
    return model


# Method 3: Evaluate the model and display results
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model using accuracy, classification report, and confusion matrix.

    Parameters:
        model: Trained classifier model.
        X_test (np.array): Feature data for testing.
        y_test (np.array): Labels for testing.
        model_name (str): Name of the model to include in the plot title.
    """
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

    # Print classification report (precision, recall, F1-score)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print(f"{model_name} Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Visualize Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'{model_name} Confusion Matrix Heatmap')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()


# Method 4: Visualize performance metrics for both models (Precision, Recall, F1-Score)
def visualize_metrics(classification_report_dict, model_name):
    """
    Visualizes precision, recall, and F1-score for each class in a bar chart.

    Parameters:
        classification_report_dict (dict): The classification report dictionary obtained from sklearn's classification_report().
        model_name (str): The name of the model to include in the plot title.
    """
    classes = list(classification_report_dict.keys())[:-3]  # Excluding 'accuracy', 'macro avg', 'weighted avg'
    precision = [classification_report_dict[class_name]['precision'] for class_name in classes]
    recall = [classification_report_dict[class_name]['recall'] for class_name in classes]
    f1_score = [classification_report_dict[class_name]['f1-score'] for class_name in classes]

    # Plotting Precision, Recall, and F1-Score for each class
    x = np.arange(len(classes))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title(f'{model_name} Precision, Recall, and F1-Score for each Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    fig.tight_layout()
    plt.show()


# Main Code to run the methods
if __name__ == "__main__":
    # Set the path to your dataset folder
    image_folder = r"C:\Users\yahya_k6rln48\OneDrive\Desktop\project2_AI\bigger_dataset"

    # Load the dataset
    X, y = load_images(image_folder)

    if len(X) == 0:
        print("No images were loaded. Check your dataset and path.")
    else:
        print(f"Images loaded: {len(X)}")

        # Split dataset into training and testing sets (70% training, 30% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Naive Bayes model
        nb_model = train_model('naive_bayes', X_train, y_train)
        print("\nEvaluating Naive Bayes Model:")
        evaluate_model(nb_model, X_test, y_test, "Naive Bayes")

        # Train Decision Tree model
        dt_model = train_model('decision_tree', X_train, y_train)
        print("\nEvaluating Decision Tree Model:")
        evaluate_model(dt_model, X_test, y_test, "Decision Tree")

        # Train Neural Network model (Feedforward Neural Network)
        nn_model = train_model('neural_network', X_train, y_train)
        print("\nEvaluating Neural Network (FNN) Model:")
        evaluate_model(nn_model, X_test, y_test, "Neural Network (FNN)")

        # Visualize the metrics for Naive Bayes
        print("\nVisualizing Metrics for Naive Bayes Model:")
        visualize_metrics(classification_report(y_test, nb_model.predict(X_test), output_dict=True), "Naive Bayes")

        # Visualize the metrics for Decision Tree
        print("\nVisualizing Metrics for Decision Tree Model:")
        visualize_metrics(classification_report(y_test, dt_model.predict(X_test), output_dict=True), "Decision Tree")

        # Visualize the metrics for Neural Network
        print("\nVisualizing Metrics for Neural Network (FNN) Model:")
        visualize_metrics(classification_report(y_test, nn_model.predict(X_test), output_dict=True),
                          "Neural Network (FNN)")
