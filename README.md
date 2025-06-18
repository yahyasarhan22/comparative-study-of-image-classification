# **Image Classification with Naive Bayes, Decision Tree, and Neural Network**

This repository contains code for an image classification task using three different machine learning models: **Naive Bayes**, **Decision Tree**, and **Neural Network (Feedforward Neural Network)**. The goal is to classify images of different categories (e.g., dogs, cats, birds) using various algorithms and compare their performance.

## **Overview**

In this project, the following machine learning algorithms are implemented and evaluated:

1. **Naive Bayes** - A probabilistic classifier based on Bayes' Theorem with an assumption of feature independence.
2. **Decision Tree** - A non-linear model that recursively splits the dataset into subsets based on the feature that best separates the data.
3. **Neural Network (Feedforward Neural Network)** - A multi-layer perceptron (MLP) used for learning non-linear relationships between input features.

### **Dataset**

The dataset consists of images of three different classes (e.g., **dogs**, **cats**, and **birds**). The images are resized to 32x32 pixels and flattened into 1D vectors for input into the models.

### **Libraries Used**

* **scikit-learn**: For building and training machine learning models.
* **Matplotlib**: For plotting evaluation results such as confusion matrices and performance metrics.
* **Seaborn**: For better visualization of confusion matrices.
* **scikit-image**: For image loading and preprocessing.

## **Installation**

To get started, clone the repository and install the required libraries using **pip**:

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
pip install -r requirements.txt
```

### **Required Libraries**

You can install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` should contain:

```
numpy
scikit-learn
scikit-image
matplotlib
seaborn
```

## **Usage**

### **1. Dataset**

* Place your dataset inside the `dataset/` folder.

  * The dataset should be organized into subfolders for each class (e.g., `dogs`, `cats`, `birds`).
  * Inside each subfolder, place the respective images (e.g., `dog1.jpg`, `cat2.jpg`).

### **2. Running the Code**

To run the model training and evaluation, simply execute the Python script:

```bash
python model.py
```

This script will:

1. Load and preprocess the dataset.
2. Split the data into training and testing sets.
3. Train three different models: **Naive Bayes**, **Decision Tree**, and **Neural Network**.
4. Evaluate the models using accuracy, classification report, and confusion matrix.
5. Visualize the performance metrics for each model.

### **3. Modifying the Dataset Path**

Make sure to update the dataset path in the script (`model.py`) where the images are loaded. In the script, modify:

```python
image_folder = r"C:\path_to_your_dataset"
```

### **4. Performance Metrics**

After training, the models are evaluated based on:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

These metrics are printed in the console, and visualized as confusion matrix heatmaps and bar charts for precision, recall, and F1-score.

## **Example Output**

You will see the following type of output:

```
Naive Bayes Accuracy: 0.85
Naive Bayes Classification Report:
               precision    recall  f1-score   support

           dog       0.87      0.88      0.87       200
           cat       0.82      0.80      0.81       250
         bird       0.85      0.86      0.85       300

    accuracy                           0.85       750
   macro avg       0.85      0.85      0.85       750
weighted avg       0.85      0.85      0.85       750

Naive Bayes Confusion Matrix:
[[176  12  12]
 [ 18 200  32]
 [ 22  23 255]]

# Visualized confusion matrix heatmap and precision/recall/F1-score bar chart
```

## **Evaluation Results**

The following models are compared based on their accuracy, precision, recall, F1-score, and confusion matrix:

1. **Naive Bayes**
2. **Decision Tree**
3. **Neural Network (FNN)**

The **Neural Network** model showed the best performance with the highest accuracy and balanced precision/recall.

## **Conclusion**

This project demonstrates how different machine learning models can be used for image classification tasks. The **Neural Network** outperformed both **Naive Bayes** and **Decision Tree** in terms of accuracy and overall classification performance.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Suggestions for Future Work**

* **Data Augmentation**: Use techniques like **rotation**, **scaling**, and **flipping** to increase the size of the training data and prevent overfitting.
* **Hyperparameter Tuning**: Experiment with different hyperparameters for the models (e.g., number of hidden layers in the neural network, depth of the decision tree).
* **Ensemble Methods**: Combine multiple models to improve accuracy.

