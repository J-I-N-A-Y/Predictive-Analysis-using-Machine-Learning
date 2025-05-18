# Predictive-Analysis-using-Machine-Learning

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:JINAY SHAH

*INTERN ID*:CT04DL198

*DOMAIN*:DATA ANALYTICS

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTHOSH 

## DESCRIPTION

This Python script performs a complete machine learning workflow using the breast cancer dataset from scikit-learn. It involves data preprocessing, feature selection, model training, and performance evaluation. Each step is carefully structured to demonstrate a practical approach to binary classification using a real-world dataset and a Random Forest classifier.

---

### **1. Importing Libraries**

The script begins by importing essential libraries. `pandas` and `numpy` are used for data manipulation. `matplotlib.pyplot` is imported for visualization, though it isn’t used in the script. From scikit-learn, multiple modules are used: dataset loading, preprocessing, model selection, feature selection, ensemble methods, and evaluation metrics.

---

### **2. Loading the Dataset**

The script loads the built-in **Breast Cancer Wisconsin dataset** using `load_breast_cancer()` from `sklearn.datasets`. This dataset includes features computed from digitized images of breast masses, which are used to predict whether a tumor is **malignant** or **benign**. The features (`X`) are stored in a DataFrame with descriptive column names, while the target variable (`y`) is stored as a Series.

It prints the shape of the dataset and the class labels (`malignant` and `benign`), giving a quick understanding of the data size and classification target.

---

### **3. Feature Scaling**

Before modeling, the data is standardized using `StandardScaler`. Feature scaling ensures that each feature contributes equally to the model by transforming the data to have a mean of 0 and a standard deviation of 1. This step is important, especially for distance-based or regularization-based models.

---

### **4. Feature Selection**

The script then applies **univariate feature selection** using `SelectKBest` with the `f_classif` scoring function, which uses ANOVA F-values to assess the relationship between each feature and the target variable. It selects the top 10 features that are most strongly associated with the output class. This helps reduce dimensionality, improve model performance, and decrease overfitting risk. The names of the selected features are printed for reference.

---

### **5. Train-Test Split**

Next, the dataset is split into training and testing sets using `train_test_split`, reserving 20% of the data for testing. The `random_state` parameter ensures reproducibility of the split. This step is crucial for evaluating the model's generalization performance on unseen data.

---

### **6. Model Training**

The classifier used is `RandomForestClassifier`, an ensemble learning method that builds multiple decision trees and merges their results for better accuracy and control over overfitting. The model is trained using the training dataset (`X_train` and `y_train`).

---

### **7. Model Evaluation**

After training, predictions (`y_pred`) are made on the test dataset (`X_test`). The model's performance is evaluated using three key metrics:

* **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
* **Classification Report**: Includes precision, recall, F1-score, and support for each class.
* **Accuracy Score**: Indicates the overall percentage of correctly predicted instances.

These metrics provide a comprehensive view of how well the model distinguishes between benign and malignant tumors.

## OUTPUT

Dataset shape: (569, 30)
Target labels: ['malignant' 'benign']
Selected Features: ['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']
Confusion Matrix:
 [[40  3]
 [ 2 69]]
Classification Report:

               precision    recall  f1-score   support

           0       0.95      0.93      0.94        43
           1       0.96      0.97      0.97        71

    accuracy                           0.96       114
    
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

Accuracy Score: 0.956140350877193
