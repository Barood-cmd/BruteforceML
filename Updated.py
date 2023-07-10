from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import warnings
import time
start_time = time.time()
warnings.filterwarnings("ignore")
# Define the range of values to iterate over for n_components in PCA and NMF
n_components_range = [1, 2, 3, 4]
n_components_nmf_range = [1, 2, 3]
# Define the range of values to iterate over for k in SelectKBest
k_range = [1, 2, 3, 4]
# Define the range of values to iterate over for cv in RFECV
cv_range = [3, 5, 10]
def apply_feature_reduction(reduction_model, X_train, X_test):
    if isinstance(reduction_model, (PCA, NMF)):
        n_components_range = [1, 2, 3, 4] if isinstance(reduction_model, PCA) else [1, 2, 3]
        for n_components in n_components_range:
            reduction_model.n_components = n_components
            X_train_reduced = reduction_model.fit_transform(X_train, y_train)
            X_test_reduced = reduction_model.transform(X_test)
            yield X_train_reduced, X_test_reduced
    elif isinstance(reduction_model, SelectKBest):
        k_range = [1, 2, 3, 4]
        for k in k_range:
            reduction_model.k = k
            X_train_reduced = reduction_model.fit_transform(X_train, y_train)
            X_test_reduced = reduction_model.transform(X_test)
            yield X_train_reduced, X_test_reduced
    elif isinstance(reduction_model, (RFECV, SelectFromModel)):
        cv_range = [3, 5, 10]
        for cv in cv_range:
            reduction_model.cv = cv
            X_train_reduced = reduction_model.fit_transform(X_train, y_train)
            X_test_reduced = reduction_model.transform(X_test)
            yield X_train_reduced, X_test_reduced

def apply_feature_scaling(scaling_model, X_train_reduced, X_test_reduced):
    for scaling_name, scaling_model in scaling_model.items():
        X_train_scaled = scaling_model.fit_transform(X_train_reduced)
        X_test_scaled = scaling_model.transform(X_test_reduced)
        yield scaling_name, X_train_scaled, X_test_scaled

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

best_confusion_matrix = None
best_accuracy = 0.0
best_model = None

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Define a dictionary of popular models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Multi-layer Perceptron": MLPClassifier(),
    "Extra Trees": ExtraTreesClassifier()
}

# Define a dictionary of feature reduction techniques
feature_reduction_techniques = {
    "PCA": PCA(),
    "NMF": NMF(),
    "SelectKBest": SelectKBest(chi2),
    "Recursive Feature Elimination": RFECV(estimator=LogisticRegression()),
    "Lasso": SelectFromModel(estimator=Lasso()),
    "Linear SVC": SelectFromModel(estimator=LinearSVC())
}

# Define a dictionary of feature scaling techniques
feature_scaling_techniques = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

# Iterate through feature reduction techniques
for reduction_name, reduction_model in feature_reduction_techniques.items():
    for X_train_reduced, X_test_reduced in apply_feature_reduction(reduction_model, X_train, X_test):
        for scaling_name, X_train_scaled, X_test_scaled in apply_feature_scaling(feature_scaling_techniques, X_train_reduced, X_test_reduced):
            for model_name, model in models.items():
                # Train the model
                model.fit(X_train_scaled, y_train)
                # Make predictions on the scaled test set
                y_pred = model.predict(X_test_scaled)
                confusion_mat = confusion_matrix(y_test, y_pred)
                # Calculate accuracy from the confusion matrix
                accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)
                # Update the best model and accuracy if necessary
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
                    best_confusion_matrix = confusion_mat

# Print the best model and its accuracy for the current feature reduction and scaling techniques
if isinstance(reduction_model, SelectKBest):
    print(f"Feature Reduction Technique: {reduction_name}")
    print(f"k: {reduction_model.k}")
elif isinstance(reduction_model, (PCA, NMF)):
    print(f"Feature Reduction Technique: {reduction_name}")
    print(f"n_components: {reduction_model.n_components}")
else:
    print(f"Feature Reduction Technique: {reduction_name}")
    print(f"cv: {reduction_model.cv}")

print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_accuracy}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
