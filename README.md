# BruteforceML
Brute force Automated Model Selection and Hyperparameter Tuning for Classification Using Feature Reduction Techniques and Scaling
This repository contains a Python script that demonstrates an automated approach for selecting the best classification model and optimizing hyperparameters
using various feature reduction techniques and feature scaling methods. The script utilizes popular machine learning algorithms and evaluates their performance on the Iris dataset.

It employs techniques such as Principal Component Analysis (PCA),
Non-Negative Matrix Factorization (NMF), SelectKBest, Recursive Feature Elimination (RFECV), Lasso, and Linear SVC to reduce the feature space.
Additionally, it applies feature scaling techniques such as StandardScaler and MinMaxScaler. The script performs an exhaustive search over different combinations of feature reduction,
scaling, and classification algorithms to find the best model with the highest accuracy. The results,
including the chosen feature reduction technique, scaling technique, best model, and accuracy, are printed for each combination.
This code provides a valuable resource for automating the model selection and hyperparameter tuning process in classification tasks, enabling efficient and reliable analysis of datasets.
