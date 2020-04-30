"""
This is a sample solution code from Question2
We plot the curve by max depth verse AUC
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score
import matplotlib.pyplot as plt
import os

# Decision Tree Classifier
def decision_tree_clf(X_train, X_test, y_train, y_test):
    #Build a decsion tree classifier with depth = depth
    depth = 3
    dct = DecisionTreeClassifier(max_depth=depth)
    dct.fit(X_train, y_train)
    y_pred = dct.predict(X_test)
    print("Decision Tree Precision:", accuracy_score(y_test, y_pred))
    # export tree structure
    export_graphviz(dct, out_file="../result/CART/dct.dot")
    os.system("dot -T png ../result/CART/dct.dot -o ../result/CART/dct.png")
    pass

# Plot the curve AUC verse max depth for decision tree
def decision_tree_auc(X_train, X_test, y_train, y_test):
    aucs = []
    max_depth_to_show = 60
    for depth in range(1, max_depth_to_show):
        dct = DecisionTreeClassifier(max_depth=depth)
        dct.fit(X_train, y_train)
        y_pred = dct.predict(X_test)
        aucs.append(roc_auc_score(y_test, y_pred))
    plt.plot(list(range(1, max_depth_to_show)), aucs, label='Decsion Tree Classifier')
    pass

# Random Forest Classifier
def random_forest_clf(X_train, X_test, y_train, y_test):
    # Choose an appropraite depth. If the depth is too large, the model is on the risk of overfitting.
    depth = 3
    rf = RandomForestClassifier(n_estimators=10, max_depth=depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Precision:", accuracy_score(y_test, y_pred))
    for idx, estimator in enumerate(rf.estimators_):
        # export tree structure
        export_graphviz(estimator, out_file=f"../result/RandomForest/rf_estimator{idx}.dot")
        os.system(f"dot -T png ../result/RandomForest/rf_estimator{idx}.dot -o ../result/RandomForest/rf_estimator{idx}.png")
    pass

# Plot the curve AUC verse max depth for random forest
def random_forest_auc(X_train, X_test, y_train, y_test):
    aucs = []
    max_depth_to_show = 60
    for depth in range(1,max_depth_to_show):
        rf = RandomForestClassifier(n_estimators=10, max_depth=depth)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        aucs.append(roc_auc_score(y_test, y_pred))
    plt.plot(list(range(1,max_depth_to_show)), aucs, color="red", label='Random Forest Classifier')
    pass


# Data Preparation
def data_preparation(df):
    # first 57 columns are features
    # last column is labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Read Data
    df = pd.read_csv("../data/spambase.data", delimiter=",", header=None)
    # Train-Test split
    X_train, X_test, y_train, y_test = data_preparation(df)
    # Decision tree classifier (CART)
    decision_tree_clf(X_train, X_test, y_train, y_test)
    # Random forest classifier
    random_forest_clf(X_train, X_test, y_train, y_test)

    # plot AUC curve for decision tree
    decision_tree_auc(X_train, X_test, y_train, y_test)
    # plot AUC curve for random forest
    random_forest_auc(X_train, X_test, y_train, y_test)
    plt.title("Max Depth verse AUC")
    plt.xlabel("Max Depth of Each Tree")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()
    pass
