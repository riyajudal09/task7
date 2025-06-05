# svm_example.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Create/load datasets
# 1a) Synthetic 2D dataset (for visualization of decision boundaries)
X_toy, y_toy = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

# Split toy data into train and test (we’ll only visualize on full 2D)
X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(
    X_toy, y_toy, test_size=0.3, random_state=42
)

# 1b) Real-world “Breast Cancer Wisconsin (Diagnostic)” dataset (binary classification)
breast = datasets.load_breast_cancer()
X_bc = breast.data        # shape (569, 30)
y_bc = breast.target      # shape (569,)


# Standardize the breast cancer features (very important for SVM)
scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(X_bc)

# Split into train/test for evaluation
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
    X_bc_scaled, y_bc, test_size=0.3, random_state=42, stratify=y_bc
)
# 2. Train and visualize on the 2D toy data

# Scale the toy data as well
scaler_toy = StandardScaler()
X_toy_train_scaled = scaler_toy.fit_transform(X_toy_train)
X_toy_test_scaled = scaler_toy.transform(X_toy_test)

# Instantiate two SVMs: linear and RBF
svm_linear_toy = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf_toy = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Fit both on the toy training data
svm_linear_toy.fit(X_toy_train_scaled, y_toy_train)
svm_rbf_toy.fit(X_toy_train_scaled, y_toy_train)

# Function to plot decision boundary in 2D
def plot_decision_boundary(model, X, y, ax, title):
    """
    - model: a fitted 2D classifier
    - X: 2D numpy array (scaled or raw, but consistent with how model was trained)
    - y: binary labels
    - ax: matplotlib Axes to plot on
    - title: plot title string
    """
    # Define grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on each point of the grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, levels=[-0.5, 0.5, 1.5])

    # Scatter the original training points
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap=plt.cm.coolwarm,
        edgecolors='k', s=20
    )
    ax.set_title(title)
    ax.set_xlabel('Feature 1 (scaled)')
    ax.set_ylabel('Feature 2 (scaled)')


# Create a figure with two subplots: linear vs. RBF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(
    svm_linear_toy,
    X_toy_train_scaled,
    y_toy_train,
    axes[0],
    title="SVM Linear Kernel (Toy 2D)"
)
plot_decision_boundary(
    svm_rbf_toy,
    X_toy_train_scaled,
    y_toy_train,
    axes[1],
    title="SVM RBF Kernel (Toy 2D)"
)
plt.tight_layout()
plt.show()

# Print test accuracy on the toy set
for name, clf in [("Linear SVM", svm_linear_toy), ("RBF SVM", svm_rbf_toy)]:
    y_pred_test = clf.predict(X_toy_test_scaled)
    acc = accuracy_score(y_toy_test, y_pred_test)
    print(f"{name} Test Accuracy on Toy Data: {acc:.3f}")

# 3. Train, tune, and evaluate on the Breast Cancer dataset

# 3a) First, fit a baseline Linear SVM and RBF SVM with default hyperparameters
svm_linear_bc = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf_bc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

svm_linear_bc.fit(X_bc_train, y_bc_train)
svm_rbf_bc.fit(X_bc_train, y_bc_train)

# Evaluate on the hold-out test set
def evaluate_model(clf, X_test, y_test, model_name):
    y_pred = clf.predict(X_test)
    print(f"\n=== {model_name} Evaluation on Breast Cancer Test Set ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

evaluate_model(svm_linear_bc, X_bc_test, y_bc_test, "Linear SVM (C=1)")
evaluate_model(svm_rbf_bc, X_bc_test, y_bc_test, "RBF SVM (C=1, gamma='scale')")

# 3b) Hyperparameter tuning with GridSearchCV (for RBF SVM)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_bc_train, y_bc_train)

print("\nBest parameters found by grid search:")
print(grid_search.best_params_)

best_rbf_svm = grid_search.best_estimator_

# Evaluate the best RBF SVM on the hold-out test set
evaluate_model(best_rbf_svm, X_bc_test, y_bc_test, "Best RBF SVM (after GridSearchCV)")


# 3c) (Optional) Cross‐validation scores for the Linear SVM
cv_scores_linear = cross_val_score(svm_linear_bc, X_bc_train, y_bc_train, cv=5, scoring='accuracy')
print("\n5-fold CV accuracy scores for Linear SVM (C=1):", cv_scores_linear)
print("Mean CV accuracy:", cv_scores_linear.mean())

# 3d) (Optional) Cross‐validation scores for the best RBF SVM
cv_scores_rbf = cross_val_score(best_rbf_svm, X_bc_train, y_bc_train, cv=5, scoring='accuracy')
print("\n5-fold CV accuracy scores for Best RBF SVM:", cv_scores_rbf)
print("Mean CV accuracy:", cv_scores_rbf.mean())

