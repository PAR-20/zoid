from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

def train_test_evaluation(X, y, test_size=0.2, random_state=42):
    """Simple train-test split evaluation"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC AUC for binary classification
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }

        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 50)

    return results, X_train, X_test, y_train, y_test

def cross_validation_evaluation(X, y, n_splits=5, random_state=42):
    """Cross-validation evaluation"""
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }

    # Define cross-validation strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = {}

    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        cv_results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'all_scores': cv_scores
        }

        print(f"Model: {name}")
        print(f"Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"Std Accuracy: {cv_scores.std():.4f}")
        print("-" * 50)

    return cv_results

def parameter_tuning(X, y, model_type='svm', random_state=42):
    """Parameter tuning using GridSearchCV"""
    if model_type == 'svm':
        model = SVC(probability=True, random_state=random_state)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        raise ValueError("Unsupported model type")

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def save_model(model, filename):
    """Save trained model to file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load trained model from file"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(results):
    """Plot accuracy comparison between models"""
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{acc:.4f}',
                 ha='center', va='bottom',
                 fontweight='bold')

    plt.title('Model Accuracy Comparison', fontsize=15)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate text
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig('/Users/dalm1/Desktop/reroll/Progra/par20/results/accuracy_comparison.png')
    plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def train_test_evaluation(X, y, test_size=0.2, random_state=42):
    """Simple train-test split evaluation"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC AUC for binary classification
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }

        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 50)

    return results, X_train, X_test, y_train, y_test

    # Add feature standardization and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split the PCA-transformed data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC AUC for binary classification
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }

        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 50)

    # Save preprocessing components
    save_model(scaler, '/Users/dalm1/Desktop/reroll/Progra/par20/results/scaler.pkl')
    save_model(pca, '/Users/dalm1/Desktop/reroll/Progra/par20/results/pca_model.pkl')

    return results

def cross_validation_evaluation(X, y, n_splits=5, random_state=42):
    """Cross-validation evaluation"""
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }

    # Define cross-validation strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = {}

    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        cv_results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'all_scores': cv_scores
        }

        print(f"Model: {name}")
        print(f"Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"Std Accuracy: {cv_scores.std():.4f}")
        print("-" * 50)

    return cv_results

def parameter_tuning(X, y, model_type='svm', random_state=42):
    """Parameter tuning using GridSearchCV"""
    if model_type == 'svm':
        model = SVC(probability=True, random_state=random_state)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        raise ValueError("Unsupported model type")

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def save_model(model, filename):
    """Save trained model to file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load trained model from file"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(results):
    """Plot accuracy comparison between models"""
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{acc:.4f}',
                 ha='center', va='bottom',
                 fontweight='bold')

    plt.title('Model Accuracy Comparison', fontsize=15)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate text
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig('/Users/dalm1/Desktop/reroll/Progra/par20/results/accuracy_comparison.png')
    plt.show()
