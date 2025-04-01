import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

from data_exploration import explore_image_dataset, download_dataset_from_huggingface
from preprocessing import preprocess_images, extract_features
from model_training import (
    train_test_evaluation, cross_validation_evaluation,
    parameter_tuning, save_model, load_model, plot_confusion_matrix,
    plot_accuracy_comparison
)
from deep_learning import (
    create_cnn_model, create_transfer_learning_model,
    train_model_with_data_augmentation, plot_training_history
)
from som_visualization import train_som, visualize_som_umatrix, visualize_som_component_planes

dataset_paths = [
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset1'
]

for path in dataset_paths:
    os.makedirs(path, exist_ok=True)

# Create output directory for results
os.makedirs('results', exist_ok=True)

# Vérifier si le dataset est vide et proposer de télécharger un dataset
datasets_empty = all(len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]) == 0 for path in dataset_paths)

if datasets_empty:
    print("Aucun dataset trouvé ou dataset vide. Téléchargement d'un dataset depuis Hugging Face...")
    download_from_huggingface = input("Voulez-vous télécharger un dataset depuis Hugging Face? (oui/non): ").lower() == 'oui'

    if download_from_huggingface:
        dataset_name = input("Entrez le nom du dataset Hugging Face (ex: 'keremberke/chest-xray-classification'): ")
        downloaded_path = download_dataset_from_huggingface(dataset_name)

        if downloaded_path:
            # Ajouter le dataset téléchargé à la liste des datasets à explorer
            dataset_paths.append(downloaded_path)
            print(f"Dataset téléchargé avec succès dans {downloaded_path}")
        else:
            print("Échec du téléchargement du dataset. Le programme va s'arrêter.")
            exit()
    else:
        print("Aucun dataset disponible. Le programme va s'arrêter.")
        exit()

# 1. Data Exploration
print("\nStep 1: Data Exploration")
valid_datasets = []
for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if classes:
            valid_datasets.append(dataset_path)
            explore_image_dataset(dataset_path)
        else:
            print(f"Le répertoire {dataset_path} ne contient pas de classes (sous-répertoires).")
    else:
        print(f"Le répertoire {dataset_path} n'existe pas.")

if not valid_datasets:
    print("Aucun dataset valide trouvé. Le programme va s'arrêter.")
    exit()

# Utiliser uniquement les datasets valides pour la suite
dataset_paths = valid_datasets

# 2. Data Preprocessing
print("\nStep 2: Data Preprocessing")
# Load and preprocess images from dataset
def load_dataset(dataset_path, target_size=(150, 150)):
    images = []
    labels = []
    classes = []

    for class_idx, class_name in enumerate(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(img_path)
                    labels.append(class_idx)

    # Preprocess images
    X = preprocess_images(images, target_size=target_size)
    y = np.array(labels)

    return X, y, classes

# Load dataset
X, y, classes = load_dataset(dataset_paths[0])
print(f"Dataset: {X.shape} images, {len(np.unique(y))} classes")

# 3. Feature Extraction (optional for traditional ML)
print("\nStep 3: Feature Extraction")
# Extract features using PCA for traditional ML models
X_features, pca, scaler = extract_features(X, n_components=0.95)

# 4. Traditional ML Models with Simple Train-Test Split
print("\nStep 4: Traditional ML Models with Simple Train-Test Split")
results_simple, X_train, X_test, y_train, y_test = train_test_evaluation(X_features, y)

# Plot accuracy comparison
plot_accuracy_comparison(results_simple)

# 5. Cross-Validation Evaluation
print("\nStep 5: Cross-Validation Evaluation")
cv_results = cross_validation_evaluation(X_features, y)

# 6. Parameter Tuning
print("\nStep 6: Parameter Tuning")
best_model, best_params, best_score = parameter_tuning(X_features, y, model_type='svm')

# Save the best model
save_model(best_model, 'results/best_traditional_model.pkl')

# 7. Deep Learning Models
print("\nStep 7: Deep Learning Models")
# Split dataset for deep learning
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split training data into train and validation
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
    X_train_dl, y_train_dl, test_size=0.2, random_state=42, stratify=y_train_dl
)

# Reshape for CNN input (add channel dimension if grayscale)
if len(X_train_dl.shape) == 3:
    X_train_dl = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], X_train_dl.shape[2], 1)
    X_val_dl = X_val_dl.reshape(X_val_dl.shape[0], X_val_dl.shape[1], X_val_dl.shape[2], 1)
    X_test_dl = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], X_test_dl.shape[2], 1)

# Create and train CNN model
cnn_model = create_cnn_model(
    input_shape=X_train_dl.shape[1:],
    num_classes=len(np.unique(y))
)

cnn_model, history = train_model_with_data_augmentation(
    cnn_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl,
    batch_size=32, epochs=30, data_aug=True
)

# Plot training history
plot_training_history(history)

# Evaluate CNN model
cnn_evaluation = cnn_model.evaluate(X_test_dl, y_test_dl)
print(f"CNN Test Accuracy: {cnn_evaluation[1]:.4f}")

# Save CNN model
cnn_model.save('results/best_cnn_model.h5')

# 8. SOM Visualization (Bonus)
print("\nStep 8: SOM Visualization")
# Train SOM on dataset features
som, scaler, X_scaled = train_som(X_features, map_size=(10, 10))

# Visualize SOM
visualize_som_umatrix(som, X_scaled, y)

# 9. Results Comparison
print("\nStep 9: Results Comparison")
# Compare all models
model_names = list(results_simple.keys()) + ['CNN']
accuracies = [results_simple[name]['accuracy'] for name in results_simple.keys()] + [cnn_evaluation[1]]

plt.figure(figsize=(12, 6))
plt.bar(model_names, accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/model_comparison.png')
plt.show()

# 10. Save Results Summary
print("\nStep 10: Save Results Summary")
# Create summary dataframe
summary = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
})

# Add cross-validation results
for name, result in cv_results.items():
    if name in summary['Model'].values:
        idx = summary[summary['Model'] == name].index[0]
        summary.loc[idx, 'CV_Accuracy'] = result['mean_accuracy']
        summary.loc[idx, 'CV_Std'] = result['std_accuracy']

# Save summary
summary.to_csv('results/model_summary.csv', index=False)

print("\n")

print (" (              (       *  ")
print (" )\ )    (      )\ )  (  `       )")
print ("(()/(    )\    (()/(  )\))(   ( /(")
print (" /(_))((((_)(   /(_))((_)()\  )\())")
print ("(_))_  )\ _ )\ (_))  (_()((_)((_)\ ")
print (" |   \ (_)_\(_)| |   |  \/  | / (_)")
print (" | |) | / _ \  | |__ | |\/| | | |  ")
print (" |___/ /_/ \_\ |____||_|  |_| |_| ")

print("Results saved to 'results' directory")

print("\nAnalysis Complete")
