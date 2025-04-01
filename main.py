import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

# Import custom modules
from data_exploration import explore_image_dataset
from preprocessing import preprocess_images, extract_features
from model_training import (
    train_test_evaluation, cross_validation_evaluation, 
    parameter_tuning, save_model, load_model, plot_confusion_matrix
)
from deep_learning import (
    create_cnn_model, create_transfer_learning_model, 
    train_model_with_data_augmentation, plot_training_history
)
from som_visualization import train_som, visualize_som_umatrix, visualize_som_component_planes

# Define paths to datasets
dataset_paths = [
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset1',
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset2',
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset3'
]

# Create output directory for results
os.makedirs('results', exist_ok=True)

# 1. Data Exploration
print("Step 1: Data Exploration")
for dataset_path in dataset_paths:
    explore_image_dataset(dataset_path)

# 2. Data Preprocessing
print("\nStep 2: Data Preprocessing")
# Load and preprocess images from dataset1 (for training and validation)
# Assuming dataset structure: dataset/class/image.jpg
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

# Load datasets
X1, y1, classes1 = load_dataset(dataset_paths[0])
X2, y2, classes2 = load_dataset(dataset_paths[1])
X3, y3, classes3 = load_dataset(dataset_paths[2])

print(f"Dataset 1: {X1.shape} images, {len(np.unique(y1))} classes")
print(f"Dataset 2: {X2.shape} images, {len(np.unique(y2))} classes")
print(f"Dataset 3: {X3.shape} images, {len(np.unique(y3))} classes")

# 3. Feature Extraction (optional for traditional ML)
print("\nStep 3: Feature Extraction")
# Extract features using PCA for traditional ML models
X1_features, pca1, scaler1 = extract_features(X1, n_components=0.95)
X2_features, pca2, scaler2 = extract_features(X2, n_components=0.95)
X3_features, pca3, scaler3 = extract_features(X3, n_components=0.95)

# 4. Traditional ML Models with Simple Train-Test Split
print("\nStep 4: Traditional ML Models with Simple Train-Test Split")
# Use dataset 1 for this evaluation
results_simple, X1_train, X1_test, y1_train, y1_test = train_test_evaluation(X1_features, y1)

# 5. Cross-Validation Evaluation
print("\nStep 5: Cross-Validation Evaluation")
# Use dataset 1 for cross-validation
cv_results = cross_validation_evaluation(X1_features, y1)

# 6. Parameter Tuning
print("\nStep 6: Parameter Tuning")
# Use dataset 2 for parameter tuning
best_model, best_params, best_score = parameter_tuning(X2_features, y2, model_type='svm')

# Save the best model
save_model(best_model, 'results/best_traditional_model.pkl')

# 7. Deep Learning Models
print("\nStep 7: Deep Learning Models")
# Split dataset 3 for deep learning
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3
)

# Further split training data into train and validation
X3_train, X3_val, y3_train, y3_val = train_test_split(
    X3_train, y3_train, test_size=0.2, random_state=42, stratify=y3_train
)

# Reshape for CNN input (add channel dimension if grayscale)
if len(X3_train.shape) == 3:
    X3_train = X3_train.reshape(X3_train.shape[0], X3_train.shape[1], X3_train.shape[2], 1)
    X3_val = X3_val.reshape(X3_val.shape[0], X3_val.shape[1], X3_val.shape[2], 1)
    X3_test = X3_test.reshape(X3_test.shape[0], X3_test.shape[1], X3_test.shape[2], 1)

# Create and train CNN model
cnn_model = create_cnn_model(
    input_shape=X3_train.shape[1:], 
    num_classes=len(np.unique(y3))
)

cnn_model, history = train_model_with_data_augmentation(
    cnn_model, X3_train, y3_train, X3_val, y3_val, 
    batch_size=32, epochs=30, data_aug=True
)

# Plot training history
plot_training_history(history)

# Evaluate CNN model
cnn_evaluation = cnn_model.evaluate(X3_test, y3_test)
print(f"CNN Test Accuracy: {cnn_evaluation[1]:.4f}")

# Save CNN model
cnn_model.save('results/best_cnn_model.h5')

# 8. SOM Visualization (Bonus)
print("\nStep 8: SOM Visualization")
# Train SOM on dataset 1 features
som, scaler, X1_scaled = train_som(X1_features, map_size=(10, 10))

# Visualize SOM
visualize_som_umatrix(som, X1_scaled, y1)

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
print("Results saved to 'results' directory")

print("\nAnalysis Complete!")