# Zoidberg

<img src="https://github.com/PAR-20/zoid/blob/main/visualisation/sample_images.png?raw=true" alt="Comparaison des performances" width="800">

## 📊 Résultats des Comparaisons
Les résultats de la comparaison des performances des modèles d'apprentissage automatique sont présentés dans le graphique ci-dessous.

<img src="results/accuracy_comparison.png" alt="Comparaison des performances" width="800">

## 🎯 Objectifs du Projet
1. **Détection Automatique de Pneumonie** : Analyser des radiographies pulmonaires pour identifier les cas de pneumonie
2. **Classification Binaire** : Distinguer les images "Normales" vs "Pneumonie"
3. **Validation Scientifique** : Comparer différentes approches d'IA pour la tâche médicale

## 📂 Structure des Données
dataset1/
├── normal/
│   ├── im1.png
│   └── ... (1341 images)
└── pneumonia/
├── im3875.png
└── ... (3875 images)

## 🔧 Méthodologie Technique

### 1. Préprocessing des Images
```python
def preprocess_images(...):
    # Conversion en niveaux de gris
    # Redimensionnement 150x150 pixels
    # Application de CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Normalisation des valeurs de pixels [0-1]

Thought process
```
## 🔍 For Beginners: Understanding the Medical Image Analysis

### 🤖 How the Computer "Learns" to Diagnose

The system uses special math patterns (called "machine learning models") to analyze X-ray images. Here's a simple explanation of what's happening:

1. **Image Preparation** (Like Organizing X-Rays):
   - Convert images to black & white, numbers
   - Enhance contrast to see details better
   - Standardize sizes to 150x150 pixels

2. **Pattern Recognition** (Finding Disease Signs):
   - The computer looks for patterns in:
     - Lung texture (bumpy vs smooth)
     - Light/dark areas (fluid vs air)
     - Shape irregularities

3. **Decision Making** (Diagnosis Prediction):
   - Uses Logistic Regression (our main math tool) to:
   - Weigh different patterns
   - Calculate infection probability
   - Make final diagnosis (pneumonia/normal)

📊 **Simple Regression Analogy**:
Imagine drawing a line through a scatter plot - our model constantly adjusts this "decision boundary" to best separate pneumonia cases from normal scans.

🩺 **Medical Connection**:
The model's predictions are based on patterns found in thousands of historical X-rays, learning to recognize what pneumonia typically looks like compared to healthy lungs.
🔍 **Understanding the Model**:
- The model's accuracy is like knowing how well it can predict pneumonia cases.
- The time it takes to process an image is like how long it takes a doctor to diagnose a patient.


# Projet de Classification d'Images Médicales

<img src="results/accuracy_comparison.png" alt="Comparaison des performances" width="800">


## 🎯 Objectifs du Projet
1. **Détection Automatique de Pneumonie** : Analyser des radiographies pulmonaires pour identifier les cas de pneumonie
2. **Classification Binaire** : Distinguer les images "Normales" vs "Pneumonie"
3. **Validation Scientifique** : Comparer différentes approches d'IA pour la tâche médicale

## 📂 Structure des Données
```
dataset1/
├── normal/
│   ├── im1.png
│   └── ... (1341 images)
└── pneumonia/
├── im3875.png
└── ... (3875 images)
```

## 🔧 Méthodologie Technique

### 1. Préprocessing des Images

```python
def preprocess_images(...):
    # Conversion en niveaux de gris
    # Redimensionnement 150x150 pixels
    # Application de CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Normalisation des valeurs de pixels [0-1]
```

### 2. Extraction de Caractéristiques
### Approche Traditionnelle

def extract_features(images):
    # Calcul d'histogrammes de textures
    # Analyse de motifs locaux (LBP)
    # Réduction de dimension avec PCA

Approche Deep Learning :


```python
def create_cnn_model(...):
```

### - Architecture CNN avec 3 couches convolutionnelles
### - MaxPooling pour réduire la dimension spatiale
### - Couche Dense finale avec activation sigmoïde

### 3. Entraînement des Modèles
Modèles Testés :

- Régression Logistique
- SVM (Machine à Vecteurs de Support)
- Random Forest
- Gradient Boosting
- Réseau de Neurones Artificiel (MLP)
- CNN (Réseau de Neurones Convolutif)
Validation Croisée :

````python
kf = KFold(n_splits=5, shuffle=True)
for model in models:
    scores = cross_val_score(model, X, y, cv=kf)
````


## 📊 Résultats Clés

## MODEL LOSS
<img src="https://github.com/PAR-20/zoid/blob/main/visualisation/model_loss.png?raw=true" alt="Comparaison des performances" width="800">

## U-MATRIX
<img src="https://github.com/PAR-20/zoid/blob/main/visualisation/u-matrix.png?raw=true" alt="Comparaison des performances" width="800">



### Visualisations
````python
def visualize_som_umatrix(...):
    # Cartographie des similarités entre cas médicaux
    # Identification de clusters pathologiques

```python
Dataset: dataset1
Classes: ['pneumonia', 'normal']
Class distribution: {'pneumonia': 3875, 'normal': 1341}
Sample image sizes: [(1048, 736), (984, 672), (992, 712), (1224, 888), (864, 480)]
--------------------------------------------------

Comparing statistics across all datasets:
Un seul dataset valide (dataset1), pas de comparaison nécessaire.
Exploration summary saved to exploration_summary.csv

Step 1: Data Exploration
Dataset: dataset1
Classes: ['pneumonia', 'normal']
Class distribution: {'pneumonia': 3875, 'normal': 1341}
Sample image sizes: [(1048, 736), (984, 672), (992, 712), (1224, 888), (864, 480)]
--------------------------------------------------

Step 2: Data Preprocessing
Dataset: (5216, 150, 150) images, 2 classes

Step 3: Feature Extraction
Original dimensions: (5216, 22500)
PCA dimensions: (5216, 1405)
Explained variance ratio: 0.9500

Step 4: Traditional ML Models with Simple Train-Test Split
Model: Logistic Regression
Accuracy: 0.9521
Precision: 0.9527
Recall: 0.9521
F1 Score: 0.9523
ROC AUC: 0.9886
--------------------------------------------------
Model: SVM
Accuracy: 0.9761
Precision: 0.9760
Recall: 0.9761
F1 Score: 0.9760
ROC AUC: 0.9957
--------------------------------------------------
Model: Random Forest
Accuracy: 0.8372
Precision: 0.8462
Recall: 0.8372
F1 Score: 0.8142
ROC AUC: 0.9523
--------------------------------------------------
Model: Gradient Boosting
Accuracy: 0.9444
Precision: 0.9443
Recall: 0.9444
F1 Score: 0.9433
ROC AUC: 0.9836
--------------------------------------------------
Model: Neural Network
Accuracy: 0.9550
Precision: 0.9547
Recall: 0.9550
F1 Score: 0.9548
ROC AUC: 0.9904
--------------------------------------------------

Step 5: Cross-Validation Evaluation
Model: Logistic Regression
Mean Accuracy: 0.9515
Std Accuracy: 0.0044
--------------------------------------------------
Model: SVM
Mean Accuracy: 0.9707
Std Accuracy: 0.0040
--------------------------------------------------
Model: Random Forest
Mean Accuracy: 0.8315
Std Accuracy: 0.0112
--------------------------------------------------
Model: Gradient Boosting
Mean Accuracy: 0.9379
Std Accuracy: 0.0067
--------------------------------------------------
Model: Neural Network
Mean Accuracy: 0.9519
Std Accuracy: 0.0090
--------------------------------------------------

Step 6: Parameter Tuning
