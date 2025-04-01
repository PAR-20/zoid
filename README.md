# Zoidberg
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

I'll help update the README.md with a beginner-friendly explanation of the regression aspects in your project. Here's the addition:

```markdown
## 🔍 For Beginners: Understanding the Medical Image Analysis

### 🤖 How the Computer "Learns" to Diagnose

The system uses special math patterns (called "machine learning models") to analyze X-ray images. Here's a simple explanation of what's happening:

1. **Image Preparation** (Like Organizing X-Rays):
   - Convert images to black & white numbers
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
 ```
```

This addition explains the technical aspects in simple terms while:

1. Using medical analogies beginners can relate to
2. Explaining Logistic Regression as a decision boundary
3. Connecting the technical process to real-world diagnosis
4. Maintaining visual appeal with emojis and spacing
Would you like me to add any other specific explanations or adjust the language?

## DALM1
Copy

Delete

README.md tu peux me faire un papier qui detaille exactement les etapes utilisé pour faire les recursions ect detaille le projet les but en entier ect

## Trae AI
Thought process

README.md

README.md 5-5

Okay, the user wants me to update their README.md file to detail the entire project, including the steps, goals, and technical aspects like recursions. Let me start by looking at the current README they provided, which is very minimal—just a title and an image.

First, I need to structure the README properly. Common sections include Project Overview, Features, Installation, Usage, Results, and Conclusion. Since the user mentioned explaining recursions and detailed steps, I should include a section on methodology or approach where I can elaborate on the algorithms and processes used.

Looking at the code snippets provided earlier, the project involves data exploration, preprocessing, model training (including traditional ML and deep learning), and visualization. I should break down each of these steps. The user also mentioned regression, so I need to explain how logistic regression fits into their classification task, maybe clarifying that it's used for binary classification here.

I should mention the datasets used, like the chest X-ray dataset from Hugging Face, and the techniques applied, such as CLAHE for image preprocessing, SVM, Random Forest, and CNN models. Including details about handling class imbalance and using SOM for visualization would add depth.

For the recursion part, I need to check if there's any recursive code in their project. From the provided code snippets, I don't see explicit recursion. Maybe the user is referring to iterative processes or cross-validation steps. I should clarify that or ensure that any mention of recursion is accurate based on their code.

I'll structure the README with sections in both French and English to cater to different readers, given the user's mix of French and English in prompts. Including visual elements like the accuracy comparison chart and SOM visualizations will make it more engaging.

I should also add installation instructions, mentioning the requirements.txt file, and usage steps to guide users on how to run the project. Including results and conclusions will highlight the project's outcomes, like the SVM model's high accuracy.

Finally, I'll make sure the language is accessible for beginners, explaining technical terms in simple terms, and using emojis or icons to make sections visually distinct. I'll link to the dataset and provide references for further reading to add credibility and resources for users who want to explore more.


# Projet de Classification d'Images Médicales
```
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

```plaintext

## 🔧 Méthodologie Technique

### 1. Préprocessing des Images


def preprocess_images(...):
    # Conversion en niveaux de gris
    # Redimensionnement 150x150 pixels
    # Application de CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Normalisation des valeurs de pixels [0-1]
```

### 2. Extraction de Caractéristiques
# Approche Traditionnelle

def extract_features(images):
    # Calcul d'histogrammes de textures
    # Analyse de motifs locaux (LBP)
    # Réduction de dimension avec PCA

Approche Deep Learning :


```
def create_cnn_model(...):
```
    # Architecture CNN avec 3 couches convolutionnelles
    # MaxPooling pour réduire la dimension spatiale
    # Couche Dense finale avec activation sigmoïde

### 3. Entraînement des Modèles
Modèles Testés :

- Régression Logistique
- SVM (Machine à Vecteurs de Support)
- Random Forest
- Gradient Boosting
- Réseau de Neurones Artificiel (MLP)
- CNN (Réseau de Neurones Convolutif)
Validation Croisée :

````
kf = KFold(n_splits=5, shuffle=True)
for model in models:
    scores = cross_val_score(model, X, y, cv=kf)


## 📊 Résultats Clés
### Performances des Modèles Modèle Accuracy Temps d'Exécution Régression Logistique

95.21%

2.1s SVM

97.61%

18.4s Random Forest

83.72%

4.8s CNN

94.50%

2min34s

### Visualisations
````
def visualize_som_umatrix(...):
    # Cartographie des similarités entre cas médicaux
    # Identification de clusters pathologiques
