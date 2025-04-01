import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_images(image_paths, target_size=(150, 150), apply_clahe=True):
    """
    Preprocess images for model training

    Parameters:
    - image_paths: List of image file paths
    - target_size: Size to resize images to
    - apply_clahe: Whether to apply CLAHE for contrast enhancement

    Returns:
    - Preprocessed images as numpy array
    """
    processed_images = []

    for img_path in image_paths:
        # Load image
        img = cv2.imread(img_path)

        # Convert to grayscale if color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize
        img = cv2.resize(img, target_size)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # Normalize pixel values to [0, 1]
        img = img / 255.0

        processed_images.append(img)

    return np.array(processed_images)

def extract_features(images, n_components=0.95):
    """
    Extract features using PCA

    Parameters:
    - images: Preprocessed images
    - n_components: Number of PCA components or variance to retain

    Returns:
    - PCA features
    """
    # Flatten images
    n_samples = len(images)
    flat_images = images.reshape(n_samples, -1)

    # Standardize
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(flat_images)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(scaled_images)

    print(f"Original dimensions: {flat_images.shape}")
    print(f"PCA dimensions: {pca_features.shape}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    return pca_features, pca, scaler

def extract_texture_features(images):
    """Extract texture features using GLCM"""
    pass

def extract_shape_features(images):
    """Extract shape features"""
    pass
