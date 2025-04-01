import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import cv2

# Load datasets (adjust paths as needed)
# Assuming you have 3 datasets in different folders
dataset_paths = [
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset1',
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset2',
    '/Users/dalm1/Desktop/reroll/Progra/par20/dataset3'
]

# Function to explore image dataset
def explore_image_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    class_distribution = {}
    image_sizes = []
    
    for class_name in classes:
        if os.path.isdir(os.path.join(dataset_path, class_name)):
            class_path = os.path.join(dataset_path, class_name)
            images = os.listdir(class_path)
            class_distribution[class_name] = len(images)
            
            # Sample a few images to check dimensions
            for img_name in images[:5]:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                image_sizes.append(img.size)
    
    print(f"Dataset: {dataset_path}")
    print(f"Classes: {classes}")
    print(f"Class distribution: {class_distribution}")
    print(f"Sample image sizes: {image_sizes[:5]}")
    print("-" * 50)
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
    plt.title(f"Class Distribution in {os.path.basename(dataset_path)}")
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Display sample images
    plt.figure(figsize=(15, 10))
    for i, class_name in enumerate(classes):
        if os.path.isdir(os.path.join(dataset_path, class_name)):
            class_path = os.path.join(dataset_path, class_name)
            images = os.listdir(class_path)
            if images:
                plt.subplot(1, len(classes), i+1)
                img_path = os.path.join(class_path, images[0])
                img = plt.imread(img_path)
                plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                plt.title(class_name)
                plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Analyze image intensity distribution
    plt.figure(figsize=(15, 5))
    for i, class_name in enumerate(classes):
        if os.path.isdir(os.path.join(dataset_path, class_name)):
            class_path = os.path.join(dataset_path, class_name)
            images = os.listdir(class_path)
            if images:
                # Sample up to 10 images for intensity analysis
                sample_images = images[:min(10, len(images))]
                intensities = []
                
                for img_name in sample_images:
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        intensities.append(img.flatten())
                
                if intensities:
                    # Concatenate all pixel values
                    all_intensities = np.concatenate(intensities)
                    
                    plt.subplot(1, len(classes), i+1)
                    plt.hist(all_intensities, bins=50, alpha=0.7)
                    plt.title(f"{class_name} Intensity")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    return class_distribution, image_sizes

def analyze_dataset_statistics(dataset_paths):
    """Analyze and compare statistics across datasets"""
    all_distributions = []
    all_sizes = []
    
    for path in dataset_paths:
        dist, sizes = explore_image_dataset(path)
        all_distributions.append(dist)
        all_sizes.append(sizes)
    
    # Compare class balance across datasets
    plt.figure(figsize=(12, 6))
    dataset_names = [os.path.basename(path) for path in dataset_paths]
    
    # Create a DataFrame for easier plotting
    df_list = []
    for i, dist in enumerate(all_distributions):
        for class_name, count in dist.items():
            df_list.append({
                'Dataset': dataset_names[i],
                'Class': class_name,
                'Count': count
            })
    
    if df_list:
        df = pd.DataFrame(df_list)
        
        # Plot class distribution comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Class', y='Count', hue='Dataset', data=df)
        plt.title('Class Distribution Comparison Across Datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Compare image size distributions
    plt.figure(figsize=(12, 6))
    for i, sizes in enumerate(all_sizes):
        if sizes:
            widths = [size[0] for size in sizes]
            heights = [size[1] for size in sizes]
            plt.scatter(widths, heights, alpha=0.5, label=dataset_names[i])
    
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Image Dimensions Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Fonction pour télécharger un dataset depuis Hugging Face
def download_dataset_from_huggingface(dataset_name, save_dir='/Users/dalm1/Desktop/reroll/Progra/par20/downloaded_dataset'):
    """
    Télécharge un dataset depuis Hugging Face et le sauvegarde localement
    
    Parameters:
    - dataset_name: Nom du dataset sur Hugging Face (ex: 'keremberke/chest-xray-classification')
    - save_dir: Répertoire où sauvegarder le dataset
    
    Returns:
    - Chemin vers le dataset téléchargé
    """
    try:
        from datasets import load_dataset
        import shutil
        
        print(f"Téléchargement du dataset '{dataset_name}' depuis Hugging Face...")
        
        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)
        
        # Télécharger le dataset
        dataset = load_dataset(dataset_name)
        
        # Créer des sous-répertoires pour chaque classe
        if 'train' in dataset:
            if 'label' in dataset['train'].features:
                # Cas où le dataset a des labels numériques
                label_names = dataset['train'].features['label'].names
                for label_id, label_name in enumerate(label_names):
                    os.makedirs(os.path.join(save_dir, label_name), exist_ok=True)
                
                # Sauvegarder les images par classe
                for example in dataset['train']:
                    img = example['image']
                    label_id = example['label']
                    label_name = label_names[label_id]
                    
                    # Sauvegarder l'image
                    img_path = os.path.join(save_dir, label_name, f"{example['id'] if 'id' in example else hash(str(example))}.png")
                    img.save(img_path)
            
            elif 'image' in dataset['train'].features and isinstance(dataset['train'][0]['image'], Image.Image):
                # Cas où le dataset contient directement des images sans labels
                os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
                
                for i, example in enumerate(dataset['train']):
                    img = example['image']
                    img_path = os.path.join(save_dir, 'images', f"image_{i}.png")
                    img.save(img_path)
        
        print(f"Dataset téléchargé et sauvegardé dans {save_dir}")
        return save_dir
    
    except ImportError:
        print("Veuillez installer les packages nécessaires avec:")
        print("pip install datasets huggingface_hub")
        return None
    except Exception as e:
        print(f"Erreur lors du téléchargement du dataset: {str(e)}")
        return None

# Explore each dataset individually
for dataset_path in dataset_paths:
    explore_image_dataset(dataset_path)

# Compare statistics across datasets
print("\nComparing statistics across all datasets:")
analyze_dataset_statistics(dataset_paths)

# Save a summary of the exploration
def save_exploration_summary(dataset_paths, output_file='/Users/dalm1/Desktop/reroll/Progra/par20/exploration_summary.csv'):
    """Save a summary of the dataset exploration to a CSV file"""
    summary_data = []
    
    for path in dataset_paths:
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        total_images = 0
        class_counts = {}
        
        for class_name in classes:
            class_path = os.path.join(path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
        
        dataset_name = os.path.basename(path)
        summary_data.append({
            'Dataset': dataset_name,
            'Total Images': total_images,
            'Number of Classes': len(classes),
            'Classes': ', '.join(classes),
            'Class Distribution': str(class_counts)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Exploration summary saved to {output_file}")

# Save exploration summary
save_exploration_summary(dataset_paths)

if __name__ == "__main__":
    # Option pour télécharger un dataset depuis Hugging Face
    download_from_huggingface = input("Voulez-vous télécharger un dataset depuis Hugging Face? (oui/non): ").lower() == 'oui'
    
    if download_from_huggingface:
        dataset_name = input("Entrez le nom du dataset Hugging Face (ex: 'keremberke/chest-xray-classification'): ")
        downloaded_path = download_dataset_from_huggingface(dataset_name)
        
        if downloaded_path:
            # Ajouter le dataset téléchargé à la liste des datasets à explorer
            dataset_paths.append(downloaded_path)
    
    # Explore each dataset individually
    for dataset_path in dataset_paths:
        explore_image_dataset(dataset_path)
    
    # Compare statistics across datasets
    print("\nComparing statistics across all datasets:")
    analyze_dataset_statistics(dataset_paths)
    
    # Save exploration summary
    save_exploration_summary(dataset_paths)
    
    print("Data exploration complete!")