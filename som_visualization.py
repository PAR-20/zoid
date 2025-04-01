import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

def train_som(X, map_size=(10, 10), sigma=1.0, learning_rate=0.5, iterations=1000):
    """
    Train a Self-Organizing Map

    Parameters:
    - X: Input data
    - map_size: Size of the SOM grid
    - sigma: Spread of the neighborhood function
    - learning_rate: Initial learning rate
    - iterations: Number of training iterations

    Returns:
    - Trained SOM
    """
    # Normalize data to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize SOM
    som = MiniSom(map_size[0], map_size[1], X_scaled.shape[1],
                  sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian')

    # Random initialization of weights
    som.random_weights_init(X_scaled)

    # Train SOM
    som.train_random(X_scaled, iterations)

    return som, scaler, X_scaled

def visualize_som_umatrix(som, X_scaled, y=None, map_size=(10, 10)):
    """Visualize SOM U-Matrix"""
    plt.figure(figsize=(10, 10))

    # U-Matrix (distance between neighboring neurons)
    umatrix = som.distance_map()
    plt.subplot(1, 1, 1)
    plt.imshow(umatrix, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title('U-Matrix')

    # If labels are provided, mark winning neurons
    if y is not None:
        # Find winning neuron for each sample
        win_map = {}
        for i, x in enumerate(X_scaled):
            winner = som.winner(x)
            if winner not in win_map:
                win_map[winner] = []
            win_map[winner].append(i)

        # Plot markers for each class
        for winner, indices in win_map.items():
            # Get most common class for this winner
            if len(indices) > 0:
                most_common_class = np.bincount(y[indices]).argmax()
                plt.plot(winner[0] + 0.5, winner[1] + 0.5,
                         'o', markerfacecolor='None',
                         markeredgecolor=plt.cm.tab10(most_common_class),
                         markersize=12, markeredgewidth=2)

    plt.tight_layout()
    plt.show()

def visualize_som_component_planes(som, feature_names, map_size=(10, 10)):
    """Visualize SOM component planes"""
    n_features = len(feature_names)
    cols = min(3, n_features)
    rows = int(np.ceil(n_features / cols))

    plt.figure(figsize=(15, rows * 4))

    for i, feature_name in enumerate(feature_names):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(som.get_weights()[:, :, i], cmap='coolwarm')
        plt.colorbar()
        plt.title(f'Component Plane: {feature_name}')

    plt.tight_layout()
    plt.show()
