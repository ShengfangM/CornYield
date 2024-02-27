import random
import numpy as np


def train_test_split_indices(indices, test_size=0.2, random_seed=None):
    """Randomly split a list of indices into training and test sets."""
    if random_seed is not None:
        random.seed(random_seed)

    # Shuffle the indices randomly
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)

    # Calculate the split point based on the test_size
    split_point = int(len(shuffled_indices) * (1 - test_size))

    # Split the indices into training and test sets
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]

    return train_indices, test_indices


def adaptive_avg_pool2d(input_array, output_size):
    # Input dimensions
    _, _, input_height, input_width = input_array.shape
    
    # Output dimensions
    output_height, output_width = output_size
    
    # Calculate pooling size for each dimension
    pool_height = input_height // output_height
    pool_width = input_width // output_width
    
    output_array = np.zeros((input_array.shape[0], input_array.shape[1], output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            # Compute the average value within the region
            output_array[:,:,i,j] = np.mean(input_array[:, :, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width], axis=(2, 3))
    
    return output_array

