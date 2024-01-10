import random

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

