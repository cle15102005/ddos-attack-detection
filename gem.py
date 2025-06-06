import torch
import numpy as np

class GEMDetector:
    """
    Generalized Extreme Machine (GEM) Detector for anomaly detection.
    """
    
    def __init__(self, k: int = 10, prototype_k: int = 5, prototype_threshold: float = 0.5, use_cuda: bool = False):
        """
        Initializes the GEMDetector.

        Args:
            k (int): The number of nearest neighbors to consider for anomaly scoring.
                     The anomaly score for a test point is the distance to its k-th
                     nearest neighbor in the learned prototypes.
            prototype_k (int): The number of nearest neighbors to consider during
                               prototype selection. Used to determine if a data point
                               is 'novel' enough to become a new prototype.
                               This is a crucial hyperparameter for the prototype learning phase.
            prototype_threshold (float): The distance threshold for adding new prototypes.
                                         A data point is added as a prototype if its
                                         distance to its prototype_k-th nearest neighbor
                                         in the current prototype set exceeds this value.
                                         This is a critical hyperparameter that needs tuning
                                         to control the number and density of prototypes.
            use_cuda (bool): If True, attempts to use CUDA for computations
                             if a GPU is available.
        """
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(prototype_k, int) or prototype_k <= 0:
            raise ValueError("prototype_k must be a positive integer.")
        if not isinstance(prototype_threshold, (int, float)) or prototype_threshold <= 0:
            raise ValueError("prototype_threshold must be a positive number.")

        self.k = k
        self.prototype_k = prototype_k
        self.prototype_threshold = prototype_threshold
        self.use_cuda = use_cuda

        # Determine the device (CPU or GPU)
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for GEMDetector.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for GEMDetector.")

        # M will store the learned prototypes, which replace the full training_data
        self.prototypes = None
        self.is_fitted = False

    #offline phase
    def fit(self, X: np.ndarray):
        """
        Fits the GEM detector to the training data by learning a set of prototypes (M)
        from the normal data, following Algorithm 1 from the GEM paper.

        Args:
            X (np.ndarray): The training data (feature matrix), assumed to be normal data.
                            Expected shape: (n_samples, n_features).
                            This data should be scaled prior to fitting.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")
        if X.shape[0] == 0:
            raise ValueError("Training data X cannot be empty.")
        if X.shape[0] < self.prototype_k:
             print(f"Warning: Number of training samples ({X.shape[0]}) is less than prototype_k ({self.prototype_k}). "
                   f"This might affect prototype selection. Setting prototype_k to min(num_samples, prototype_k).")
             self.prototype_k = min(X.shape[0], self.prototype_k)


        # Convert training data to PyTorch tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Initialize prototypes (M) with the first data point
        self.prototypes = X_tensor[0:1].clone()
        print(f"Initialized prototypes with 1 sample. Total samples to process: {X_tensor.shape[0]}")

        # Iterate through the rest of the training data to select prototypes
        for i in range(1, X_tensor.shape[0]):
            current_sample = X_tensor[i:i+1] # Ensure it's a 2D tensor (1, n_features)

            # Calculate pairwise Euclidean distances between current_sample and current prototypes
            # torch.cdist(input1, input2, p=2.0) calculates Euclidean distance
            distances = torch.cdist(current_sample, self.prototypes, p=2.0) # Shape: (1, num_prototypes)

            # If there are not enough prototypes for prototype_k, simply add the sample if it's new
            if self.prototypes.shape[0] < self.prototype_k:
                # For small prototype sets, if the sample is not identical to any existing prototype, add it.
                # A more robust check might be needed if exact duplicates are not intended.
                # Here, we assume if it's not exactly identical, it's new enough for a small M.
                # A more robust approach would be to calculate distance to *nearest* prototype.
                min_dist_to_prototypes = distances.min() if distances.shape[1] > 0 else float('inf')
                if min_dist_to_prototypes > 1e-6: # Check if it's not a duplicate
                    self.prototypes = torch.cat((self.prototypes, current_sample), dim=0)
            else:
                # Find the prototype_k smallest distances
                # We need the prototype_k-th smallest distance for the threshold check
                k_smallest_distances, _ = torch.topk(distances, k=self.prototype_k, largest=False, dim=1)

                # The prototype selection criterion: distance to prototype_k-th nearest neighbor
                dk_distance = k_smallest_distances[:, -1].item() # Get the scalar value

                # If the current sample's k-th nearest neighbor distance to M is greater than threshold, add it as a prototype
                if dk_distance > self.prototype_threshold:
                    self.prototypes = torch.cat((self.prototypes, current_sample), dim=0)

        self.is_fitted = True
        print(f"GEMDetector fitted with {self.prototypes.shape[0]} prototypes learned from {X.shape[0]} training samples.")

    #online phase
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts anomaly scores for new data points using the learned prototypes (M).
        The anomaly score for each point is the distance to its k-th nearest
        neighbor among the prototypes. Higher scores indicate a higher
        likelihood of being an anomaly.

        Args:
            X (np.ndarray): The data points for which to calculate anomaly scores.
                            Expected shape: (n_samples, n_features).
                            This data should be scaled prior to prediction.

        Returns:
            np.ndarray: An array of anomaly scores, one for each input sample.
        """
        if not self.is_fitted or self.prototypes is None:
            raise RuntimeError("GEMDetector has not been fitted yet or no prototypes were learned. Call .fit() first.")
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")
        if X.shape[1] != self.prototypes.shape[1]:
            raise ValueError(f"Input data features ({X.shape[1]}) must match "
                             f"prototype features ({self.prototypes.shape[1]}).")
        if self.prototypes.shape[0] < self.k:
            print(f"Warning: Number of learned prototypes ({self.prototypes.shape[0]}) is less than k ({self.k}). "
                  f"Setting k to min(num_prototypes, k).")
            self.k = min(self.prototypes.shape[0], self.k)
            if self.k == 0: # Handle case where no prototypes are learned
                return np.zeros(X.shape[0]) # Or raise an error, depending on desired behavior


        # Convert input data to a PyTorch tensor and move to the selected device
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Calculate pairwise Euclidean distances between test samples and learned prototypes
        # Resulting shape: (n_test_samples, num_prototypes)
        distances = torch.cdist(X_tensor, self.prototypes, p=2.0)

        # Find the k smallest distances for each test sample
        # We need the values, specifically the k-th (index self.k - 1) smallest distance
        k_smallest_distances, _ = torch.topk(distances, k=self.k, largest=False, dim=1)

        # The anomaly score is the distance to the k-th nearest neighbor
        scores = k_smallest_distances[:, -1]

        # Move scores back to CPU and convert to a NumPy array
        return scores.cpu().numpy()