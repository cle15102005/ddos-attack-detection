import torch
import numpy as np
import math
from sklearn.model_selection import train_test_split # For S1/S2 partitioning

class GEMDetector:
    """
    GEM-based Nonparametric Anomaly Detection Algorithm
    as described in the provided images (Mehmet Necip Kurt et al. paper).
    This implementation strictly follows the offline and online phases.
    """

    def __init__(self, k: int = 10, s1_ratio: float = 0.8, alpha: float = 0.05,
                 cusum_h: float = None, use_cuda: bool = False):
        """
        Initializes the GEMDetector.

        Args:
            k (int): The number of nearest neighbors for calculating d_j and d_t
                     (sum of distances to kNNs).
            s1_ratio (float): The ratio of the nominal dataset X to be used for S1.
                              S2 will be (1 - s1_ratio) * N_samples.
            alpha (float): Significance level for anomaly detection. This parameter
                           is used to derive the CUSUM threshold 'h' if not provided.
                           (e.g., 0.05 for 95% confidence).
            cusum_h (float, optional): CUSUM decision interval threshold. If g_t
                                       exceeds this, an anomaly is declared.
                                       If None, it will be calibrated using S2 data.
            use_cuda (bool): If True, attempts to use CUDA for computations.
        """
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(s1_ratio, (int, float)) or not (0 < s1_ratio < 1):
            raise ValueError("s1_ratio must be a float between 0 and 1.")
        if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
            raise ValueError("alpha must be a float between 0 and 1.")

        self.k = k
        self.s1_ratio = s1_ratio
        self.alpha = alpha
        self.cusum_h = cusum_h

        # Device setup
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for GEMDetector.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for GEMDetector.")

        # Stored components from offline phase
        self.S1 = None # S1: reference set for kNN searches
        self.nominal_summary_statistics = None # d_j values from S2 against S1
        self.N2 = 0 # Size of S2

        self.is_fitted = False

        # CUSUM state for online phase
        self.g_t = 0.0 # g_t statistic for online detection

    def _calculate_k_nn_sum_distances(self, data_points: torch.Tensor, reference_set: torch.Tensor, k: int) -> torch.Tensor:
        """
        Calculates the sum of Euclidean distances to the k nearest neighbors
        for each point in 'data_points' with respect to 'reference_set'.

        Args:
            data_points (torch.Tensor): The points for which to find k-NN sums (N, D).
            reference_set (torch.Tensor): The set of potential neighbors (M, D).
            k (int): The number of nearest neighbors to sum distances for.

        Returns:
            torch.Tensor: A tensor of sum of k-NN distances (N,).
        """
        if reference_set.shape[0] == 0:
            # If reference set is empty, distances are infinite
            return torch.full((data_points.shape[0],), float('inf')).to(self.device)

        # Calculate all pairwise Euclidean distances
        distances = torch.cdist(data_points, reference_set, p=2.0) # Shape: (N, M)

        # Find the k smallest distances for each data point
        # torch.topk returns values and indices. We need the values.
        # We also handle the case where k is larger than the reference_set size
        actual_k = min(k, reference_set.shape[0])
        
        # If actual_k is 0 (e.g. reference_set was size 0), sum is 0
        if actual_k == 0:
            return torch.zeros(data_points.shape[0]).to(self.device)

        k_smallest_distances, _ = torch.topk(distances, k=actual_k, largest=False, dim=1)

        # Sum these k smallest distances for each data point
        sum_k_distances = torch.sum(k_smallest_distances, dim=1) # Shape: (N,)
        return sum_k_distances

    def fit(self, X: np.ndarray):
        """
        Fits the GEM detector to the training data (nominal dataset X)
        following the Offline phase described in the paper.

        Args:
            X (np.ndarray): The training data (feature matrix), assumed to be normal data.
                            Expected shape: (n_samples, n_features).
                            This data should be scaled prior to fitting.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")
        if X.shape[0] < self.k:
            raise ValueError(f"Number of training samples ({X.shape[0]}) must be at least k ({self.k}).")
        if X.shape[0] < 2:
            raise ValueError("Training data must have at least 2 samples for S1/S2 partitioning.")

        X_tensor = torch.from_numpy(X).float().to(self.device)

        # 1. Uniformly partition X into S1 and S2
        # Use sklearn's train_test_split to handle splitting and shuffling
        S1_np, S2_np = train_test_split(X, train_size=self.s1_ratio, random_state=42) # Fixed random_state for reproducibility

        self.S1 = torch.from_numpy(S1_np).float().to(self.device)
        self.N2 = S2_np.shape[0]
        S2_tensor = torch.from_numpy(S2_np).float().to(self.device)
        
        if self.S1.shape[0] == 0:
            raise ValueError("S1 is empty. Adjust s1_ratio or provide more data.")
        if self.N2 == 0:
            raise ValueError("S2 is empty. Adjust s1_ratio or provide more data.")
        if self.S1.shape[0] < self.k:
            print(f"Warning: S1 size ({self.S1.shape[0]}) is less than k ({self.k}). k for d_j and d_t calculation will be min(S1.shape[0], k).")


        print(f"Dataset X partitioned: S1={self.S1.shape[0]} samples, S2={self.N2} samples.")
        
        # 2. For each data point x_j in S2, search for kNNs among the set S1
        #    Compute d_j = sum of distances to its kNNs in S1.
        self.nominal_summary_statistics = self._calculate_k_nn_sum_distances(S2_tensor, self.S1, self.k)
        
        # 3. Sort {d_j : x_j subset S2} in ascending order.
        self.nominal_summary_statistics = torch.sort(self.nominal_summary_statistics).values # Store sorted values

        print(f"Nominal summary statistics (d_j) calculated for {self.N2} samples.")

        # 4. Calibrate CUSUM threshold (h) if not provided.
        # This requires simulating the g_t behavior for normal data (S2).
        # We need to estimate s_hat_t for each d_j in S2 against the full nominal_summary_statistics
        
        if self.cusum_h is None:
            g_t_values_on_s2 = []
            current_g_t_calibration = 0.0

            # Iterate through the nominal_summary_statistics to simulate CUSUM on S2
            # This approximates the behavior of g_t on normal data
            for dj_value in self.nominal_summary_statistics:
                # Calculate p_hat_t for dj_value against the entire nominal_summary_statistics (d_j set)
                p_hat_t_calib = (self.nominal_summary_statistics <= dj_value).sum().item() / self.N2
                
                # Handle p_hat_t = 0 (or very small) for log
                p_hat_t_calib = max(1e-10, p_hat_t_calib) 
                
                s_hat_t_calib = math.log(p_hat_t_calib)
                
                # CUSUM formula from paper: g_t = max(0, g_t-1 + s_t)
                current_g_t_calibration = max(0.0, current_g_t_calibration + s_hat_t_calib)
                g_t_values_on_s2.append(current_g_t_calibration)
            
            if len(g_t_values_on_s2) > 0:
                # Set cusum_h as a high percentile of the g_t values observed on normal S2 data.
                # The alpha parameter can be used here. A (1-alpha) percentile is a common choice.
                self.cusum_h = np.percentile(g_t_values_on_s2, (1 - self.alpha) * 100)
                print(f"CUSUM threshold (h) calibrated to {(1 - self.alpha) * 100:.2f}th percentile of g_t on S2: {self.cusum_h:.4f}")
            else:
                self.cusum_h = 1.0 # Default if no S2 samples or calibration issues
                print("Warning: Could not calibrate CUSUM threshold (h). Setting to default 1.0.")

        self.is_fitted = True
        print("GEMDetector offline phase completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs real-time anomaly detection on new data points
        following the Online phase described in the paper.

        Args:
            X (np.ndarray): The data points for which to calculate anomaly scores and
                            make real-time anomaly declarations.
                            Expected shape: (n_samples, n_features).
                            This data should be scaled prior to prediction.

        Returns:
            np.ndarray: A 2D array where the first column is the instantaneous d_t score,
                        and the second column is binary anomaly declarations (1 for anomaly, 0 for normal).
        """
        if not self.is_fitted:
            raise RuntimeError("GEMDetector has not been fitted yet. Call .fit() first.")
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")
        if X.shape[1] != self.S1.shape[1]:
            raise ValueError(f"Input data features ({X.shape[1]}) must match "
                             f"S1 features ({self.S1.shape[1]}).")

        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        instantaneous_dt_scores = np.zeros(X.shape[0])
        anomaly_declarations = np.zeros(X.shape[0], dtype=int)

        # Process each data point sequentially, as per the online phase
        for i in range(X_tensor.shape[0]):
            x_t = X_tensor[i:i+1] # Ensure it's a 2D tensor (1, n_features)

            # 1. Search for kNNs of x_t among the set S1. Compute d_t.
            dt_value = self._calculate_k_nn_sum_distances(x_t, self.S1, self.k).item()
            instantaneous_dt_scores[i] = dt_value

            # 2. Compute p_hat_t (empirical tail probability)
            # p_hat_t = (1/N2) * sum_{j=1 to N2} sign(d_t > d_j)
            # This means the fraction of d_j values (from S2) that are smaller than d_t.
            # Since self.nominal_summary_statistics is sorted, we can use searchsorted.
            num_dj_less_than_dt = torch.searchsorted(self.nominal_summary_statistics, dt_value, right=True).item()
            p_hat_t = num_dj_less_than_dt / self.N2
            
            # Handle p_hat_t = 0 case for log (add a small epsilon)
            p_hat_t = max(1e-10, p_hat_t) # Ensure p_hat_t is never exactly zero for log

            # 3. Compute s_hat_t = ln(p_hat_t)
            s_hat_t = math.log(p_hat_t)

            # 4. Compute g_t = max(0, g_t-1 + s_t_hat)
            self.g_t = max(0.0, self.g_t + s_hat_t)

            # 5. Declare an anomaly if g_t > h
            if self.g_t > self.cusum_h:
                anomaly_declarations[i] = 1
                # Reset g_t to 0 upon anomaly declaration ("stop the procedure" and restart)
                self.g_t = 0.0

        return np.column_stack((instantaneous_dt_scores, anomaly_declarations))

    def reset_cusum_state(self):
        """Resets the g_t CUSUM statistic to 0."""
        self.g_t = 0.0
        print("CUSUM state (g_t) reset.")