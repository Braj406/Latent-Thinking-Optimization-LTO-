def compute_matrix_entropy(layer_representations):
    """Paper formulation: Matrix-based von Neumann entropy using Gram matrix."""
    # Disable gradient tracking to conserve GPU memory during inference
    with torch.no_grad():
        # Upcast the hidden states to 32-bit floats to prevent small values from rounding to 0.0 (float16 underflow)
        hidden_states_float32 = layer_representations.to(torch.float32)
        # Compute the Gram matrix by multiplying the representation matrix by its transpose
        gram_matrix = torch.matmul(hidden_states_float32, hidden_states_float32.T)
        # Calculate the trace (the sum of the main diagonal elements) of the Gram matrix
        gram_matrix_trace = torch.trace(gram_matrix)

        # If the trace is practically zero, return an entropy of 0.0 to prevent division by zero errors
        if gram_matrix_trace <= 1e-9:
            return 0.0

        # Normalize the Gram matrix by dividing by its trace so its eigenvalues sum to 1 (acting like probabilities)
        normalized_gram_matrix = gram_matrix / gram_matrix_trace
        # Calculate the eigenvalues of the normalized, symmetric Gram matrix
        eigenvalues = torch.linalg.eigvalsh(normalized_gram_matrix)
        # Filter out any negative or zero eigenvalues caused by floating-point precision inaccuracies
        valid_eigenvalues = eigenvalues[eigenvalues > 1e-9]
        # Compute the von Neumann entropy by applying the Shannon entropy formula to the valid eigenvalues
        entropy_value = -torch.sum(valid_eigenvalues * torch.log(valid_eigenvalues)).item()
        # Return the calculated scalar entropy
        return entropy_value

def calculate_effective_rank(layer_representations):
    """Paper formulation: Effective Rank via Shannon entropy of singular values."""
    # Disable gradient computation to save memory
    with torch.no_grad():
        # Cast the representations to float32 for mathematical stability
        hidden_states_float32 = layer_representations.to(torch.float32)

        # Attempt to compute the Singular Value Decomposition (SVD) of the matrix
        try:
            # We only extract the singular values (the middle tensor), discarding the left and right singular vectors
            _, singular_values, _ = torch.linalg.svd(hidden_states_float32, full_matrices=False)
        # Catch exceptions where the SVD algorithm fails to converge on the GPU
        except RuntimeError:
            # Return a default effective rank of 1.0 if the SVD fails
            return 1.0

        # Filter out singular values that are practically zero to avoid log(0) errors
        valid_singular_values = singular_values[singular_values > 1e-9]
        # Sum the remaining valid singular values to use as a normalization denominator
        sum_of_singular_values = torch.sum(valid_singular_values)
        # If the sum of singular values is zero, the effective rank is minimal (1.0)
        if sum_of_singular_values <= 1e-9:
            return 1.0

        # Normalize the singular values so they sum to 1, creating a probability distribution
        singular_value_probabilities = valid_singular_values / sum_of_singular_values
        # Calculate the Shannon entropy of this singular value probability distribution
        singular_value_entropy = -torch.sum(singular_value_probabilities * torch.log(singular_value_probabilities)).item()
        # The effective rank is defined as the exponential of the calculated entropy
        effective_rank_value = math.exp(singular_value_entropy)
        # Return the final calculated effective rank
        return effective_rank_value

def calculate_anisotropy(layer_representations):
    """Paper formulation: Anisotropy (Variance explained by dominant orientation)."""
    # Disable gradients for memory efficiency
    with torch.no_grad():
        # Convert the input matrix to 32-bit floats for higher precision math
        hidden_states_float32 = layer_representations.to(torch.float32)
        # Calculate the mean representation vector across all tokens (dimension 0)
        mean_representation_vector = hidden_states_float32.mean(dim=0, keepdim=True)
        # Mean-center the data by subtracting the mean vector from every token's representation
        centered_hidden_states = hidden_states_float32 - mean_representation_vector
        # Compute the unnormalized covariance matrix by multiplying the centered data by its transpose
        covariance_matrix = torch.matmul(centered_hidden_states, centered_hidden_states.T)
        # Extract the singular values (which are equivalent to eigenvalues for this symmetric covariance matrix)
        eigenvalues = torch.linalg.svdvals(covariance_matrix)
        # Sum all the eigenvalues to determine the total variance in the latent space
        total_variance = torch.sum(eigenvalues)
        # If the total variance is near zero, the space is maximally anisotropic (collapsed), return 1.0
        if total_variance <= 1e-9:
            return 1.0
        # Anisotropy is the ratio of the largest eigenvalue (variance in the dominant direction) to the total variance
        anisotropy_value = (eigenvalues[0] / total_variance).item()
        # Return the computed anisotropy ratio
        return anisotropy_value

def calculate_intrinsic_dimension(layer_representations, trimming_factor=0.1):
    """Paper formulation: Intrinsic Dimension via Two-Nearest-Neighbor Estimation."""
    # Disable gradient tracking
    with torch.no_grad():
        # Upcast the input to float32 to ensure accurate distance calculations
        hidden_states_float32 = layer_representations.to(torch.float32)
        # Count the number of points (tokens) present in the representation matrix
        num_tokens = hidden_states_float32.shape[0]

        # The algorithm requires at least 3 points to find a 1st and 2nd nearest neighbor; return 0.0 if not met
        if num_tokens < 3:
            return 0.0

        # Calculate the standard Euclidean (L2) distance between every pair of token representations
        pairwise_distances = torch.cdist(hidden_states_float32, hidden_states_float32, p=2)
        # Initialize lists to store the distance to each token's first and second nearest neighbors
        nearest_neighbor_distances = []
        second_nearest_neighbor_distances = []

        # Iterate through every token to identify its specific neighbors
        for i in range(num_tokens):
            # Sort the distances from token 'i' to all other tokens in ascending order
            sorted_distances_from_token, _ = torch.sort(pairwise_distances[i])
            # Append the distance to the closest neighbor (Index 1, because Index 0 is the token itself at distance 0)
            nearest_neighbor_distances.append(sorted_distances_from_token[1])
            # Append the distance to the second closest neighbor (Index 2)
            second_nearest_neighbor_distances.append(sorted_distances_from_token[2])

        # Convert the Python lists of distances into PyTorch tensors
        nearest_neighbor_tensor = torch.stack(nearest_neighbor_distances)
        second_nearest_neighbor_tensor = torch.stack(second_nearest_neighbor_distances)
        # Clamp the nearest neighbor distances to a tiny minimum value to prevent division by zero in the next step
        clamped_nearest_neighbor_tensor = torch.clamp(nearest_neighbor_tensor, min=1e-7)
        # Calculate the ratio of the second nearest distance to the first nearest distance (mu)
        distance_ratios = second_nearest_neighbor_tensor / clamped_nearest_neighbor_tensor
        # Sort these distance ratios in ascending order to prepare for the empirical cumulative distribution
        sorted_distance_ratios, _ = torch.sort(distance_ratios)

        # Initialize lists for the X and Y coordinates needed for the final linear regression
        log_distance_ratios_x = []
        empirical_cumulative_distribution_y = []

        # Loop through the sorted ratios to generate the regression points
        for rank in range(1, num_tokens + 1):
            # Calculate the empirical cumulative distribution function (CDF) value
            empirical_cdf = rank / num_tokens
            # Ensure the distance ratio is not effectively zero before taking the logarithm
            safe_ratio_value = max(sorted_distance_ratios[rank - 1].item(), 1e-9)
            # X-coordinate: The natural logarithm of the distance ratio
            log_distance_ratios_x.append(math.log(safe_ratio_value))
            # Y-coordinate: -log(1 - CDF), adding a tiny epsilon to prevent taking the log of zero at the final rank
            empirical_cumulative_distribution_y.append(-math.log(1 - empirical_cdf + 1e-9))

        # Apply the trimming factor to remove extreme outlier distance ratios from the dataset
        if trimming_factor > 0:
            # Determine how many data points to keep based on the trimming factor percentage
            num_points_to_keep = int((1 - trimming_factor) * num_tokens)
            # Slice the X coordinates list to keep only the non-outlier points
            log_distance_ratios_x = log_distance_ratios_x[:num_points_to_keep]
            # Slice the Y coordinates list to match the trimmed X coordinates
            empirical_cumulative_distribution_y = empirical_cumulative_distribution_y[:num_points_to_keep]

        # Convert the final trimmed X and Y coordinate lists into PyTorch tensors
        x_tensor = torch.tensor(log_distance_ratios_x)
        y_tensor = torch.tensor(empirical_cumulative_distribution_y)

        # Calculate the denominator for the least-squares regression line passing through the origin (sum of X^2)
        regression_denominator = torch.sum(x_tensor ** 2)
        # If the denominator is zero (all X values are 0), return an intrinsic dimension of 0.0 to prevent math errors
        if regression_denominator == 0:
            return 0.0

        # Calculate the slope of the regression line (sum(X*Y) / sum(X^2)), which mathematically estimates the intrinsic dimension
        intrinsic_dimension_estimate = (torch.sum(x_tensor * y_tensor) / regression_denominator).item()
        # Return the final calculated intrinsic dimension
        return intrinsic_dimension_estimate
