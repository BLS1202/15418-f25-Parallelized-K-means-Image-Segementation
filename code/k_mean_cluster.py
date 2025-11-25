import cv2
import numpy as np

def manual_kmeans(pixel_data, k, max_iterations=10, epsilon=1.0):
    """
    Performs K-Means clustering from scratch using NumPy.

    Args:
        pixel_data (np.array): 2D array of pixels (num_pixels, 3).
        k (int): The number of clusters.
        max_iterations (int): Maximum number of iterations to perform.
        epsilon (float): The threshold for convergence. If the total change in
                         centroids is less than this, the algorithm stops.

    Returns:
        tuple: A tuple containing:
            - labels (np.array): The cluster index for each pixel.
            - centroids (np.array): The final cluster centroids (colors).
    """
    # 1. Initialization: Randomly select K pixels as initial centroids
    num_pixels = pixel_data.shape[0]
    random_indices = np.random.choice(num_pixels, size=k, replace=False)
    centroids = pixel_data[random_indices]
    
    print(f"Starting K-Means with K={k}.")

    for i in range(max_iterations):
        # A faster, vectorized version of the assignment step
        # Calculate distances from all pixels to all centroids at once using broadcasting
        # Shape: (num_pixels, 1, 3) - (k, 3) -> (num_pixels, k, 3)
        distances = np.linalg.norm(pixel_data[:, np.newaxis] - centroids, axis=2)
        
        # 2. Assignment Step: Assign labels based on the minimum distance
        labels = np.argmin(distances, axis=1)

        # 3. Update Step: Recalculate centroids as the mean of assigned pixels
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            # Get all pixels assigned to the current cluster
            points_in_cluster = pixel_data[labels == cluster_idx]
            
            # If a cluster is empty, keep its old centroid
            if len(points_in_cluster) > 0:
                new_centroids[cluster_idx] = np.mean(points_in_cluster, axis=0)
            else:
                # Re-initialize empty clusters to a random pixel to avoid collapse
                new_centroids[cluster_idx] = pixel_data[np.random.choice(num_pixels)]

        # 4. Convergence Check
        # Calculate the total distance the centroids moved
        centroid_shift = np.sum(np.linalg.norm(new_centroids - centroids, axis=1))
        
        # Update centroids for the next iteration
        centroids = new_centroids
        
        print(f"Iteration {i+1}/{max_iterations}: Centroid shift = {centroid_shift:.2f}")

        # If the centroids moved less than our epsilon threshold, we've converged
        if centroid_shift < epsilon:
            print("Converged!")
            break
            
    return labels, centroids.astype(np.uint8)


# --- Main execution ---
if __name__ == '__main__':
    # 1. Configuration
    image_path = '../img/camera_man.jpg'  # Replace with your image path
    K = 8                      # The number of clusters (colors)

    # 2. Load the Image
    # OpenCV loads images in BGR format. We'll work directly in BGR.
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'")
        exit()

    # Get image dimensions
    height, width, _ = original_image.shape

    # 3. Prepare Data for K-Means
    # Reshape the image to be a list of pixels (N_pixels, 3) 
    # (number of pixels) * (R, G, B)
    pixel_data = original_image.reshape((-1, 3))
    
    # Convert to float32 for K-Means calculations
    pixel_data = np.float32(pixel_data)
    
    # 4. Run the manual K-Means algorithm
    labels, centroids = manual_kmeans(pixel_data, K)
    
    # 5. Reconstruct the Segmented Image
    # Map each pixel's label to its corresponding centroid color
    segmented_data = centroids[labels]
    
    # Reshape the segmented data back to the original image dimensions
    segmented_image = segmented_data.reshape((height, width, 3))

    # 6. Create a Comparison Image and Display
    # To display side-by-side, we can concatenate the images horizontally
    comparison_image = np.hstack((original_image, segmented_image))

    # 6. Display the Results in Separate Windows
    # Display the original image in its own window
    cv2.imshow('Original Image', original_image)

    # Display the segmented image in a second, separate window
    cv2.imshow(f'Segmented Image (K={K})', segmented_image)

    # Wait for a key press and then close all windows
    print("\nDisplaying results in two windows. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

