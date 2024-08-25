import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances

# Define the number of roads and vehicles on each road
m = 10
n = 50

# Generate random data for vehicles on each road, however it can be taken as input as well
data = np.zeros((m * n, 2))
for i in range(m):
    for j in range(n):
        size = np.random.choice([2, 4, 8])
        speed = np.random.uniform(0, 60)
        data[i * n + j, :] = [size, speed]

# Define the FCM function
def FCM(data, n_clusters=3, max_iter=100, m=2, error=1e-5):
    # Randomly initialize the membership matrix
    membership_mat = np.random.rand(len(data), n_clusters)
    
    # Ensuring the values to be between 0 to 1, we do the following division
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])
    
    # Repeat until convergence or max iterations is reached
    iteration = 0
    centroids_final = np.zeros((n_clusters, 2))
    
    while iteration < max_iter:
        # Compute cluster centers at each iteration
        centroids = np.dot(data.T, membership_mat) / np.sum(membership_mat, axis=0)
        
        # Compute distances from data points to cluster centers
        distances = euclidean_distances(data, centroids)
        
        # Update membership matrix
        membership_mat_new = np.power(distances, -2 / (m - 1))
        membership_mat_new = np.divide(membership_mat_new, np.sum(membership_mat_new, axis=1)[:, np.newaxis])
        
        # Check for convergence
        if np.max(np.abs(membership_mat_new - membership_mat)) < error:
            break
        
        membership_mat = membership_mat_new
        centroids_final = centroids
        iteration += 1
    
    # Assign each road to a cluster based on membership values
    clusters = pairwise_distances_argmin(data, centroids)
    membership_mat = membership_mat[np.arange(len(data)), clusters]
    
    return clusters, membership_mat, centroids_final

# Compute the sum of vehicle sizes and average speed for each road
road_data = np.zeros((m, 2))  # road_data will be used as the numpy array on which the FCM task will be performed
for i in range(m):
    # Calculating the size of traffic for each road
    road_data[i, 0] = np.sum(data[i * n:(i + 1) * n, 0])
    
    # Calculating the average speed of the traffic on each road
    road_data[i, 1] = np.mean(data[i * n:(i + 1) * n, 1])

# Apply FCM to cluster the roads based on congestion level
clusters, membership_mat, centroids_final = FCM(road_data)

# Print the centroids of each road where the axis = 0 represents the size of the traffic
# and the axis = 1 represents the average speed on that road (here, cluster center)
for i in centroids_final:
    print(i)

# Print the membership matrix for each road
for i in range(m):
    print("Road {}: Membership: {}".format(i + 1, membership_mat[i]))
