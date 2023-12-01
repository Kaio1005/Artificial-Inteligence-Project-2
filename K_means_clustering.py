import numpy as np

class Centroid:

    def __init__(self, _value) -> None:
        self.points = []
        self.value = _value
        self.points_idx = []
    
    def add_point (self, new_point, idx):
        self.points.append(new_point)
        self.points_idx.append(idx)
    
    def clear_cluster (self):
        self.points.clear()
        self.points_idx.clear()
    
    def update_value (self):
        self.value = np.mean(self.points, axis = 0)

class K_means_cluster:

    def __init__(self, _k, _initial_data, _labels, _max_iterations, _random_seed = 42) -> None:
        self.k = _k
        self.initial_data = _initial_data
        self.labels = _labels
        self.max_iterations = _max_iterations
        self.centroids = []
        np.random.seed(_random_seed)
    
    def k_means (self):
        self.init_centroids()

        for i in range (self.max_iterations):
            self.points_to_cluster ()

            self.update_centroids ()
            if (i < self.max_iterations - 1):
                for centroid in self.centroids:
                    centroid.clear_cluster()
    
    def euclidean_dist (self, u, v):
        return np.sqrt(np.sum((u-v)**2))
    
    def init_centroids (self):
        sample = self.k_point_sample ()

        for point in sample:
            self.centroids.append(Centroid(point))
    
    def k_point_sample (self):
        sample = self.initial_data[np.random.choice(self.initial_data.shape[0], size = self.k, replace = False)]
        return sample
    
    def points_to_cluster (self):
        for i, point in enumerate(self.initial_data):
            centroid_idx = self.point_to_cluster(point)
            self.centroids[centroid_idx].add_point(point, i)
    
    def point_to_cluster (self, point):
        dist_min = []
        for i, centroid in enumerate(self.centroids):
            if len(dist_min) == 0:
                dist_min.append((i, self.euclidean_dist(point, centroid.value)))
            else:
                current_dist = self.euclidean_dist(point, centroid.value)

                if (current_dist < dist_min[0][1]):
                    dist_min[0] = (i, current_dist)
            
        return (dist_min[0][0])

    def update_centroids (self):

        for centroid in self.centroids:
            centroid.update_value()