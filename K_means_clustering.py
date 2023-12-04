import numpy as np

'''
This class implements the Centroids and stores and manipulate its data
'''
class Centroid:

    def __init__(self, _value) -> None:
        self.points = []
        self.value = _value
        self.points_idx = []
    
    '''
    Add a point to the centroid's cluster
    '''
    def add_point (self, new_point, idx):
        self.points.append(new_point)
        self.points_idx.append(idx)
    
    '''
    Clear the entire cluster
    '''
    def clear_cluster (self):
        self.points.clear()
        self.points_idx.clear()
    
    '''
    Update the value in the centroid as a mean of all the points on the cluster
    '''
    def update_value (self):
        self.value = np.mean(self.points, axis = 0)

'''
This class implements the K-means algorithm
'''
class K_means_cluster:

    def __init__(self, _k, _initial_data, _labels, _max_iterations, _random_seed = 42) -> None:
        self.k = _k
        self.initial_data = _initial_data
        self.labels = _labels
        self.max_iterations = _max_iterations
        self.centroids = []
        self.total_stayed = 0
        self.total_quited = 0
        np.random.seed(_random_seed)
    
    '''
    Given a set of points and a integer K, put the points into K clusters, according to proximity
    '''
    def k_means (self):
        self.init_centroids()

        for i in range (self.max_iterations):
            self.points_to_cluster ()

            self.update_centroids ()
            if (i < self.max_iterations - 1):
                for centroid in self.centroids:
                    centroid.clear_cluster()
    
    '''
    Return the euclidean distance between two points
    '''
    def euclidean_dist (self, u, v):
        return np.sqrt(np.sum((u-v)**2))
    
    '''
    Initialize k centroids choosing a random point in the set
    '''
    def init_centroids (self):
        sample = self.k_point_sample ()

        for point in sample:
            self.centroids.append(Centroid(point))
    
    '''
    Sample k points from the set
    '''
    def k_point_sample (self):
        sample = self.initial_data[np.random.choice(self.initial_data.shape[0], size = self.k, replace = False)]
        return sample

    '''
    Link all the points to a cluster using the point_to_cluster function as aux
    '''    
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

    '''
    Update the values of centroids
    '''
    def update_centroids (self):
        for centroid in self.centroids:
            centroid.update_value()
    
    '''
    Count all the players that stayed or quited the league
    '''
    def count_all_stayed_quit (self):
        for centroid in self.centroids:
            for idx in centroid.points_idx:
                if self.labels[idx] == 1:
                    self.total_stayed += 1
                else:
                    self.total_quited += 1

    '''
    Count the proportion of each class on each kernel.
    '''
    def count_proportions (self):
        proportions = {}

        for i, centroid in enumerate(self.centroids):
            stayed = 0
            quited = 0

            for idx in centroid.points_idx:
                if self.labels[idx] == 1:
                    stayed += 1
                else:
                    quited += 1
            
            proportions [i] = ((stayed / (stayed + quited), quited / (stayed + quited)), (stayed / self.total_stayed, quited / self.total_quited))
        
        return proportions