import numpy as np
from collections import Counter

class KNN_classfier:

    def __init__ (self, _k, _train, _train_classes) -> None:
        self.k = _k
        self.train = _train
        self.train_classes = _train_classes

    def select_neighbors (self, neighbor_dict):
        selected = []
        keys = neighbor_dict.keys()
        
        for i in range (self.k):
            selected.append(keys[i])
        
        return selected


    def classifie (self, _target):
        distances = {}
        for i, player in enumerate(self.train):
            distances[i] = np.distance.euclidean(_target, player)
        
        sorted_neighbors = sorted(distances.items(), key=lambda x:x[1])
        sorted_neighbors = dict(sorted_neighbors)

        neighbors = self.select_neighbors(sorted_neighbors)

        classes = []
        for neighbor in neighbors:
            classes.append(self.train_classes[neighbor])

        counter = Counter(classes)
        player_class = counter.most_common(n = 1)[0][0]

        return (player_class)
    
    
