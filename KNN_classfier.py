import numpy as np
from collections import Counter

class KNN_classfier:

    def __init__ (self, _k, _train, _test, _train_classes, _test_classes) -> None:
        self.k = _k
        self.train = _train
        self.test = _test
        self.train_classes = _train_classes
        self.test_classes = _test_classes

    def select_neighbors (self, neighbor_dict):
        selected = []
        keys = list(neighbor_dict.keys())
        
        for i in range (self.k):
            selected.append(keys[i])
        
        return selected


    def classifie (self, _target):
        distances = {}
        for i, player in enumerate(self.train): 
            distances[i] = self.euclidean_dist(_target, player)
        
        sorted_neighbors = sorted(distances.items(), key=lambda x:x[1])
        sorted_neighbors = dict(sorted_neighbors)

        neighbors = self.select_neighbors(sorted_neighbors)

        classes = []
        for neighbor in neighbors:
            classes.append(self.train_classes[neighbor][0])

        counter = Counter(classes)
        player_class = counter.most_common(n = 1)[0][0]

        return (player_class)
    
    def euclidean_dist (self, u, v):
        return np.sqrt(np.sum((u-v)**2))
    
    def test_fun (self):
        classes = {}
        for i, player in enumerate(self.test):
            classes[i] = self.classifie(player)
        
        TP, FP, TN, FN = self.confusion_matrix(classes)
        accuracy = self.accuracy (TP, FP, TN, FN)
        print(f"Accuracy is: {accuracy}")
        precision = self.precision (TP, FP)
        print(f"Precision is: {precision}")
        recall = self.recall (TP, FN)
        print(f"Recall is: {recall}")
        F1 = self.F1 (precision, recall)
        print(f"F1 is: {F1}")

    def confusion_matrix (self, classes):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range (len(list(classes.keys()))):
            if (classes[i] == 1 and self.test_classes[i] == 1):
                TP += 1
            elif (classes[i] == 1 and self.test_classes[i] == 0):
                FP += 1
            elif (classes[i] == 0 and self.test_classes[i] == 0):
                TN += 1
            elif (classes[i] == 0 and self.test_classes[i] == 1):
                FN += 1
        
        print(f"TP = {TP}")
        print(f"FP = {FP}")
        print(f"TN = {TN}")
        print(f"FN = {FN}")

        return (TP,FP,TN,FN)

    def accuracy (self, TP, FP, TN, FN):
        accuracy = (TP+TN) / (TP+FP+TN+FN)
        return accuracy
    
    def precision (self, TP, FP):
        precision = TP / (TP+FP)
        return precision
    
    def recall (self, TP, FN):
        recall = TP / (TP+FN)
        return recall
    
    def F1 (self, precision, recall):
        F1 = (2 * precision * recall) / (precision + recall)
        return F1