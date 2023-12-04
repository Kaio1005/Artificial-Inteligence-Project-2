import numpy as np
from collections import Counter

'''
This class implements the KNN algorithm, and is responsable to retain its data and functions to perform proper work
'''
class KNN_classfier:

    def __init__ (self, _k, _train, _test, _train_classes, _test_classes) -> None:
        self.k = _k
        self.train = _train
        self.test = _test
        self.train_classes = _train_classes
        self.test_classes = _test_classes

    '''
    Given a neighbor:distance dictionary return the K closest neighbors to the point
    '''
    def select_neighbors (self, neighbor_dict):
        selected = []
        keys = list(neighbor_dict.keys())
        
        for i in range (self.k):
            selected.append(keys[i])
        
        return selected


    '''
    Given a list of points return a list containing each point predicted class
    '''
    def classifie (self, _target):
        distances = {}
        for i, player in enumerate(self.train): 
            distances[i] = self.euclidean_dist(_target, player)
        
        #sorts the dictionary by distance
        sorted_neighbors = sorted(distances.items(), key=lambda x:x[1])
        sorted_neighbors = dict(sorted_neighbors)

        neighbors = self.select_neighbors(sorted_neighbors)

        classes = []
        for neighbor in neighbors:
            classes.append(self.train_classes[neighbor][0])

        counter = Counter(classes)
        player_class = counter.most_common(n = 1)[0][0]

        return (player_class)
    
    '''
    Given two points returns the euclidean distance between them
    '''
    def euclidean_dist (self, u, v):
        return np.sqrt(np.sum((u-v)**2))
    
    '''
    Function to classifie a test_set and return its metrics
    '''
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

    '''
    Return the values of TP, FP, TN, FN in these order
    '''
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

    '''
    Calculate and return accuracy
    '''
    def accuracy (self, TP, FP, TN, FN):
        accuracy = (TP+TN) / (TP+FP+TN+FN)
        return accuracy
    
    '''
    Calculate and return precision
    '''
    def precision (self, TP, FP):
        precision = TP / (TP+FP)
        return precision
    
    '''
    Calculate and return recall
    '''
    def recall (self, TP, FN):
        recall = TP / (TP+FN)
        return recall
    
    '''
    Calculate and return F1-score
    '''
    def F1 (self, precision, recall):
        F1 = (2 * precision * recall) / (precision + recall)
        return F1