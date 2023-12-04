import sys
import pandas as pd
import numpy as np
import KNN_classfier
import K_means_clustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#ao executar colocar na linha de comando da seguinte maneira 'python3 main.py algoritmo  K treino teste'  

algorithm = sys.argv[1]
K = int(sys.argv[2])
train = sys.argv[3]
test = sys.argv[4]


train = pd.read_csv(train)
test = pd.read_csv(test)
train_classes = pd.DataFrame()
train_classes['TARGET_5Yrs'] = train['TARGET_5Yrs'].copy(deep = True)
train_classes = train_classes.to_numpy()
complete_train = train.copy(deep = True)
train = train.drop('TARGET_5Yrs', axis = 1)
train = train.to_numpy()

test_classes = pd.DataFrame()
test_classes['TARGET_5Yrs'] = test['TARGET_5Yrs'].copy(deep = True)
test_classes = test_classes.to_numpy()

test = test.drop('TARGET_5Yrs', axis = 1)
test = test.to_numpy()

if algorithm == 'KNN':
    print("Author implementation:")
    classfier = KNN_classfier.KNN_classfier(K, train, test, train_classes, test_classes)
    classfier.test_fun()

    #Extra points comparation with sklearn implementation
    print("sklearn implementation:")
    train_classes = train_classes.ravel()
    test_classes = test_classes.ravel()
    model = KNeighborsClassifier(K)
    model.fit(train, train_classes)
    y_pred = model.predict(test)
    conf = metrics.confusion_matrix(test_classes, y_pred)
    print(conf)
    accuracy = metrics.accuracy_score(test_classes, y_pred)
    print(f"Accuracy is: {accuracy}")
    precision = metrics.precision_score(test_classes, y_pred)
    print(f"Precision is: {precision}")
    recall = metrics.recall_score(test_classes, y_pred)
    print(f"Recall is: {recall}")
    F1 = metrics.f1_score(test_classes, y_pred)
    print(f"F1 is: {F1}")

elif algorithm == 'K-means':
    clusters = K_means_clustering.K_means_cluster(K, train, train_classes, 500)
    clusters.k_means()
    cluster_test_dict = {}

    for i, point in enumerate(test):
        cluster_test_dict[i] = clusters.point_to_cluster(point)
    
    clusters.count_all_stayed_quit()
    proportions = clusters.count_proportions()

    
    print("Centroids are:")
    for centroid in clusters.centroids:
        print(centroid.value)
    for i in list(proportions.keys()):
        print(f"In centroid {i}, we had {proportions[i][0][0]*100 :.2f}% of players that stayed in the league, and {proportions[i][0][1]*100 :.2f}% of players that quited.")
        print(f"This represents {proportions[i][1][0]*100 :.2f}% of the total players that stayed in the league, and {proportions[i][1][1]*100 :.2f}% of players that quited.")   

    