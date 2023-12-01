import sys
import pandas as pd
import numpy as np
import KNN_classfier
import K_means_clustering
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

train = train.drop('TARGET_5Yrs', axis = 1)
train = train.to_numpy()

test_classes = pd.DataFrame()
test_classes['TARGET_5Yrs'] = test['TARGET_5Yrs'].copy(deep = True)
test_classes = test_classes.to_numpy()

test = test.drop('TARGET_5Yrs', axis = 1)
test = test.to_numpy()

if algorithm == 'KNN':
    classfier = KNN_classfier.KNN_classfier(K, train, test, train_classes, test_classes)
    classfier.test_fun()
elif algorithm == 'K-means':
    clusters = K_means_clustering.K_means_cluster(K, train, train_classes, 100)
    clusters.k_means()
    cluster_test_dict = {}

    for i, point in enumerate(test):
        cluster_test_dict[i] = clusters.point_to_cluster(point)
    
    print(cluster_test_dict)