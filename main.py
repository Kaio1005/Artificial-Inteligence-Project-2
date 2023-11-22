import sys
import pandas as pd
import numpy as np
import KNN_classfier
#ao executar colocar na linha de comando da seguinte maneira 'python3 main.py algoritmo treino teste' 
#Se o algoritmo for KNN colocar também o valor de K após o teste 'python3 main.py algoritmo treino teste K' 

algorithm = sys.argv[1]
train = sys.argv[2]
test = sys.argv[3]

train = pd.read_csv(train)
test = pd.read_csv(test)


if algorithm == 'KNN':
    K = int(sys.argv[4])
    classfier = KNN_classfier(K, train, test)