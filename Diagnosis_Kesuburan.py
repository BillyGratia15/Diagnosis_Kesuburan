import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('fertility.csv',
    names = ['Season','Age','CD', 'ACST','SI','HF','FAC','SH','NHS','Diagnosis'], 
    header = 0
)

# LABELLING
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['CD']=label.fit_transform(df['CD'])
# print(label.classes_)
df['ACST']=label.fit_transform(df['ACST'])
# print(label.classes_)
df['SI']=label.fit_transform(df['SI'])
# print(label.classes_)
df['HF']=label.fit_transform(df['HF'])
# print(label.classes_)
df['FAC']=label.fit_transform(df['FAC'])
# print(label.classes_)
df['SH']=label.fit_transform(df['SH'])
# print(label.classes_)

df = df.drop(['Season'],axis = 1)
# print(df)
x = df.drop(['Diagnosis'],axis = 1)
y = df['Diagnosis']

# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans=ColumnTransformer(
    [('one_hot_encoder',OneHotEncoder(categories='auto'),[4,5,6])],
    remainder='passthrough'
)
x = np.array(coltrans.fit_transform(x),dtype=np.float64)

from sklearn.model_selection import train_test_split
xtr,xtest,ytr,ytest = train_test_split(
    x,
    y,
    test_size = .1
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLog=LogisticRegression(solver='liblinear',multi_class='auto')
modelLog.fit(xtr,ytr)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
modelRandom=RandomForestClassifier()
modelRandom.fit(xtr,ytr)

# KNN
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=10)
modelKNN.fit(xtr,ytr)
# print(round(modelKNN.score(xtest,ytest)*100,2),'%')

# HF ['less than 3 months ago' 'more than 3 months ago' 'no']
# FAC['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']
# SH['daily' 'never' 'occasional']
# Age
# CD [0 1] = ['no' 'yes']
# ACST [0 1] = ['no' 'yes']
# SI [0 1] = ['no' 'yes']
# NHS
# Diagnosis: ['Altered' 'Normal']

A = [0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]
B = [0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]
C = [1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]
D = [0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]
E = [0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]

print('Arin, prediksi kesuburan:', (modelLog.predict([A])),'(Logistic Regression)')
print('Arin, prediksi kesuburan:', (modelKNN.predict([A])),'(K-Nearest Neighbors)')
print('Arin, prediksi kesuburan:', (modelRandom.predict([A])),'(Random Forest Classifier)')
print(' ')
print('Bebi, prediksi kesuburan:', (modelLog.predict([B])),'(Logistic Regression)')
print('Bebi, prediksi kesuburan:', (modelKNN.predict([B])),'(K-Nearest Neighbors)')
print('Bebi, prediksi kesuburan:', (modelRandom.predict([B])),'(Random Forest Classifier)')
print(' ')
print('Caca, prediksi kesuburan:', (modelLog.predict([C])),'(Logistic Regression)')
print('Caca, prediksi kesuburan:', (modelKNN.predict([C])),'(K-Nearest Neighbors)')
print('Caca, prediksi kesuburan:', (modelRandom.predict([C])),'(Random Forest Classifier)')
print(' ')
print('Dini, prediksi kesuburan:', (modelLog.predict([D])),'(Logistic Regression)')
print('Dini, prediksi kesuburan:', (modelKNN.predict([D])),'(K-Nearest Neighbors)')
print('Dini, prediksi kesuburan:', (modelRandom.predict([D])),'(Random Forest Classifier)')
print(' ')
print('Enno, prediksi kesuburan:', (modelLog.predict([E])),'(Logistic Regression)')
print('Enno, prediksi kesuburan:', (modelKNN.predict([E])),'(K-Nearest Neighbors)')
print('Enno, prediksi kesuburan:', (modelRandom.predict([E])),'(Random Forest Classifier)')
