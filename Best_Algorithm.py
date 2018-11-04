# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:32:23 2018

@author: Anand
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = datasets.load_iris()

targets = data.target

target_names = data.target_names

features = data.data

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)

#Linear Regresion
linear_classifier = LinearRegression()
linear_classifier.fit(x_train, y_train)

linear_prediction = linear_classifier.predict(x_test)

linear_score = linear_classifier.score(x_test, linear_prediction)
            
print("Linear Regression Score: ", linear_score)


#KMeans Clustering
kmeans_classifier = KMeans(n_clusters=3)
kmeans_classifier.fit(x_train, y_train)

kmeans_prediction = kmeans_classifier.predict(x_test)

kmeans_score = 0
kmeans_cm = confusion_matrix(kmeans_prediction, y_test)
for i in range(len(kmeans_cm)):
    for j in range(len(kmeans_cm[i])):
        if i == j:
            kmeans_score += kmeans_cm[i][j]
            
print("KMeans Score: ", kmeans_score / len(kmeans_prediction))


#KNearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=4)
knn_classifier.fit(x_train, y_train)

knn_prediction = knn_classifier.predict(x_test)

knn_score = 0
knn_cm = confusion_matrix(knn_prediction, y_test)
for i in range(len(knn_cm)):
    for j in range(len(knn_cm[i])):
        if i == j:
            knn_score += knn_cm[i][j]
            
print("KNN Score: ", knn_score / len(knn_prediction))


#Support Vector Machine
svm_classifier = svm.SVR(kernel='rbf')
svm_classifier.fit(x_train, y_train)

svm_prediction = svm_classifier.predict(x_test)

svm_score = svm_classifier.score(x_test, svm_prediction)
            
print("SVM Score: ", svm_score)


#Naive Bayes Theorem
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

nb_prediction = nb_classifier.predict(x_test)

nb_score = 0
nb_cm = confusion_matrix(nb_prediction, y_test)
for i in range(len(nb_cm)):
    for j in range(len(nb_cm[i])):
        if i == j:
            nb_score += nb_cm[i][j]
            
print("Naive Bayes Score: ", nb_score / len(nb_prediction))
