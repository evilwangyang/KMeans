#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/10/11 21:17
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  KMeans
# @File      :  KMeans_test1.py
# @Software  :  PyCharm Community Edition

# ********************************************************* 
import numpy as np

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float,curLine))
		dataMat.append(fltLine)
	return dataMat

def distEclud(vecA,vecB):
	return np.sqrt(sum(np.power(vecA-vecB,2)))

def randCent(dataSet,k):
	n = np.shape(dataSet)[1]
	centroids = np.mat(np.zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
	return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	m = np.shape(dataSet)[0]
	clusterAssment = np.mat(np.zeros((m,2)))
	centroids = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:].A[0],dataSet[i,:].A[0])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print(centroids)
		for cent in range(k):
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
			centroids[cent,:] = np.mean(ptsInClust,axis=0)
	return centroids,clusterAssment

if __name__ == '__main__':
	datMat = np.mat(loadDataSet('testSet.txt'))

	# print(min(datMat[:,0]))
	# print(min(datMat[:,1]))
	# print(max(datMat[:,0]))
	# print(max(datMat[:,1]))
	#
	# print(randCent(datMat,2))
	# print(distEclud(datMat[0].A[0],datMat[1].A[0]))

	myCentroids,clustAssing = kMeans(datMat,4)
