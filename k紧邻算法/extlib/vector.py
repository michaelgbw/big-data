# -*- coding: utf-8 -*- 
import math

def vectorAdd(v,w):
	return [vi + wi 
	for vi,wi in zip(v,w)]

def vector_subtract(v,w):
	return [vi - wi 
	for vi,wi in zip(v,w)]
def vector_sum(vectors):
	return reduce(vectorAdd, vectors)
#向量点乘
def dot(w,v):
	return sum(vi * wi for vi,wi in zip(w,v))

#向量平方和：
def sum_of_squares(v):
	return dot(v,v)
def squared_distance(v, w):
	return sum_of_squares(vector_subtract(v, w))
#向量的距离
def distance(v, w):
	return math.sqrt(squared_distance(v, w))
#一个向量乘以一个标量
def scalar_multiply(c, v):
	return [c * v_i for v_i in v]
#计算向量的均值
def vector_mean(vectors):
	n = len(vectors)
	return scalar_multiply(1/n,vector_sum(vectors))

########test##########
print " import vector success"