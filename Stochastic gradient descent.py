# encoding=utf-8
#http://blog.csdn.net/zbc1090549839/article/details/38149561
#随机梯度算法
# matrix_A  训练集两个参数x,y
matrix_A = [[1,4], [2,5], [5,1], [4,2]]
Matrix_y = [19,26,19,20]
dataNum = len(Matrix_y)
#对系数的初步猜测
theta = [1,3]
#学习速率
leraing_rate = 0.005
loss = 50
iters = 1
Eps = 0.0001
while loss>Eps and iters <1000 :
	loss = 0
	for i in range(dataNum-1) :
		h = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1]
		theta[0] = theta[0] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][0]
		theta[1] = theta[1] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][1]
		#y = theta1*x1 +theta2*x2
		#验证原本的公式的预测值（最小二乘法）
		for i in range(dataNum-1) :
			Error = 0
			Error = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1] - Matrix_y[i]
			Error = 0.5 * Error*Error
			loss = loss +Error
	iters = iters +1
print 'theta=',theta
print 'iters=',iters