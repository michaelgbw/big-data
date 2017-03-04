# encoding=utf-8
#训练数据的切分
def split_data(data,pro):
	result = [],[]
	for row in data:
		result[0 if random.random() <pro else 1].append(row)
	return result
def train_test_split(x, y, test_pct):
	data = zip(x, y) # 成对的对应值
	train, test = split_data(data, 1 - test_pct) # 划分这个成对的数据集
	x_train, y_train = zip(*train) # 魔法般的解压技巧
	x_test, y_test = zip(*test)
	return x_train, x_test, y_train, y_test
