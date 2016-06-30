import numpy as np
from oversampler import Oversampler
from enums import Oversampling

data = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10], [11,11,11], [12,12,12], [13,13,13], [14,14,14], [15,15,15], [16,16,16], [17,17,17], [18,18,18], [19,19,19]])
target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

def oversamplingRepeater():

	over = Oversampler(Oversampling.Repeat,data, target,True)
	over_data, over_target = over.balance()

	print(over_data)
	print(over_target)

	assert len(over_target) == len(over_data) == 30
	assert len(over_data[0 == over_target]) == 15
	assert len(over_data[1 == over_target]) == 15

def oversamplingSmoteRegular():

	over = Oversampler(Oversampling.SmoteRegular,data, target,True)
	over_data, over_target = over.balance()

	print(len(over_data))
	print(len(over_target))
	# assert len(over_target) == len(over_data) == 30
	# assert len(over_data[0 == over_target]) == 15
	# assert len(over_data[1 == over_target]) == 15

def main():
	
	oversamplingRepeater()
	oversamplingSmoteRegular()


if __name__ == '__main__':
	main()


