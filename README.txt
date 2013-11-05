README

1. 201001043_part1.cpp contains the code to train and test the ms2cd dataset. 
			
	 Compile: g++ 201001043_part1.cpp
	 Run: ./a.out <algorithm number>
	 Output: Accuracy on test data

************************************************************************************************************************************************************************

2. 201001043_part2.cpp contains the code to train and test the mnist dataset
	
	 Compile: g++ 201001043_part2.cpp -O3
	 Run: ./a.out <algorithm number>
	 Output: Accuracy on test data

************************************************************************************************************************************************************************

3. 201001043_classify.cpp contains the code to read a test case file with 5 test cases in the format of MNIST data (with space separated 784 values per test case. See file 'image_vector' for exact format) and output the result using the pre-computed classifier vectors.

	 Compile: g++ 201001043_classify.cpp
	 Run: ./a.out <algorithm number> filename
	 Output: Class of test cases

************************************************************************************************************************************************************************

<algorithm number> are: 
1. Single Sample perceptron
2. Batch Perceptron
3. Single Sample Perceptron With Margin
5. Batch Perceptron With Margin
6. Batch Relaxation With Margin

***********************************************************************************************************************************************************************
