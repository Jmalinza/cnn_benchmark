#include "matrix.h"
#include "layers.h"
#include <cstddef>
#include <random>
#include <iostream>
#include <valarray>
#include <vector>

using namespace std;

// testing
void test_indexing();
void test_ops();
void test_windows();
void cnr1();
void cnr2();
void cnr3();
// utility functions
template <typename T> 
Tensor<T> fill_tensor(size_t& batches, size_t& rows, size_t& cols, size_t& channels);
template <typename T>
void print_valarray(const char* label, const valarray<T>& a);

 
int main(int argc, char** argv) {

	int cnr_number = 0;

   	if(argc > 1){
   		cnr_number = atoi(argv[1]);
   	}

   	if(cnr_number == 1){
   		cnr1();
   	} else if(cnr_number == 2){
   		cnr2();
   	} else if( cnr_number == 3){
   		cnr3();
   	} else{
   		cnr1();
   		cnr2();
   		cnr3();
   	}

   	return 0;
} 

void cnr1(){
	float data_range_low  = -10;
	float data_range_high = 10;
	random_device rd;
	mt19937 e(rd());
	uniform_real_distribution<> dist(data_range_low, data_range_high);
	// input tensor size
	size_t batch_size = 1; 
	size_t im_channels_1 = 3;
	size_t rows_1 = 224;
	size_t cols_1 = 224;
	// convolution filter size
	size_t f_num_1 = 64;
	size_t f_rows_1  = 7;
	size_t f_cols_1  = 7;
	size_t bias = 0;
	size_t stride = 2;
	// batch norm parameters
	float gamma = 1;
	float beta = 1; 
	float eps = 1e-10;

	// // CNR 1 ~ conv1
	Tensor<float> input1   = fill_tensor<float>(batch_size, rows_1, cols_1, im_channels_1);
	Tensor<float> filter1 = fill_tensor<float>(f_num_1, f_rows_1, f_cols_1, im_channels_1);
	std::cout << "cnr1 starting (1, 224, 224, 3) -> (1, 112, 112, 64)..." << std::endl;
	Tensor<float> c1 = convolution_layer(input1, filter1, bias, stride);
	std::cout << "convolution1: ";
	c1.print_shape();
	Tensor<float> n1 = batchnorm_layer(c1, gamma, beta, eps);
	std::cout << "batch norm1: ";
	n1.print_shape();
	Tensor<float> r1 = relu_layer(n1);
	std::cout << "relu1: ";
	r1.print_shape();
}

void cnr2(){
	float data_range_low  = -10;
	float data_range_high = 10;
	random_device rd;
	mt19937 e(rd());
	uniform_real_distribution<> dist(data_range_low, data_range_high);
	// input tensor size
	size_t batch_size = 1; 
	size_t im_channels_2 = 128;
	size_t rows_2 = 28;
	size_t cols_2 = 28;
	// convolution filter size
	size_t f_num_2_1  = 128;
	size_t f_rows_2_1 = 3;
	size_t f_cols_2_1  = 3;
	size_t stride_2_1  = 1;
	size_t bias = 0;
	// batch norm parameters
	float gamma = 1;
	float beta = 1; 
	float eps = 1e-10;
	// CNR 2 ~ conv3_x 3x3, 128->128 into shortcut no extra conv
	Tensor<float> shortcut_2  = fill_tensor<float>(batch_size, rows_2, cols_2, im_channels_2);
	Tensor<float> input2   = fill_tensor<float>(batch_size, rows_2, cols_2, im_channels_2);
	Tensor<float> filter2_1 = fill_tensor<float>(f_num_2_1, f_rows_2_1, f_cols_2_1, im_channels_2);
	std::cout << "cnr2 starting (1, 28, 28, 128) -> (1, 28, 28, 512) + shortcut..." << std::endl;
	Tensor<float> c2 = convolution_layer(input2, filter2_1, bias, stride_2_1);
	std::cout << "convolution1: ";
	c2.print_shape();
	Tensor<float> n2 = batchnorm_layer(c2, gamma, beta, eps);
	std::cout << "batch norm1: ";
	n2.print_shape();

	std::cout << "shortcut2: ";
	shortcut_2.print_shape();
	Tensor<float> shortcut_out_2 = add_layer(shortcut_2, n2);
	std::cout << "shortcut_out2: ";
	shortcut_out_2.print_shape();

	Tensor<float> r2 = relu_layer(shortcut_out_2);
	std::cout << "relu1: ";
	r2.print_shape();
}
void cnr3(){
	float data_range_low  = -10;
	float data_range_high = 10;
	random_device rd;
	mt19937 e(rd());
	uniform_real_distribution<> dist(data_range_low, data_range_high);
	// input tensor size
	size_t batch_size = 1; 
	size_t im_channels_3 = 256;
	size_t rows_3 = 14;
	size_t cols_3 = 14;
	// convolution filter size
	size_t f_num_3  = 512;
	size_t f_rows_3 = 3;
	size_t f_cols_3  = 3;
	size_t stride_3  = 2;
	size_t bias = 0;
	// batch norm parameters
	float gamma = 1;
	float beta = 1; 
	float eps = 1e-10;
	// CNR 3 ~ conv5_x 3x3, 256->512 into shortcut with extra conv
	Tensor<float> shortcut_3  = fill_tensor<float>(batch_size, rows_3, cols_3, im_channels_3);
	Tensor<float> input3   = fill_tensor<float>(batch_size, rows_3, cols_3, im_channels_3);
	Tensor<float> filter3_1 = fill_tensor<float>(f_num_3, f_rows_3, f_cols_3, im_channels_3);
	Tensor<float> filter3_2 = fill_tensor<float>(f_num_3, f_rows_3, f_cols_3, im_channels_3);
	std::cout << "cnr3 starting (1, 256, 14, 14) -> (1, 512, 7, 7) + shortcut w/ extra conv..." << std::endl;
	Tensor<float> c3 = convolution_layer(input3, filter3_1, bias, stride_3);
	std::cout << "convolution3: ";
	c3.print_shape();
	Tensor<float> n3 = batchnorm_layer(c3, gamma, beta, eps);
	std::cout << "batch norm3: ";
	n3.print_shape();

	Tensor<float> s_c = convolution_layer(shortcut_3, filter3_2, bias, stride_3);
	std::cout << "s_c: ";
	s_c.print_shape();
	Tensor<float> s_n = batchnorm_layer(s_c, gamma, beta, eps);
	std::cout << "s_batch_norm: ";
	s_n.print_shape();


	Tensor<float> out = add_layer(s_n, n3);
	std::cout << "s_batch_norm + batch_norm3: ";
	out.print_shape();
	Tensor<float> r3 = relu_layer(out);
	std::cout << "relu3: ";
	r3.print_shape();
}

template <typename T> 
Tensor<T> fill_tensor(size_t& batches, size_t& rows, size_t& cols, size_t& channels){
	random_device rd;
	mt19937 e(rd());
	uniform_real_distribution<> dist(-1,1);
	Tensor<T> return_tensor(batches);
	int size = rows*cols;
	for(int n = 0; n< batches; ++n){
		Image<T> im(channels, rows, cols);
		for(int c = 0; c < channels; ++c){
			T data[size];
			for(int j = 0; j < size; ++j){
				data[j] = dist(e);
			}
			valarray<T> v(data, size);
			im.data[c].update(v);
		}
		return_tensor.push_back(im);
	}
	return return_tensor;
}

void test_windows(){
	int size = 16;
	int rows = 4;
	int cols = 4;
	float a[] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	valarray<float> v1(a, size);
	Matrix<float> m1(rows,cols,v1);
	m1.printMatrix("m1: ");

	Matrix<float> m2 = m1.window(1,1,3,3);
	m2.printMatrix("m2 = m1.window(1,1,3,3) ");

	Matrix<float> m3 = m1.window(1,2,3,3);
	m3.printMatrix("m3 = m1.window(1,2,3,3) ");

	Matrix<float> m4 = m1.window(2,1,3,3);
	m4.printMatrix("m4 = m1.window(2,1,3,3) ");

	Matrix<float> m5 = m1.window(2,2,3,3);
	m5.printMatrix("m5 = m1.window(2,2,3,3) ");

	Matrix<float> m6 = m1.window(0,0,3,3);
	m6.printMatrix("m6 = m1.window(0,0,3,3) ");

	Matrix<float> m7 = m1.window(1,0,3,3);
	m7.printMatrix("m7 = m1.window(1,0,3,3) ");

	Matrix<float> m8 = m1.window(1,-1,3,3);
	m8.printMatrix("m8 = m1.window(1,-1,3,3) ");
}

void test_ops(){
	int size = 12;
	int rows = 3;
	int cols = 4;
	float a[] = { 1, 1, 1 ,0 ,2, 2, 2, 0, 3, 3, 3 , 0};
	float b[] = { 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0};
	valarray<float> v1(a, size);
	valarray<float> v2(b, size);
	// valarray<float> c(cdata, n*n);
	Matrix<float> m1(rows,cols,v1);
	Matrix<float> m2(rows,cols,v2);
	m1.printMatrix("m1");
	m2.printMatrix("m2");
	Matrix<float> m3 = m2.transpose();
	m3.printMatrix("m3 = t(m2)");
	Matrix<float> m4 = m2.multiply(m3);
	m4.printMatrix("m4 = m2 * m3");

	Matrix<float> m5 = m2.add(m2);
	m5.printMatrix("m5 = m2 + m2");

	Matrix<float> m6 = m2.sumRows();
	m6.printMatrix("m6 =  sumRows(m2)");

	Matrix<float> m7 = m2.sumCols();
	m7.printMatrix("m7 =  sumCols(m2)");

	cout << "m2.sum() = " << m2.sum() << endl;
	cout << "m6.sum() = " << m6.sum() << endl;
	cout << "m7.sum() = " << m7.sum() << endl;

	float c[] = { 1,-1 ,2,-2 ,3, -3, 4, -4, 5, -5, 0 , 100};
	valarray<float> v3(c, size);
	Matrix<float> m8(rows,cols,v3);
	m8.printMatrix("m8");
	m8 = m8.relu();

	m8.printMatrix("m8 = relu (m8)");

	Matrix<float> m9 = m8.tanh();
	m9.printMatrix("m9 = tanh(m8)");

	Matrix<float> m10 = m9.sigmoid();
	m10.printMatrix("m10 = sigm(m9)");

	float d[] = { 1,2,3,4,5,6,7,8,9,10,11,12};
	float e[] = { 1,2,3,4,5,6,7,8,9,10,11,12};
	valarray<float> v4(d, size);
	valarray<float> v5(d, size);
	Matrix<float> m11(rows,cols,v4);
	Matrix<float> m12 = m11.add(1);
	m11.printMatrix("m11");
	m12.printMatrix("m12 = m11 + 1");
	Matrix<float> m13 = m11.multiplyElements(m12);
	m13.printMatrix("m13 = m11 elem * m12 elem");
}

void test_indexing(){
	int size = 12;
	int r = 4;
	int c = 3;
	float a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	valarray<float> v1(a,size);
	Matrix<float> m1(r,c,v1);
	cout << "number of rows: " << m1.rows() << '\n'
  	<< "number of cols: " << m1.cols() << endl;
  	m1.printMatrix("m1");

  	valarray<float> r1 = m1.row(0);
  	valarray<float> r2 = m1.row(1);
  	valarray<float> r3 = m1.row(2);
  	valarray<float> r4 = m1.row(3);  

  	print_valarray("r1", r1);
  	print_valarray("r2", r2);
  	print_valarray("r3", r3);
  	print_valarray("r4", r4);

  	valarray<float> c1 = m1.col(0);
  	valarray<float> c2 = m1.col(1);
  	valarray<float> c3 = m1.col(2);

  	print_valarray("c1", c1);
  	print_valarray("c2", c2);
  	print_valarray("c3", c3);

  	cout << "m1(0,0) : " << m1(0,0)<< endl;
  	cout << "m1(0,1) : " << m1(0,1)<< endl;
  	cout << "m1(0,2) : " << m1(0,2)<< endl;
  	cout << "m1(1,0) : " << m1(1,0)<< endl;
  	cout << "m1(1,1) : " << m1(1,1)<< endl;
  	cout << "m1(1,2) : " << m1(1,2)<< endl;
  	cout << "m1(2,0) : " << m1(2,0)<< endl;
  	cout << "m1(2,1) : " << m1(2,1)<< endl;
  	cout << "m1(2,2) : " << m1(2,2)<< endl;
  	cout << "m1(3,0) : " << m1(3,0)<< endl;
  	cout << "m1(3,1) : " << m1(3,1)<< endl;
  	cout << "m1(3,2) : " << m1(3,2)<< endl;


  	m1.printMatrix("m1");

  	float one = m1(0,0);
  	float two = m1(0,1);
  	float three = m1(0,2);

  	cout << "m1(0,0) = m1(0,1) + m1(0,2) " << endl;
  	m1(0,0) = m1(0,1) + m1(0,2);
  	m1.printMatrix("new m1");

}

template <typename T>
void print_valarray(const char* label, const valarray<T>& a) {
	cout << label << ": ";
	for(size_t i = 0; i < a.size(); ++i){
		cout << a[i] << ' ';
	}
	cout << endl;
}

