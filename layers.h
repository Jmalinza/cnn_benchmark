#pragma once
#include "matrix.h"
#include <vector>
#include <cmath>
#include <iostream>

template <typename T>
struct Image{
	std::vector<Matrix<T>> data;
	std::size_t im_chan;
	std::size_t im_rows;
	std::size_t im_cols;

	Image(int im_channels, int rows, int cols){
		data.reserve(im_channels);
		im_chan = im_channels;
		im_cols = cols;
		im_rows = rows;
		for(int i = 0; i < im_channels; ++i){
			Matrix<T>  m(rows, cols);
			data.push_back(m);
		}
	}

	void push_back(const Matrix<T> &image){
		data.push_back(image);
	}

	void printImage(){
		int index = 0;
		for(Matrix<T> this_channel: data){
			std::cout << "  image #: " << index << std::endl;
			this_channel.printMatrix();
			++index;
		}
	}

	std::size_t im_channels() const {return im_chan;}
	std::size_t rows() const {return im_rows;}
	std::size_t cols() const {return im_cols;}

	std::size_t size() const{ return data.size();}
};

template <typename T>
struct Tensor{
	std::vector<Image<T>> data;

	Tensor(int batch_size){
		data.reserve(batch_size);
	}

	void push_back(Image<T> &image){
		data.push_back(image);
	}

	void printTensor(const char* label){
		std::cout << label << ": ";
		print_shape();
		int batch_index = 0;
		std::cout << " batch size: " << data.size() << std::endl;
		for(Image<T> this_image: data){
			std::cout << " Batch #: " << batch_index << std::endl;
			std::cout << " image channels: " << this_image.data.size() << std::endl;
			this_image.printImage();
			++batch_index;
		}
	}

	std::size_t rows() const {return data[0].rows();}
	std::size_t cols() const {return data[0].cols();}
	std::size_t channels() const{ return data[0].im_channels();}

	std::size_t size() const{ return data.capacity();}

	void print_shape(){
		std::cout << "(" << size() << ", " << rows() << ", " << cols() << ", " << channels() << ")" << std::endl;
	}
};

template <typename T>
Tensor<T> add_layer(const Tensor<T>& a, const Tensor<T>& b){
	int batch_size = a.data.size();
	int i_index = 0;
	Tensor<T> return_tensor(batch_size);
	for(Image<T> this_image_a: a.data){
		Image<T> this_image_b = b.data[i_index];
		Image<T> output_image(this_image_a.im_channels(), this_image_a.rows(), this_image_a.cols());
		int c_index = 0;
		for(Matrix<T> this_channel_a: this_image_a.data){
			Matrix<T> this_channel_b = this_image_b.data[c_index];
			Matrix<T> result = this_channel_a.add(this_channel_b);
			output_image.data[c_index].update(result.array());
			++c_index;
		}
		return_tensor.push_back(output_image);
		++i_index;
	}
	return return_tensor;
}

template <typename T>
Tensor<T> convolution_layer(const Tensor<T> &input, 
							 const Tensor<T> &filter, float bias, std::size_t stride){
	int batch_size = input.data.size();

	std::size_t im_rows = input.rows();
	std::size_t im_cols = input.cols();
	std::size_t f_rows  = filter.rows();
	std::size_t f_cols  = filter.cols();
	std::size_t f_num   = filter.data.size();

	// std::cout << "batch size = " << batch_size << std::endl;
	// std::cout << " im_rows = " << im_rows ;
	// std::cout << " im_cols = " << im_cols << std::endl;
	// std::cout << " f_rows = " << f_rows ;
	// std::cout << " f_cols = " << f_cols ;
	// std::cout << " f_num = " << f_num << std::endl;
	// we assume equal padding
	std::size_t padding = f_rows/2;
	std::size_t o_rows = (im_rows - f_rows + 2*padding)/stride + 1;
	std::size_t o_cols = (im_cols - f_cols + 2*padding)/stride + 1;
	assert(o_cols == o_rows);
	// std::size_t o_size = o_rows*o_cols*f_num*batch_size;

	// std::cout << " padding = " << padding << std::endl;
	// std::cout << " o_rows = " << o_rows;
	// std::cout << " o_cols = " << o_cols << std::endl;

	Tensor<T> return_tensor(batch_size); // we need f_num output matrixs
	// std::cout << " return_tensor.size() = " << return_tensor.size() << std::endl;
	for(Image<T> this_image: input.data){ // for each image in the batch
		Image<T> output_image(f_num, o_rows, o_cols);
		// std::cout << " output_image.size() = " << output_image.size() << std::endl;
		int f_index = 0;
		for(Image<T> this_filter: filter.data){ // for each filter
			// std::cout << "filter - " << f_index <<std::endl;
			// std::cout << "this_filter : " <<std::endl;
			// this_filter.printImage();
			int c_index = 0;
			for(Matrix<T> this_channel: this_image.data){ // for each channel in the image
				Matrix<T> this_filter_channel = this_filter.data[c_index]; // get the correct filter channel
				// std::cout << "channel - " << c_index <<std::endl;
				// this_filter_channel.printMatrix(" this_filter_channel : ");
				// this_channel.printMatrix("this channel :");
				int o_rows_index = 0;
				int o_cols_index = 0;
				for(int i = 0; i < this_channel.rows(); i += stride){ // iterate over the image
					for(int j = 0; j < this_channel.cols(); j += stride){
						Matrix<T> input_window = this_channel.window(i,j,f_rows, f_cols); // get  window same size as filter
						// input_window.printMatrix(" input_window: ");
						T result = (input_window.multiplyElements(this_filter_channel).sum());
						// std::cout << " result - " << result <<std::endl;
						// std::cout << " o_rows_index = " << o_rows_index;
						// std::cout << " o_cols_index = " << o_cols_index << std::endl;
						output_image.data[f_index](o_rows_index,o_cols_index) = result + output_image.data[f_index](o_rows_index,o_cols_index);
						// std::cout << " output_image : " <<std::endl;
						// output_image.printImage();
						o_cols_index++;
						if(o_cols_index > o_cols -1) o_cols_index = 0;
					}
					++o_rows_index;
					if(o_rows_index > o_rows - 1) o_rows_index = 0;
				}
				c_index++;
			}
			output_image.data[f_index] = output_image.data[f_index].add(bias);
			f_index++;
		}
		return_tensor.push_back(output_image);
	}

	return return_tensor;

}

template <typename T>
Tensor<T> batchnorm_layer(const Tensor<T>  &input, float gamma, float beta, float eps){
	int batch_size = input.data.size();
	Tensor<T> return_tensor(batch_size);

	for(Image<T> this_image: input.data){
		Image<T> output_image(this_image.im_channels(), this_image.rows(), this_image.cols());
		int c_index = 0;
		for(Matrix<T> this_channel: this_image.data){
			float mu  = this_channel.sum()/this_channel.size();
			float var = (( pow( this_channel.array()- mu , 2) ).sum()) / this_channel.size(); // sum( (x-mu)^2 ) / size

			std::valarray<T> this_matrix_norm = (this_channel.array() - mu) / std::sqrt(var + eps);

			std::valarray<T> this_matrix_output = gamma*this_matrix_norm + beta;

			output_image.data[c_index].update(this_matrix_output);
			++c_index;
		}
		return_tensor.push_back(output_image);
	}
	return return_tensor;

}

template <typename T>
Tensor<T> relu_layer(const Tensor<T>  &input){
	int batch_size = input.data.size();
	Tensor<T> return_tensor(batch_size);
	for(Image<T> this_image: input.data){
		Image<T> output_image(this_image.im_channels(), this_image.rows(), this_image.cols());
		int c_index = 0;
		for(Matrix<T> this_channel: this_image.data){
			output_image.data[c_index].update(this_channel.relu().array());
			++c_index;
		}
		return_tensor.push_back(output_image);
	}
	return return_tensor;
}