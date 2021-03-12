#pragma once
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <valarray>

template <typename T>
class Matrix
{
public:
    Matrix(size_t height, size_t width): m_cols(width), m_rows(height), m_storage(width*height) {  }
    Matrix(size_t height, size_t width, std::valarray<T> &data): m_cols(width), m_rows(height), m_storage(data){  }

    // write new data into the matrix
    std::valarray<T> update(const std::valarray<T> &data);

    // get the number of rows in the Matrix
    std::size_t rows() const {return m_rows;}
    // get the number of columns in the Matrix
    std::size_t cols() const {return m_cols;}
    // get a copy of the data in the Matrix
    std::valarray<T> array() const {return m_storage;}
    // retrieve the data from row r of the Matrix
    std::valarray<T> row(std::size_t r) const;
    // retrieve the data from col c of the Matrix
    std::valarray<T> col(std::size_t c) const;
    // retrieve the size of the array
    std::size_t size() const { return m_storage.size();}

    // basic item reference
    T& operator()(std::size_t r, std::size_t c);
    // basic item retrieval
    T operator()(std::size_t r, std::size_t c) const;

    // Multiplies 2 Matrix
    Matrix<T> multiply(const Matrix<T>& a) const;
    // multiply the elements
    Matrix<T> multiplyElements(const Matrix<T>& a) const;
    // add together
    Matrix<T> add(const Matrix<T>& a) const;
    // add constant
    Matrix<T> add(T a) const;
    // relu
    Matrix<T> relu() const;
    // sigmoid
    Matrix<T> sigmoid() const;
    // tanh
    Matrix<T> tanh() const;
    // sum along rows
    Matrix<T> sumRows() const;
    // sum along cols
    Matrix<T> sumCols() const;
    // whole sum
    T sum() const;
    // transposition of this one
    Matrix<T> transpose() const;
    // get a matrix window centered at x with width and height
    Matrix<T> window(int center_w,int center_h, int width, int height) const;


    void printMatrix(const char* label);
    void printMatrix();

    void printArray(const char* label) {
	    std::cout << label << ": ";
	    for(std::size_t i = 0; i < m_storage.size(); ++i){
	        std::cout << m_storage[i] << ' ';
	    }
	    std::cout << std::endl;

	}

private:
    size_t m_cols;
    size_t m_rows;
    std::valarray<T> m_storage;
};

template <typename T>
Matrix<T> Matrix<T>::sigmoid() const{
    std::valarray<T> result_var = 1/(1+std::exp(-m_storage));
    Matrix<T> result(m_rows, m_cols, result_var);
    return result;
}

template <typename T>
Matrix<T>  Matrix<T>::relu() const{
    Matrix<T> result(m_rows, m_cols);
    for(size_t i = 0; i < m_rows; ++i){
        for(size_t j = 0; j < m_cols; ++j) {
            if(row(i)[j] <= 0) {
                result(i,j) = 0;
            } else {
                result(i,j) = row(i)[j];
            }
        }
    }
    return result;
}

template <typename T>
Matrix<T>  Matrix<T>::tanh() const{
    std::valarray<T> result_var = std::tanh(m_storage);
    Matrix<T> result(m_rows, m_cols, result_var);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::add(const Matrix<T>& a) const{
    size_t a_rows = a.rows();
    size_t a_cols = a.cols();
    assert(a_cols == m_cols);
    assert(a_rows == m_rows);
    std::valarray<T> result_var = m_storage + a.array();
    Matrix<T> result(m_rows, m_cols, result_var);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::add(T a) const{
    std::valarray<T> result_var = m_storage + a;
    Matrix<T> result(m_rows, m_cols, result_var);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& a) const{
    size_t a_rows = a.rows();
    size_t a_cols = a.cols();
    assert(a_cols == m_rows);
    Matrix<T> result(a_cols, m_rows);
    for(size_t i = 0; i < a_cols; ++i){
        for(size_t j = 0; j < m_rows; ++j) {
            // dot product of row[i] and col a[j]
            std::valarray<T> this_row = row(i);
            std::valarray<T> this_col = a.col(j);
            result(i,j) = (this_row * this_col).sum();
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::multiplyElements(const Matrix<T>& a) const{
    size_t a_rows = a.rows();
    size_t a_cols = a.cols();
    assert(a_cols == m_cols);
    assert(a_rows == m_rows);
    Matrix<T> result(m_rows, m_cols);
        for(size_t i = 0; i < m_rows; ++i){
        std::valarray<T> this_row = row(i);
        std::valarray<T> a_row    = a.row(i);
        for(size_t j = 0; j < m_cols; ++j) {
            result(i,j) = a_row[j] * this_row[j];
        }
    }
    return result;
}


template <typename T>
std::valarray<T> Matrix<T>::update(const std::valarray<T> &data) { 
    m_storage = data;
    return m_storage;
}

template <typename T>
std::valarray<T> Matrix<T>::row(std::size_t r) const{
    return m_storage[std::slice(r * cols(), cols(), 1)];
}

template <typename T>
std::valarray<T> Matrix<T>::col(std::size_t c) const{
    return m_storage[std::slice(c, rows(), cols())];

}

// basic item reference
template <typename T>
T& Matrix<T>::operator()(std::size_t r, std::size_t c){
    assert(m_rows > r);
    assert(m_cols > c); 
    return m_storage[r * cols() + c];
}

// basic item retrieval
template <typename T>
T Matrix<T>::operator()(std::size_t r, std::size_t c) const {
    assert(m_rows > r);
    assert(m_cols > c); 
    return row(r)[c];
}

template <typename T>
Matrix<T> Matrix<T>::sumRows() const{
    Matrix<T> result(m_rows,1);
    for(size_t i = 0; i < m_rows; ++i){
        result(i,0) = row(i).sum();
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::sumCols() const{
    Matrix<T> result(1,m_cols);
    for(size_t i = 0; i < m_cols; ++i){
        result(0,i) = col(i).sum();
    }
    return result;

}

template <typename T>
T Matrix<T>::sum() const{
    return m_storage.sum();
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const{
  Matrix<T> result(m_cols, m_rows);
  for(std::size_t i = 0; i < m_cols  ; ++i){
    for(std::size_t j = 0; j < m_rows ; ++j){
        result(i,j) = col(i)[j];
    }
  }

  return result;
}

template <typename T>
Matrix<T> Matrix<T>::window(int center_r, int center_c, int height, int width) const{
    /* https://github.com/MichaelHancock/Matrix/blob/master/matrix.h */
    int row1 = center_r - height/2;
    int row2 = center_r + height/2;
    int col1 = center_c - width/2;
    int col2 = center_c + width/2;
    Matrix<T> result(height,width);
    int indexRow = 0;
    int indexCol = 0;

    for (int i = row1; i <= row2; ++i){
        for (int j = col1; j <= col2; ++j){
            result(indexRow, indexCol) = row(i)[j];
            indexCol++;
        }
        indexRow++;
        indexCol = 0;
    }
    return result;
}



template <typename T>
void Matrix<T>::printMatrix(const char* label){
    std::cout << label << ": " << std::endl;
    size_t size = m_cols * m_rows;
    assert(size <= m_storage.size());
    for(size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << m_storage[i];
        std::cout << ((i+1)%m_cols ? ' ' : '\n');
    }
    std::cout << std::endl;
}

template <typename T>
void Matrix<T>::printMatrix(){
    size_t size = m_cols * m_rows;
    assert(size <= m_storage.size());
    for(size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << m_storage[i];
        std::cout << ((i+1)%m_cols ? ' ' : '\n');
    }
    std::cout << std::endl;
}