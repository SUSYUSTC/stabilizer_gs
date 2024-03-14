#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Eigen {
    template<typename Scalar>
    using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar>
    using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar>
    using RowVectorX = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    using MatrixXb = MatrixX<bool>;
    using VectorXb = VectorX<bool>;
    using RowVectorXb = RowVectorX<bool>;
}

template <typename T>
std::ostream& print_eigen(std::ostream &os, const T& M) {
    os << M << std::endl;
    return os;
}

std::ostream& pmi(std::ostream &os, const Eigen::MatrixXi& M) {
    return print_eigen(os, M);
}

std::ostream& pmd(std::ostream &os, const Eigen::MatrixXd& M) {
    return print_eigen(os, M);
}

std::ostream& pmb(std::ostream &os, const Eigen::MatrixXb& M) {
    return print_eigen(os, M);
}

std::ostream& pvi(std::ostream &os, const Eigen::VectorXi& M) {
    return print_eigen(os, M);
}

std::ostream& pvd(std::ostream &os, const Eigen::VectorXd& M) {
    return print_eigen(os, M);
}

std::ostream& pvb(std::ostream &os, const Eigen::VectorXb& M) {
    return print_eigen(os, M);
}

template <typename Scalar>
using vec2scalar = Scalar (*)(const Eigen::VectorX<Scalar>&);
template <typename Scalar>
using vec2vec = Eigen::VectorX<Scalar> (*)(const Eigen::VectorX<Scalar>&);

template <typename Scalar>
Eigen::MatrixX<Scalar> map_rowwise_vec(const vec2vec<Scalar>& func, const Eigen::MatrixX<Scalar>& M){
    Eigen::MatrixX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(M.rows(), M.cols());
    for (int i = 0; i < M.rows(); i++) {
        result.row(i) = func(M.row(i));
    }
    return result;
}

template <typename Scalar>
Eigen::VectorX<Scalar> map_rowwise_scalar(const vec2scalar<Scalar>& func, const Eigen::MatrixX<Scalar>& M){
    Eigen::VectorX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(M.rows(), 1);
    for (int i = 0; i < M.rows(); i++) {
        result(i) = func(M.row(i));
    }
    return result;
}

template <typename Scalar>
Eigen::MatrixX<Scalar> map_colwise_vec(const vec2vec<Scalar>& func, const Eigen::MatrixX<Scalar>& M){
    Eigen::MatrixX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(M.rows(), M.cols());
    for (int i = 0; i < M.cols(); i++) {
        result.col(i) = func(M.col(i));
    }
    return result;
}

template <typename Scalar>
Eigen::VectorX<Scalar> map_colwise_scalar(const vec2vec<Scalar>& func, const Eigen::MatrixX<Scalar>& M){
    Eigen::VectorX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(M.rows(), 1);
    for (int i = 0; i < M.cols(); i++) {
        result(i) = func(M.col(i));
    }
    return result;
}

Eigen::VectorXi arange(int n, int m) {
    Eigen::VectorXi result(m - n);
    for (int i = n; i < m; i++) {
        result(i - n) = i;
    }
    return result;
}

template<typename T>
auto mod(const T& X, int N) {
	auto newX = X + N;
	return newX - (newX / N) * N;
}

Eigen::VectorXi get_where(const Eigen::VectorXb& valid) {
    int nvalid = valid.cast<int>().sum();
    Eigen::VectorXi result(nvalid);
    int count = 0;
    for (int i = 0; i < valid.size(); i++){
        if (valid(i)) {
            result(count) = i;
            count++;
        }
    }
    return result;
}

template<typename Scalar>
void set_matrix_values(Eigen::MatrixX<Scalar>& X, const Eigen::VectorXi& row_indices, const Eigen::VectorXi& col_indices, const Eigen::VectorX<Scalar>& values) {
    assert(X.rows() > row_indices.maxCoeff());
    assert(X.cols() > col_indices.maxCoeff());
    assert(row_indices.size() == values.size());
    assert(col_indices.size() == values.size());
    for (int i = 0; i < values.size(); i++){
        X(row_indices(i), col_indices(i)) = values(i);
    }
}

Eigen::VectorXi get_inv_list(int N) {
    Eigen::VectorXi result(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if ((i * j) % N == 1) {
                result[i] = j;
                break;
            }
        }
    }
    result(0) = 0;
    return result;
}

template <typename T>
auto get_binary(T M) {
    return (M.array() > 0).template cast<int>();
}

int argmax(const Eigen::VectorXi& vec){
	int index = -1;
	vec.maxCoeff(&index);
	return index;
}

int argmin(const Eigen::VectorXd& vec){
    int index = -1;
    vec.minCoeff(&index);
    return index;
}

// Function to get indices that would sort an Eigen vector
Eigen::VectorXi argsort(const Eigen::VectorXi& vec) {
    Eigen::VectorXi indices(vec.size());
    std::iota(indices.data(), indices.data() + vec.size(), 0);

    std::sort(indices.data(), indices.data() + vec.size(), [&vec](int i, int j) {
        return vec(i) < vec(j);
    });

    return indices;
}

template <typename Scalar>
Scalar get_item(const Eigen::MatrixX<Scalar>& M) {
    assert(M.rows() == 1);
    assert(M.cols() == 1);
    return M(0, 0);
}

namespace linear {
    Eigen::VectorXi get_order(const Eigen::MatrixXi& A);

    std::pair<Eigen::MatrixXi, Eigen::MatrixXi> Gauss_elimination(const Eigen::MatrixXi& input_A, const Eigen::MatrixXi& input_b, int N, bool allow_reorder = false);

    std::tuple<Eigen::VectorXb, bool, Eigen::MatrixXi> solve_modN(const Eigen::MatrixXi& A, const Eigen::MatrixXi& b, int N, bool canonicalized = false);

    Eigen::MatrixXi Gauss_elimination_Aonly(const Eigen::MatrixXi& A, int N, bool allow_reorder = false);

    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> Gauss_elimination_full(const Eigen::MatrixXi& A, int N, bool allow_reorder = false);

    Eigen::MatrixXi get_kernel_reordered(const Eigen::MatrixXi& A2, int N);

    Eigen::MatrixXi get_kernel(const Eigen::MatrixXi& A, int N);

    Eigen::VectorXi simplify_coset_representative(const Eigen::MatrixXi& A2, const Eigen::VectorXi& x, int N);
}
#endif