#include "linear.hpp"

namespace linear {
    Eigen::VectorXi get_order(const Eigen::MatrixXi& A) {
        int n = A.rows();
        int m = A.cols();

        // Sum along rows to find zeros
        Eigen::VectorXi weights(n);
        for (int i = 0; i < n; ++i) {
            if (A.row(i).sum() == 0) {
                weights(i) = m;
            }
            else {
                weights(i) = argmax(get_binary(A.row(i)));
            }
        }
        return argsort(weights);
    }

    std::pair<Eigen::MatrixXi, Eigen::MatrixXi> Gauss_elimination(const Eigen::MatrixXi& input_A, const Eigen::MatrixXi& input_b, int N, bool allow_reorder) {
        Eigen::MatrixXi A = input_A;
        Eigen::MatrixXi b = input_b;

        int n = A.rows();
        int m = A.cols();

        if (n == 0 || m == 0) {
            return std::make_pair(A, b);
        }

        Eigen::VectorXi inv_list = get_inv_list(N);

        auto div = [&N, &inv_list](int a, int b) { return (a * inv_list(b)) % N; };

        auto update = [&div, &N, &n](Eigen::MatrixXi& old_A, Eigen::MatrixXi& old_b, int k, int pivot, std::pair<int, int> slic) {
            auto [begin, end] = slic;
            Eigen::MatrixXi new_A = old_A;
            Eigen::MatrixXi new_b = old_b;
            for (int i = begin; i < end; ++i) {
                int f = div(old_A(i, pivot), old_A(k, pivot));
                new_A.row(i) = mod((old_A.row(i) - old_A.row(k) * f).array(), N);
                new_b.row(i) = mod((old_b.row(i) - old_b.row(k) * f).array(), N);
            }
            return std::make_pair(new_A, new_b);
        };

        std::vector<int> pivots;
        for (int k = 0; k < n; ++k) {
            int pivot = argmax(get_binary(A.row(k)));
            pivots.push_back(pivot);
            auto slic = std::make_pair(k+1, n);
            std::tie(A, b) = update(A, b, k, pivot, slic);
        }

        for (int k = n - 1; k >= 0; --k) {
            int pivot = pivots[k];
            auto slic = std::make_pair(0, k);
            std::tie(A, b) = update(A, b, k, pivot, slic);

            int inv = inv_list(A(k, pivot));
            if (inv != 0) {
                A.row(k) = mod((A.row(k) * inv).array(), N);
                b.row(k) = mod((b.row(k) * inv).array(), N);
            }
        }

        if (allow_reorder) {
            Eigen::VectorXi order = get_order(A);
            A = A(order, Eigen::all).eval();
            b = b(order, Eigen::all).eval();
        }

        return std::make_pair(A, b);
    }

    std::tuple<Eigen::VectorXb, bool, Eigen::MatrixXi> solve_modN(const Eigen::MatrixXi& A, const Eigen::MatrixXi& b, int N, bool canonicalized) {

        int n = A.rows();
        int m = A.cols();
        int k = b.cols();

        assert(b.rows() == n);

        if (m == 0) {
            Eigen::VectorXb is_over = (b.array() > 0).colwise().any();
            Eigen::MatrixXi x = Eigen::MatrixXi::Zero(m, k);
            return std::make_tuple(is_over, false, x);
        }

        Eigen::MatrixXi A2, b2;
        if (canonicalized) {
            A2 = A;
            b2 = b;
        } else {
            std::tie(A2, b2) = Gauss_elimination(A, b, N, true);
        }

        Eigen::VectorXi conditions = A2.rowwise().sum();
        Eigen::MatrixXb cond1 = (conditions.array() == 0).rowwise().replicate(k);
        Eigen::MatrixXb cond2 = (b2.array() != 0);
        Eigen::VectorXb is_over_ref = (cond1 && cond2).colwise().any();

        Eigen::MatrixXi b2_zero = b2(get_where(conditions.array() == 0), Eigen::all);
        Eigen::VectorXb is_over = (b2_zero.array() > 0).colwise().any();

        assert((is_over_ref.array() == is_over.array()).all());

        bool is_under = (conditions.array() > 1).any();

        Eigen::VectorXb nonzero = (conditions.array() > 0);
        Eigen::VectorXi nonzero_indices = get_where(nonzero);
        A2 = A2(nonzero_indices, Eigen::all).eval();
        b2 = b2(nonzero_indices, Eigen::all).eval();

        auto pivots = map_rowwise_scalar<int>(argmax, A2);
        Eigen::MatrixXi x = Eigen::MatrixXi::Zero(m, k);
        x(pivots, Eigen::all) = b2;

        return std::make_tuple(is_over, is_under, x);
    }

    Eigen::MatrixXi Gauss_elimination_Aonly(const Eigen::MatrixXi& A, int N, bool allow_reorder) {
        Eigen::MatrixXi b = Eigen::MatrixXi::Zero(A.rows(), 0);
        auto [A2, b2] = Gauss_elimination(A, b, N, allow_reorder);
        return A2;
    }

    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> Gauss_elimination_full(const Eigen::MatrixXi& A, int N, bool allow_reorder) {
        int n = A.rows();
        Eigen::MatrixXi b = Eigen::MatrixXi::Identity(n, n);
        return Gauss_elimination(A, b, N, allow_reorder);
    }

    Eigen::MatrixXi get_kernel_reordered(const Eigen::MatrixXi& raw_A2, int N) {
        Eigen::VectorXb nonzero = raw_A2.array().rowwise().sum() > 0;
        Eigen::MatrixXi A2 = raw_A2(get_where(nonzero), Eigen::all).eval();
        int n = A2.rows();
        int m = A2.cols();

        if (m == 0) {
            return Eigen::MatrixXi::Zero(0, 0);
        }

        Eigen::VectorXi main_cols = map_rowwise_scalar<int>(argmax, get_binary(A2));
        Eigen::VectorXi tmp_array = arange(m, m * 2);
        tmp_array(main_cols) = arange(0, n);
        //set_values<int>(tmp_array, main_cols, arange(0, n));

        Eigen::VectorXi col_order = argsort(tmp_array);
        //Eigen::MatrixXi A2_sort = get_cols(A2, col_order);
        Eigen::MatrixXi A2_sort = A2(Eigen::all, col_order);

        Eigen::MatrixXi b_sort = Eigen::MatrixXi::Identity(m - n, m - n);
        Eigen::MatrixXi a_sort = mod(-A2_sort.rightCols(m - n).array(), N);

        Eigen::MatrixXi x_sort(m, m - n);
        x_sort << a_sort, b_sort;
        Eigen::MatrixXi x = Eigen::MatrixXi::Zero(m, m - n);
        //set_rows_values(x, col_order, x_sort);
        x(col_order, Eigen::all) = x_sort;
        return x;
    }

    Eigen::MatrixXi get_kernel(const Eigen::MatrixXi& A, int N) {
        Eigen::MatrixXi A2 = Gauss_elimination_Aonly(A, N, true);
        return get_kernel_reordered(A2, N);
    }

    Eigen::VectorXi simplify_coset_representative(const Eigen::MatrixXi& A2, const Eigen::VectorXi& x, int N) {
        if (A2.rows() == 0) {
            return x;
        }

        Eigen::VectorXi arg = map_rowwise_scalar<int>(argmax, get_binary(A2));
        // assert((A2.rowwise().sparseView().value(arg) == 1).all());
        //return mod(x - A2.transpose() * get_values(x, arg), N);
        return mod((x - A2.transpose() * x(arg)).array(), N);
    }
}