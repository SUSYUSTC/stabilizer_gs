#include "linear.hpp"

void test_mod() {
	Eigen::VectorXi vec(4);
	vec << -2, -1, 0, 1;
	std::cout << "Mod: " << mod(vec.array(), 2).transpose() << std::endl;
}

void test_argsort() {
    // Create a test vector
	Eigen::VectorXi vec(5);
	vec << 1, 3, 5, 4, 5;

	// Find the index of the maximum value
	auto arg = argsort(vec);

	std::cout << "Argsort: " << arg.transpose() << std::endl;

}

void test_get_inv_list() {
    int N = 7;
    Eigen::VectorXi inv_list = get_inv_list(N);

    std::cout << "Inverse list for N=" << N << ": " << inv_list.transpose() << std::endl;
}

void test_argmax() {
    // Create a test vector
	Eigen::VectorXi vec(5);
	vec << 1, 3, 5, 4, 5;

	// Find the index of the maximum value
	int index = argmax(vec);

	std::cout << "Index of maximum value " << vec.transpose() << ": " << index << std::endl;

}

void test_get_order() {
	Eigen::MatrixXi A(3, 3);
	A << 0, 1, 0,
		 0, 0, 0,
		 2, 0, 0;

	Eigen::VectorXi order = linear::get_order(A);

	std::cout << "Order: " << order.transpose() << std::endl;
	return;
}

void test_Gaussian_elimination() {
	int N = 3;
	Eigen::MatrixXi A(3, 3);
	A << 1, 0, 1,
		 0, 1, 1,
		 1, 1, 0;

	Eigen::MatrixXi b = Eigen::MatrixXi::Identity(3, 3);

	auto [A2, b2] = linear::Gauss_elimination(A, b, N);

	std::cout << "A2: " << std::endl << A2 << std::endl;
	std::cout << "b2: " << std::endl << b2 << std::endl;
    std::cout << "b2A: " << std::endl << mod((b2 * A).array(), N) << std::endl;
}

void test_map() {
	Eigen::MatrixXi A(3, 3);
	A << 1, 0, 3,
		 0, 2, 2,
		 2, 1, 0;
    auto result = map_colwise_vec<int>(argsort, A);
    std::cout << "map argsort\n" << result << std::endl;
    auto result2 = map_rowwise_scalar<int>(argmax, A);
    std::cout << "map argmax\n" << result2.transpose() << std::endl;
}

void test_solve_modN() {
    int N = 2;
    Eigen::MatrixXi A(3, 3);
    A << 1, 0, 1,
         0, 1, 1,
         1, 1, 1;
    Eigen::MatrixXi b(3, 3);
    b << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    auto result = linear::solve_modN(A, b, N);
    std::cout << "is over " << std::get<0>(result).transpose() << std::endl;
    std::cout << "solve_modN\n" << mod((A * std::get<2>(result)).array(), N) << std::endl;
}

void test_kernel() {
    int N = 2;
    Eigen::MatrixXi A(2, 3);
    A << 1, 0, 1,
         0, 1, 1;
    auto A2 = linear::Gauss_elimination_Aonly(A, N, true);
    auto kernel = linear::get_kernel_reordered(A2, N);
    std::cout << "kernel " << kernel.transpose() << std::endl;
    std::cout << "AK " << mod((A * kernel).array(), N).transpose() << std::endl;
}

void test_simplify_coset_representative() {
    int N = 2;
    Eigen::MatrixXi A(2, 3);
    A << 1, 0, 0,
         0, 1, 1;
    Eigen::VectorXi x(3);
    x << 0, 1, 0;
    auto result = linear::simplify_coset_representative(A, x, N);
    std::cout << "coset " << result.transpose() << std::endl;
}

int main() {
	test_mod();
    test_get_inv_list();
	test_argmax();
	test_argsort();
	test_get_order();
    //test_get_rows();
    test_map();
	test_Gaussian_elimination();
    test_solve_modN();
    test_kernel();
    test_simplify_coset_representative();
    /*
	Eigen::MatrixXi A(3, 3);
	A << 1, 0, 3,
		 0, 2, 2,
		 2, 1, 0;
    int sum = 0;
    for (int i = 0; i < 10000000; i++) {
        A = map_colwise_vec<int>(argsort, A);
        sum += A(0, 1);
    }
    std::cout << "sum" << sum << std::endl;
    */
    return 0;
}
