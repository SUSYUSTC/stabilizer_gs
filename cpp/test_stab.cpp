#include <cmath>
#include "stab.hpp"

int main() {
    //std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::cout << std::boolalpha;
    std::srand(static_cast<unsigned int>(0));
    Pauli pauli(Eigen::MatrixXi::Zero(0, 6), Eigen::MatrixXi::Zero(0, 6), 0);
    Pauli pauli1 = Pauli::random(3, 6, 0);
    Pauli pauli2 = Pauli::random(4, 6, 1);
    std::cout << pauli1 << std::endl;
    std::cout << pauli2 << std::endl;
    auto commute = pauli1.commute(pauli2);
    std::cout << pauli1.commute(pauli2).transpose() << std::endl;
    std::cout << pauli1 << pauli1.hash_value() << std::endl;
    std::cout << pauli1.copy() << pauli1.copy().hash_value() << std::endl;
    Stab stab = Stab::random_stab(4, 2);
    std::cout << "stab\n" << stab.Ps << std::endl;
    for (int i = 0; i < stab.Ps.size; i++) {
        for (int j = 0; j < stab.Ps.size; j++) {
            std::cout << stab.Ps[i].commute(stab.Ps[j]) << " ";
        }
        std::cout << std::endl;
    }
    auto Ps_reduce = stab.Ps.reduce_dot();
    std::cout << Ps_reduce << std::endl;
    std::cout << stab.include(Ps_reduce) << std::endl;
    std::cout << stab.commute(Ps_reduce) << std::endl;
    Eigen::VectorXd Es = Eigen::VectorXd::Random(4);
    std::cout << Es << std::endl;
    std::cout << "before canonicalization" << std::endl;
    std::cout << stab.Ps << std::endl;
    std::cout << Es << std::endl;
    /*
    StabEnergies Estab(stab.Ps, Es);
    std::cout << "after canonicalization" << std::endl;
    std::cout << Estab.Ps << std::endl;
    std::cout << Estab.Es << std::endl;
    */
    return 0;
}