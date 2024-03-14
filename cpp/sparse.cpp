#include <iomanip>
#include "state.hpp"

int main(int argc, char** argv) {
    std::string path = argv[1];
    Hamiltonian H = Hamiltonian::from_file(path);

    FullState state(H.paulis_list());
    auto [stab, energy_gs] = state.evolve();
    std::cout << "ground state stabilizers" << std::endl;
    std::cout << stab.range(0, H.n).to_string(true) << std::endl;
    std::cout << std::setprecision(16) << "ground state energy " << energy_gs << std::endl;
    std::cout << "terms signs" << std::endl;
    for (auto [P, w]: H.original_paulis) {
        std::cout << stab.energy(P) << " ";
    }
    std::cout << std::endl;
}