#include "state.hpp"
#include <iomanip>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("verbose,v", po::value<int>()->default_value(0), "verbose level")
        ("next,n", po::value<int>()->default_value(0), "number of excited states")
        ("filename", po::value<std::string>()->required(), "filename (required)");

    po::positional_options_description p;
    p.add("filename", 1); // Allows 1 positional argument for "filename"

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    std::string path = vm["filename"].as<std::string>();
    int next = vm["next"].as<int>();
    int verbose = vm["verbose"].as<int>();
    std::cout << "verbose level " << verbose << std::endl;

    Hamiltonian H = Hamiltonian::from_file(path);

    StateMachine SM(H, next + 1);
    if (verbose > 0) {
        for (int i = 0; i< SM.n; i++) {
            auto Sright = std::get<0>(SM.Sright_all.at(i));
            std::cout << i << std::endl;
            for (auto stab: Sright) {
                std::cout << stab.to_string() << std::endl;
            }
            std::cout << std::endl;
        }
    }
    while (true) {
        auto begin = std::chrono::high_resolution_clock::now();
        SM.evolve();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "m " << SM.m << std::endl;
        std::cout << "number of states " << SM.state_dict.size() << std::endl;
        size_t us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "time " << us * 1e-6 << std::endl;
        std::cout << std::endl;
        if (verbose > 0) {
            int count = 0;
            for (auto [key, state]: SM.state_dict) {
                std::cout << count << std::endl;
                std::cout << state.to_string() << std::endl;
                if (verbose > 1) {
                    std::cout << state.stab.hash_value(false) << " ";
                    std::cout << state.valid_pauli_indices_hash() << " ";
                    std::cout << state.Sright.hash_value(false) << " ";
                }
                std::cout << std::endl;
                count++;
            }
        }
        if (SM.m == H.n) {
            break;
        }
    }

    State state_gs = SM.generate_gs();
    Stab stab = state_gs.get_stab_gs().range(0, H.n);
    std::cout << "ground state stabilizers" << std::endl;
    std::cout << stab.range(0, H.n).to_string(true) << std::endl;
    std::cout << std::setprecision(16) << "ground state energy " << state_gs.stab.Es.minCoeff() << std::endl;
    std::cout << "terms signs" << std::endl;
    for (auto [P, w]: H.original_paulis) {
        std::cout << stab.energy(P) << " ";
    }
    std::cout << std::endl;
}
