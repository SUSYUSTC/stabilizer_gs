#ifndef STATE_HPP
#define STATE_HPP

#include "stab.hpp"

extern int state_count;
extern int id_count;

std::vector<int> arange(int n);

size_t hash_vector(std::vector<int> vec);

using PauliPair = std::pair<Pauli, double>;
using PauliPairs = std::vector<PauliPair>;
using PauliPairsByLast = std::map<int, PauliPairs>;

class Hamiltonian {
public:
    int n;

    int k;

    PauliPairsByLast paulis;

    PauliPairs original_paulis;

    void copy_initializer(int n, int k, const PauliPairsByLast& paulis, const PauliPairs& original_paulis);

    Hamiltonian();

    Hamiltonian(int n, int k);

    Hamiltonian(const Hamiltonian& hamiltonian);

    Hamiltonian& operator=(const Hamiltonian& hamiltonian);

    static Hamiltonian from_file(std::string path);

    void add_Pauli(const Pauli& P, double w);

    std::vector<std::pair<Pauli, double>> paulis_list() const;

    void check_locality() const;
};

template <typename K, typename T>
void update_dict(std::map<K, T>& dict, const std::vector<std::pair<K, T>>& items, T (*merge)(const T&, const T&));

using type_branch = std::tuple<Stab, std::vector<Pauli>>;
using type_stab_maps = std::map<size_t, std::vector<int>>;
using type_Sright = std::tuple<std::vector<Stab>, type_stab_maps>;

class State {
public:
    int n;
    int k;
    Hamiltonian* hamiltonian;
    int m;
    StabEnergies stab;
    std::map<int, std::vector<int>> valid_pauli_indices;
    Stab Sright;

    State();

    State(Hamiltonian& hamiltonian, const Stab& Sright);

    void copy_initializer(const State& other);

    State(const State& other);

    State& operator=(const State& state);

    State copy() const;

    std::vector<int> valid_range() const;

    std::vector<int> Sright_range() const;

    std::vector<Pauli> get_valid_Ps(const std::vector<int>& valid_range) const;

    size_t valid_pauli_indices_hash() const;

    size_t hash_value(bool add_position = true) const;

    std::string to_string() const;

    bool check_Sright_validity(const Stab& Sright_next) const;

    void add(const Pauli& P, int index=-1);

    std::optional<State> move_to_next() const;

    static State merge_gs(const State& state1, const State& state2);

    Stab get_stab(int arg) const;

    Stab get_stab_gs() const;
};

class StateMachine {
public:
    int n;
    int k;
    int m;
    int nstates;
    Hamiltonian* hamiltonian;
    std::map<size_t, State> state_dict;
    std::vector<State> state_list;
    std::map<int, type_Sright> Sright_all;
    State state_gs;
    Eigen::VectorXi index_sign;
    double energy_gs;

    StateMachine(Hamiltonian& hamiltonian, int nstates = 1);

    std::vector<State> evolve_single(const State& raw_state);

    void evolve();

    State generate_gs();
};

class FullState {
public:
    StabEnergies stab;
    std::vector<std::pair<Pauli, double>> paulis;
    std::vector<Pauli> excluded_paulis;

    FullState();

    FullState(const std::vector<std::pair<Pauli, double>>& paulis);

    FullState copy() const;

    FullState include_pauli(const Pauli& P) const;

    FullState exclude_pauli(const Pauli& P) const;

    std::string to_string() const;

    std::tuple<Stab, double> evolve();
};

#endif
