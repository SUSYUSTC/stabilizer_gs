#include "state.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <optional>
#include <boost/chrono.hpp>

std::vector<int> arange(int n) {
    std::vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = i;
    }
    return result;
}

size_t hash_vector(std::vector<int> vec) {
    size_t seed = 0;
    for (auto& item : vec) {
        boost::hash_combine(seed, item);
    }
    return seed;
}

int state_count = 0;
int id_count = 0;

void Hamiltonian::copy_initializer(int n, int k, const PauliPairsByLast& paulis, const PauliPairs& original_paulis) {
    this->n = n;
    this->k = k;
    this->paulis = paulis;
    this->original_paulis = original_paulis;
}

Hamiltonian::Hamiltonian() {
}

Hamiltonian::Hamiltonian(int n, int k) {
    PauliPairsByLast paulis = {};
    for(int qlast = 0; qlast < n; qlast++) {
        paulis[qlast] = {};
    }
    PauliPairs original_paulis = {};
    this->copy_initializer(n, k, paulis, original_paulis);
}

Hamiltonian::Hamiltonian(const Hamiltonian& h) {
    this->copy_initializer(h.n, h.k, h.paulis, h.original_paulis);
}

Hamiltonian& Hamiltonian::operator=(const Hamiltonian& h) {
    if (this != &h) {
        this->copy_initializer(h.n, h.k, h.paulis, h.original_paulis);
    }
    return *this;
}

Hamiltonian Hamiltonian::from_file(std::string path) {
    int n;
    int k;
    std::ifstream infile(path);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> n >> k;
    Hamiltonian H(n, k);
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double w;
        std::string pauli_str;
        std::vector<int> qubits;
        iss >> w >> pauli_str;
        int tmp;
        while (iss >> tmp) {
            qubits.push_back(tmp);
        }
        Pauli P(pauli_str, qubits);
        H.add_Pauli(P, w);
    }
    return H;
}

void Hamiltonian::add_Pauli(const Pauli& P, double w) {
    this->paulis[P.end()-1].push_back(std::make_pair(P, w));
    this->original_paulis.push_back(std::make_pair(P, w));
}

PauliPairs Hamiltonian::paulis_list() const {
    std::vector<std::pair<Pauli, double>> result;
    for (int qlast = 0; qlast < this->n; qlast++) {
        for (auto& item : this->paulis.at(qlast)) {
            result.push_back(std::make_pair(item.first.range(0, n), item.second));
        }
    }
    return result;
}

void Hamiltonian::check_locality() const {
    for (int qlast = 0; qlast < this->n; qlast++) {
        auto vec = this->paulis.at(qlast);
        for (const auto& item : vec) {
            assert(item.first.end() - item.first.begin <= this->k);
        }
    }
}

template <typename K, typename T>
void update_dict(std::map<K, T>& dict, const std::vector<std::pair<K, T>>& items, T (*merge)(const T&, const T&)) {
    for (auto& item : items) {
        K key = item.first;
        if (dict.find(key) == dict.end()) {
            dict[key] = item.second;
        } else {
            dict[key] = merge(dict[key], item.second);
        }
    }
}

Eigen::VectorXi unravel_index(int index, int n) {
    Eigen::VectorXi result(n);
    for (int i = 0; i < n; i++) {
        result(i) = index % 2;
        index /= 2;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::optional<type_branch> branch_include(type_branch branch, const Pauli& P) {
    auto [stab, exclude_paulis] = branch;
    stab = stab.add(P);
    for (auto& ep : exclude_paulis) {
        if (stab.include(ep)) {
            return std::nullopt;
        }
    }
    return std::make_tuple(stab, exclude_paulis);
}

type_branch branch_exclude(type_branch branch, const Pauli& P) {
    auto [stab, exclude_paulis] = branch;
    std::vector<Pauli> new_exclude_paulis = exclude_paulis;
    new_exclude_paulis.push_back(P);
    return std::make_tuple(stab, new_exclude_paulis);
}

std::vector<Stab> evolve_stab(const Stab& stab, const std::vector<Pauli>& paulis, bool project=true) {
    int begin = stab.begin();
    int end = stab.end();
    for (auto p: paulis) {
        assert(p.end() == end - 1);
    }
    std::vector<type_branch> branches = {std::make_tuple(stab, std::vector<Pauli>())};
    for (auto& P : paulis) {
        std::vector<type_branch> new_branches;
        for (auto& branch : branches) {
            auto [stab, exclude_paulis] = branch;
            if (stab.commute(P) && !stab.include(P)) {
                auto branch1 = branch_include(branch, P);
                auto branch2 = branch_exclude(branch, P);
                if (branch1.has_value()) {
                    new_branches.push_back(branch1.value());
                }
                new_branches.push_back(branch2);
            } else {
                new_branches.push_back(branch);
            }
        }
        branches = new_branches;
    }
    int begin_next = std::max(begin - 1, 0);
    std::vector<Stab> stabs;
    for (auto& branch : branches) {
        auto [stab, _] = branch;
        stabs.push_back(stab.range(begin_next, end));
    }
    if (project) {
        std::vector<Stab> stabs_projected;
        for (auto& stab : stabs) {
            stabs_projected.push_back(stab.project(begin_next, end - 1));
        }
        return stabs_projected;
    }
    return stabs;
}

std::tuple<std::vector<Stab>, type_stab_maps> get_map_from_stabs(const std::vector<std::vector<Stab>>& stabs_list) {
    std::vector<std::vector<size_t>> hashs_list;
    for (auto& items : stabs_list) {
        std::vector<size_t> hashs;
        for (auto& item : items) {
            hashs.push_back(item.hash_value());
        }
        hashs_list.push_back(hashs);
    }
    std::map<size_t, Stab> stabs_dict;
    for (auto& items : stabs_list) {
        for (auto& item : items) {
            stabs_dict[item.hash_value()] = item;
        }
    }
    std::vector<Stab> stabs;
    for (auto& item : stabs_dict) {
        stabs.push_back(item.second);
    }
    type_stab_maps stab_maps;
    for (auto& item : stabs) {
        stab_maps[item.hash_value()] = std::vector<int>();
    }
    for (int i = 0; i < hashs_list.size(); i++) {
        for (auto& h : hashs_list[i]) {
            stab_maps[h].push_back(i);
        }
    }
    return std::make_tuple(stabs, stab_maps);
}

std::map<int, type_Sright> generate_Sright_all(const Hamiltonian& H) {
    std::vector<Stab> stabs = {Stab::empty(H.k, H.n - H.k + 1)};
    std::map<int, type_Sright> Sright_all;
    Sright_all[H.n] = std::make_tuple(stabs, type_stab_maps());
    for (int m = H.n - 1; m >= 0; m--) {
        std::vector<Pauli> paulis;
        for (auto& item : H.paulis.at(m)) {
            paulis.push_back(item.first.copy());
        }
        std::vector<std::vector<Stab>> stabs_list;
        for (auto& stab : stabs) {
            stabs_list.push_back(evolve_stab(stab, paulis));
        }
        auto result = get_map_from_stabs(stabs_list);
        stabs = std::get<0>(result);
        auto stabs_maps = std::get<1>(result);
        Sright_all[m] = std::make_tuple(stabs, stabs_maps);
        std::cout << m << ' ' << stabs.size() << std::endl;
    }
    return Sright_all;
}

std::vector<int> arange_vec(int begin, int end) {
    std::vector<int> result;
    for (int i = begin; i < end; i++) {
        result.push_back(i);
    }
    return result;
}

State::State() {
}

State::State(Hamiltonian& hamiltonian, const Stab& Sright) {
    this->n = hamiltonian.n;
    this->k = hamiltonian.k;
    this->hamiltonian = &hamiltonian;
    this->m = 0;
    this->stab = StabEnergies::empty(0, 0, this->n);
    for (int qlast = 0; qlast < this->n; qlast++) {
        this->valid_pauli_indices[qlast] = std::vector<int>(this->hamiltonian->paulis.at(qlast).size());
        std::iota(this->valid_pauli_indices[qlast].begin(), this->valid_pauli_indices[qlast].end(), 0);
    }
    this->Sright = Sright;
}

void State::copy_initializer(const State& other) {
    this->hamiltonian = other.hamiltonian;
    this->k = other.k;
    this->n = other.n;
    this->m = other.m;
    this->stab = other.stab;
    this->valid_pauli_indices.clear();
    for (auto& item : other.valid_pauli_indices) {
        std::vector<int> vec = item.second;
        this->valid_pauli_indices[item.first] = vec;
    }
    this->Sright = other.Sright;
}

State::State(const State& other) {
    this->copy_initializer(other);
}

State& State::operator=(const State& state) {
    if (this != &state) {
        this->copy_initializer(state);
    }
    return *this;
}

State State::copy() const {
    return State(*this);
}

std::vector<int> State::valid_range() const {
    return arange_vec(this->m + 1, std::min(this->m + this->k - 1, this->n));
}

std::vector<int> State::Sright_range() const {
    return arange_vec(this->m + 1, std::min(this->m + this->k + 1, this->n));
}

std::vector<Pauli> State::get_valid_Ps(const std::vector<int>& valid_range) const {
    std::vector<Pauli> Ps;
    for (int qlast : valid_range) {
        for (int i : this->valid_pauli_indices.at(qlast)) {
            Ps.push_back(this->hamiltonian->paulis.at(qlast).at(i).first);
        }
    }
    return Ps;
}

size_t State::valid_pauli_indices_hash() const {
    size_t seed = 0;
    for (int qlast : this->valid_range()) {
        std::vector<int> vec = this->valid_pauli_indices.at(qlast);
        std::sort(vec.begin(), vec.end());
        boost::hash_combine(seed, hash_vector(vec));
    }
    return seed;
}

size_t State::hash_value(bool add_position) const {
    size_t seed = 0;
    if (add_position) {
        boost::hash_combine(seed, this->m);
    }
    boost::hash_combine(seed, this->stab.hash_value(add_position));
    boost::hash_combine(seed, this->valid_pauli_indices_hash());
    boost::hash_combine(seed, this->Sright.hash_value(add_position));
    return seed;
}

std::string State::to_string() const {
    std::string string1 = "m = " + std::to_string(this->m) + "\n";
    std::string string2 = "stab = " + this->stab.to_string() + "\n";
    std::vector<Pauli> valid_Ps = this->get_valid_Ps(this->valid_range());
    std::string string3 = "Sright = " + this->Sright.to_string() + "\n";
    std::string string4 = "valid_pauli_indices = " + Pauli::concatenate(valid_Ps).to_string() + "\n";
    std::stringstream string5;
    string5 << "energies = " << this->stab.Es.transpose() << std::endl;
    std::stringstream string6;
    string6 << "Ps indices \n" << this->stab.Ps_indices << std::endl;
    std::stringstream string7;
    string7 << "Ps signs \n" << this->stab.Ps_signs << std::endl;
    return string1 + string2 + string3 + string4 + string5.str() + string6.str() + string7.str() + '\n';
}

bool State::check_Sright_validity(const Stab& Sright_next) const {
    assert(Sright_next.begin() == std::max(this->m - this->k + 2, 0));
    assert(Sright_next.end() == this->m + 2);
    std::vector<Pauli> valid_Ps_vec = this->get_valid_Ps(this->Sright_range());
    Pauli valid_Ps;
    int begin = std::max(this->m - this->k + 2, 0);
    int end = this->m + 2;
    if (valid_Ps_vec.size() > 0) {
        valid_Ps = Pauli::concatenate(valid_Ps_vec).range(begin, end, true);
    } else {
        valid_Ps = Pauli::identity(end - begin, 0, begin);
    }
    for (auto& P : Sright_next.Ps.to_vector()) {
        if (!is_in_group(valid_Ps, P)) {
            return false;
        }
    }
    return true;
}

void State::add(const Pauli& P, int index) {
    this->stab = this->stab.add(P, true, index);
    int qlast = P.end() - 1;
    for (int tmp_qlast = qlast; tmp_qlast < std::min(qlast + this->k, this->n); tmp_qlast++) {
        std::vector<int> left_terms;
        for (int tmp_index : this->valid_pauli_indices[tmp_qlast]) {
            auto [tmp_pauli, tmp_weight] = this->hamiltonian->paulis.at(tmp_qlast).at(tmp_index);
            if (this->stab.commute(tmp_pauli)) {
                left_terms.push_back(tmp_index);
            }
        }
        this->valid_pauli_indices[tmp_qlast] = left_terms;
    }
}

std::optional<State> State::move_to_next() const {
    State result = this->copy();
    Stab stab_tmp(result.stab.Ps);
    for (auto& Pright : result.Sright.Ps.to_vector()) {
        assert(stab_tmp.commute(Pright));
        if (!stab_tmp.include(Pright)) {
            stab_tmp = stab_tmp.add(Pright);
        }
    }
    for (int i = 0; i < result.hamiltonian->paulis.at(result.m).size(); i++) {
        auto [P, w] = result.hamiltonian->paulis.at(result.m).at(i);
        int index = i * result.n + result.m;
        if (!stab_tmp.include(P)) {
            continue;
        }
        if (!result.Sright.include(P)) {
            return std::nullopt;
        }
        assert(result.stab.commute(P));
        if (!result.stab.include(P)) {
            result.add(P, index);
        }
    }
    for (int i = 0; i < result.hamiltonian->paulis.at(result.m).size(); i++) {
        auto [P, w] = result.hamiltonian->paulis.at(result.m).at(i);
        if (result.stab.include(P)) {
            result.stab.add_energy(P, w);
        }
    }
    result.stab = result.stab.range(std::max(this->m - this->k + 1, 0), this->m + 1);
    return result;
}

State State::merge_gs(const State& state1, const State& state2) {
    State state = state1.copy();
    state.stab = StabEnergies::merge_gs(state1.stab, state2.stab);
    return state;
}

Stab State::get_stab(int arg) const {
    int nPs = this->stab.nPs(arg);
    Eigen::VectorXi qlast_list = mod(this->stab.Ps_indices.row(arg).array(), this->n);
    Eigen::VectorXi kindex_list = this->stab.Ps_indices.row(arg).array() / this->n;
    Eigen::VectorXi signs = this->stab.Ps_signs.row(arg)(Eigen::seqN(0, nPs));
    std::vector<Pauli> Ps;
    for (int i = 0; i < nPs; ++i) {
        int qlast = qlast_list(i);
        int kindex = kindex_list(i);
        Pauli P = (*this->hamiltonian).paulis[qlast][kindex].first;
        Ps.push_back(P);
    }
    Pauli Ps_all = Pauli::concatenate(Ps);
    Stab stab(Ps_all * signs);
    stab.canonicalize();
    return stab;
}

Stab State::get_stab_gs() const {
    int arg = argmin(this->stab.Es);
    return this->get_stab(arg);
}


StateMachine::StateMachine(Hamiltonian& hamiltonian, int nstates) {
    this->n = hamiltonian.n;
    this->k = hamiltonian.k;
    this->m = 0;
    this->nstates = nstates;
    this->hamiltonian = &hamiltonian;
    this->Sright_all = generate_Sright_all(hamiltonian);
    std::vector<Stab> Srights_zero = std::get<0>(Sright_all.at(0));
    for (auto& Sright : Srights_zero) {
        State state(hamiltonian, Sright);
        this->state_dict[state.hash_value()] = state;
        //this->state_list.push_back(state);
    }
}

std::vector<State> StateMachine::evolve_single(const State& raw_state) {
    std::optional<State> state_opt = raw_state.move_to_next();
    if (!state_opt.has_value()) {
        return std::vector<State>();
    }
    State state = state_opt.value();
    state.stab = state.stab.project_to(this->k - 1);
    std::vector<State> new_states;
    auto Sright_next_ids_map = std::get<1>(this->Sright_all.at(state.m));
    auto Sright_next_all = std::get<0>(this->Sright_all.at(state.m + 1));
    for (int index : Sright_next_ids_map.at(state.Sright.hash_value())) {
        Stab Sright_next = Sright_next_all.at(index);
        State new_state = state.copy();
        if (!new_state.check_Sright_validity(Sright_next)) {
            continue;
        }
        new_state.Sright = Sright_next;
        new_state.m += 1;
        new_states.push_back(new_state);
    }
    return new_states;
}

void StateMachine::evolve() {
    int count = 0;
    std::map<size_t, State> new_state_dict;
    int progress = 0;
    double t_evolve = 0.0;
    double t_update = 0.0;
    // #pragma omp parallel for
    std::vector<State> state_vector;
    for (const auto& item : state_dict) {
        state_vector.push_back(item.second); // item.second is the value part of the key-value pair
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < state_vector.size(); i++) {
    //for (auto& item : this->state_dict) {
        State state = state_vector[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<State> new_states = this->evolve_single(state);
        auto t2 = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            for (auto& new_state : new_states) {
                size_t hash_value = new_state.hash_value();
                if (new_state_dict.find(hash_value) != new_state_dict.end()) {
                    new_state_dict[hash_value] = State::merge_gs(new_state_dict[hash_value], new_state);
                } else {
                    new_state_dict[hash_value] = new_state;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            t_evolve += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1e-6;
            t_update += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() * 1e-6;
            std::cerr << "progress: " << ++progress << " in " << this->state_dict.size() << "\r" << std::flush;
            count++;
        }
    }
	std::cerr << std::endl;
    std::cout << "evolve time " << t_evolve << std::endl;
    std::cout << "update time " << t_update << std::endl;
    this->state_dict = new_state_dict;
    /*
    std::vector<State> new_state_list;
    for (auto& state : this->state_list) {
        std::vector<State> new_states = this->evolve_single(state);
        for (auto& new_state : new_states) {
            size_t hash_value = new_state.hash_value();
            new_state_list.push_back(new_state);
        }
    }
    */
    this->m += 1;
}

State StateMachine::generate_gs() {
    std::vector<State> state_list;
    for (auto& item : this->state_dict) {
        state_list.push_back(item.second);
    }
    int index_state = 0;
    double min_Es = std::numeric_limits<double>::max();
    for (int i = 0; i < state_list.size(); i++) {
        double min_E = state_list[i].stab.Es.minCoeff();
        if (min_E < min_Es) {
            min_Es = min_E;
            index_state = i;
        }
    }
    return state_list[index_state];
}

FullState::FullState() {
}

FullState::FullState(const std::vector<std::pair<Pauli, double>>& paulis) {
    // nmax = np.max([P.end() for P in paulis])
    int nmax = 0;
    for (auto& item : paulis) {
        nmax = std::max(nmax, item.first.end());
    }
    this->stab = StabEnergies::empty(0, 0, nmax);
    this->paulis = paulis;
    this->excluded_paulis = std::vector<Pauli>();
}

FullState FullState::copy() const {
    FullState result;
    result.stab = this->stab.copy();
    for (auto& item : this->paulis) {
        result.paulis.push_back(std::make_pair(item.first.copy(), item.second));
    }
    for (auto& P : this->excluded_paulis) {
        result.excluded_paulis.push_back(P.copy());
    }
    return result;
}

FullState FullState::include_pauli(const Pauli& P) const {
    FullState self = this->copy();
    self.stab = self.stab.add(P, true);
    std::vector<std::pair<Pauli, double>> new_paulis;
    for (auto& item : self.paulis) {
        Pauli Q = item.first;
        double w = item.second;
        if (!self.stab.commute(Q)) {
            continue;
        }
        if (self.stab.include(Q)) {
            self.stab.add_energy(Q, w);
        } else {
            StabEnergies tmp_stab = self.stab.add(Q, false);
            bool flag = false;
            for (auto& R : this->excluded_paulis) {
                if (tmp_stab.include(R)) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                new_paulis.push_back(std::make_pair(Q, w));
            }
        }
    }
    self.paulis = new_paulis;
    return self;
}

FullState FullState::exclude_pauli(const Pauli& P) const {
    FullState self = this->copy();
    self.excluded_paulis.push_back(P);
    std::vector<std::pair<Pauli, double>> new_paulis;
    for (auto& item : self.paulis) {
        Pauli Q = item.first;
        double w = item.second;
        if (!self.stab.add(Q, true).include(P)) {
            new_paulis.push_back(std::make_pair(Q, w));
        }
    }
    self.paulis = new_paulis;
    return self;
}

std::string FullState::to_string() const {
    std::string string1 = "stab = " + this->stab.to_string() + "\n";
    std::string string2 = "excluded_paulis = " + Pauli::concatenate(this->excluded_paulis).to_string() + "\n";
    std::vector<Pauli> paulis;
    std::transform(this->paulis.begin(), this->paulis.end(), std::back_inserter(paulis), [](const std::pair<Pauli, double>& item) {
        return item.first;
    });
    std::string string3 = "valid paulis = " + Pauli::concatenate(paulis).to_string() + "\n";
    std::stringstream string4;
    string4 << "energies " << this->stab.Es.transpose() << std::endl;
    return string1 + string2 + string3 + string4.str();
}

std::tuple<Stab, double> FullState::evolve() {
    FullState state = this->copy();
    std::vector<FullState> states = {state};
    int progress = 0;
    while (true) {
        bool finished = true;
        for (auto& state : states) {
            if (state.paulis.size() != 0) {
                finished = false;
                break;
            }
        }
        if (finished) {
            break;
        }
        std::vector<FullState> new_states;
        for (auto& state : states) {
            if (state.paulis.size() == 0) {
                new_states.push_back(state);
                continue;
            }
            Pauli P = state.paulis[0].first;
            new_states.push_back(state.include_pauli(P));
            new_states.push_back(state.exclude_pauli(P));
        }
        states = new_states;
    }
    std::map<size_t, FullState> state_dict;
    for (auto& state : states) {
        state_dict[state.stab.hash_value()] = state;
    }
    states = {};
    for (auto& item : state_dict) {
        states.push_back(item.second);
    }
    int index_state = 0;
    double min_Es = std::numeric_limits<double>::max();
    for (int i = 0; i < states.size(); i++) {
        double min_E = states[i].stab.Es.minCoeff();
        if (min_E < min_Es) {
            min_Es = min_E;
            index_state = i;
        }
    }
    FullState state_gs = states[index_state];
    int unraveled_index_sign = argmin(state_gs.stab.Es);
    Eigen::VectorXi index_sign = 1 - unravel_index(unraveled_index_sign, state_gs.stab.Ps.size).array() * 2;
    double energy_gs = state_gs.stab.Es(unraveled_index_sign);
    Stab stab(state_gs.stab.Ps * index_sign);
    stab.canonicalize();
    return std::make_tuple(stab, energy_gs);
}
