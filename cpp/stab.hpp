#ifndef STAB_HPP
#define STAB_HPP

#include <boost/functional/hash.hpp>
#include "linear.hpp"

extern bool energy_only;

Eigen::MatrixXi randomint2(int size, int length);

size_t hash_matrix(const Eigen::MatrixXi& M);

Eigen::VectorXi reverse_order(const Eigen::VectorXi& x);

Eigen::MatrixXi get_indices(int m);

template <typename Scalar>
Eigen::VectorX<Scalar> move_element(const Eigen::VectorX<Scalar>& x, int index, int new_index);

Eigen::VectorXi get_weights(int m);

Eigen::VectorXi get_reshape_indices(int m, int axis, bool backward);

class Pauli {
public:
    Eigen::MatrixXi z, x;
    int begin, length;
    int size;
    Eigen::VectorXi ZX_phase;

    void copy_initializer(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin, const Eigen::VectorXi& phase);

    Pauli();

    Pauli(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin, const Eigen::VectorXi& phase);

    Pauli(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin);

    Pauli(const Pauli& other);

    Pauli& operator=(const Pauli& other);

    Pauli copy() const;

    Pauli(const std::string& str, const std::vector<int>& qubits);

    static Pauli identity(int length, int size, int begin);

    static Pauli random(int length, int size, int begin);

    static Pauli from_array(const Eigen::MatrixXi& array, int begin, const Eigen::VectorXi& phase);

    static Pauli from_array(const Eigen::MatrixXi& array, int begin);

    int end() const;

    Eigen::VectorXi nY() const;

    Eigen::VectorXi group_phase() const;

    void set_group_phase(Eigen::VectorXi value);

    Eigen::VectorXi phase() const;

    Eigen::MatrixXi array() const;

    Eigen::MatrixXi array4() const;

    Pauli operator*(int c) const;

    Pauli operator*(Eigen::VectorXi c) const;

    Pauli operator[](int key) const;

    Pauli operator[](const Eigen::VectorXi& indices) const;

    void set_item(int key, Pauli value);

    Pauli range(int begin, int end, bool allow_truncation = false) const;

    Pauli simplify() const;

    static std::pair<Pauli, Pauli> extend(const Pauli& first, const Pauli& second);

    static std::vector<Pauli> extend(std::vector<Pauli> paulis);

    static Pauli concatenate(const std::vector<Pauli>& paulis);

    Eigen::VectorXb commute(const Pauli& second_input) const;

    Pauli dot(const Pauli& second_input) const;

    std::vector<Pauli> to_vector() const;

    Pauli reduce_dot() const;

    std::string to_string(bool add_header = true) const;

    friend std::ostream& operator<<(std::ostream& os, const Pauli& pauli);

    size_t hash_value(bool add_position = true) const;
};

class Stab {
public:
    Pauli Ps;
    int m, dim, nqubits;

    void copy_initializer(const Pauli& Ps);

    Stab();

    Stab(const Pauli& Ps, bool canonicalized = false);
    
    Stab(const Stab& other);

    Stab& operator=(const Stab& other);

    Stab copy() const;

    Stab add(const Pauli& P) const;

    void initialize();

    int begin() const;

    int end() const;

    void canonicalize();

    void transform(const Eigen::MatrixXi& U);

    static Stab empty(int nqubits, int begin);

    static Stab random_stab(int nqubits, int m);
    
    Stab range(int begin, int end, bool allow_truncation = false) const;

    std::tuple<bool, Eigen::VectorXi, int> decompose(const Pauli& input_P) const;

    bool include(const Pauli& P) const;
    
    bool commute(const Pauli& Q) const;

    Stab project(int begin, int end) const;

    int energy(const Pauli& Q) const;

    size_t hash_value(bool add_position = true) const;

    std::string to_string(bool align = false) const;
};

class StabEnergies : public Stab {
public:
    Eigen::VectorXd Es;
    Eigen::VectorXi nPs;
    Eigen::MatrixXi Ps_indices;
    Eigen::MatrixXi Ps_signs;

    void copy_initializer(const Pauli& Ps, const Eigen::VectorXd& Es, const Eigen::VectorXi& nPs, const Eigen::MatrixXi& Ps_indices, const Eigen::MatrixXi& Ps_signs);

    StabEnergies();

    StabEnergies(const Pauli& Ps, const Eigen::VectorXd& Es, const Eigen::VectorXi& nPs, const Eigen::MatrixXi& Ps_indices, const Eigen::MatrixXi& Ps_signs, bool canonicalized=false);

    StabEnergies(const StabEnergies& other);

    StabEnergies copy() const;

    StabEnergies& operator=(const StabEnergies&);

    static StabEnergies empty(int nqubits, int begin, int nmax);

    StabEnergies range(int begin, int end, bool allow_truncation = false) const;

    void flip_sign(int index);

    void canonicalize_signs();

    void canonicalize();

    void transform(const Eigen::MatrixXi& U);

    StabEnergies add(const Pauli& P, bool canonicalize, int index = -1) const;

    StabEnergies project_to_next() const;

    StabEnergies project_to(int k1) const;

    Pauli simplify_coset_representative(const Pauli& input_P) const;

    void add_energy(const Pauli& P, double w);

    static StabEnergies merge_gs(const StabEnergies& stab1, const StabEnergies& stab2);
};

bool is_in_group(const Pauli& Ps, const Pauli& P);

#endif
