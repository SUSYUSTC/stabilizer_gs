#include <boost/functional/hash.hpp>
#include "stab.hpp"

// Pauli
// Pauli
// Pauli


Eigen::MatrixXi randomint2(int size, int length) {
    Eigen::MatrixXi result(size, length);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < length; j++) {
            result(i, j) = rand() % 2;
        }
    }
    return result;
}

size_t hash_matrix(const Eigen::MatrixXi& M) {
    size_t seed = 0;
    boost::hash_combine(seed, size_t(M.rows()));
    boost::hash_combine(seed, size_t(M.cols()));
    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.cols(); j++) {
            boost::hash_combine(seed, M(i, j));
        }
    }
    return seed;
}

Eigen::VectorXi reverse_order(const Eigen::VectorXi& x) {
    Eigen::VectorXi result = Eigen::VectorXi::Zero(x.size());
    result(x) = arange(0, x.size());
    return result;
}

Eigen::MatrixXi get_indices(int m) {
    Eigen::MatrixXi result = Eigen::MatrixXi::Zero(pow(2, m), m);
    for (int i = 0; i < pow(2, m); i++) {
        for (int j = 0; j < m; j++) {
            result(i, j) = (i >> j) % 2;
        }
    }
    return result(Eigen::all, arange(0, m).reverse());
}

template <typename Scalar>
Eigen::VectorX<Scalar> move_element(const Eigen::VectorX<Scalar>& x, int index, int new_index) {
    Eigen::VectorX<Scalar> result = x;
    if (index >= new_index) {
        result.segment(new_index + 1, index - new_index) = x.segment(new_index, index - new_index);
    } else {
        result.segment(index, new_index - index) = x.segment(index + 1, new_index - index);
    }
    result(new_index) = x(index);
    return result;
}

Eigen::VectorXi get_weights(int m) {
    Eigen::VectorXi result = Eigen::VectorXi::Zero(m);
    for (int i = 0; i < m; i++) {
        result(i) = pow(2, m - 1 - i);
    }
    return result;

}

Eigen::VectorXi get_reshape_indices(int m, int axis, bool backward) {
    assert(axis >= 0 && axis < m);
    Eigen::MatrixXi indices = get_indices(m);
    Eigen::VectorXi reorder_index = move_element(arange(0, m), axis, 0);
    Eigen::MatrixXi indices_reordered = indices(Eigen::all, reorder_index);
    Eigen::VectorXi newindices = indices_reordered * get_weights(m);
    if (!backward) {
        newindices = reverse_order(newindices);
    }
    return newindices;
}

void Pauli::copy_initializer(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin, const Eigen::VectorXi& phase) {
    assert(z.rows() == x.rows());
    assert(z.cols() == x.cols());
    assert(phase.size() == z.rows());
    this->begin = begin;
    this->length = z.cols();
    this->size = z.rows();
    this->z = z;
    this->x = x;
    this->set_group_phase(-phase.array() + 1);
}

Pauli::Pauli() {
}

Pauli::Pauli(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin, const Eigen::VectorXi& phase) {
    this->copy_initializer(z, x, begin, phase);
}

Pauli::Pauli(const Eigen::MatrixXi& z, const Eigen::MatrixXi& x, int begin) : Pauli(z, x, begin, Eigen::VectorXi::Ones(z.rows())) {}

Pauli::Pauli(const Pauli& other) {
    this->copy_initializer(other.z, other.x, other.begin, other.phase());
}

Pauli Pauli::copy() const {
    return Pauli(this->z, this->x, this->begin, this->phase());
}

Pauli& Pauli::operator=(const Pauli& other) {
    if (this != &other) {
        this->copy_initializer(other.z, other.x, other.begin, other.phase());
    }
    return *this;
}

Pauli::Pauli(const std::string& str, const std::vector<int>& qubits) {
    // example: str = "+XYIZ", qubits = [0, 2, 3, 4]
    assert(str.size() == qubits.size());
    // minimum of qubits
    int begin = *std::min_element(qubits.begin(), qubits.end());
    int end = *std::max_element(qubits.begin(), qubits.end()) + 1;
    int length = end - begin;
    int size = qubits.size();
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(1, length);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(1, length);
    for (int i = 0; i < size; i++) {
        char c = str[i];
        int pos = qubits[i] - begin;
        if (c == 'I') {
            continue;
        } else if (c == 'X') {
            x(0, pos) = 1;
        } else if (c == 'Z') {
            z(0, pos) = 1;
        } else if (c == 'Y') {
            x(0, pos) = 1;
            z(0, pos) = 1;
        } else {
            throw std::invalid_argument("invalid character");
        }
    }
    Eigen::VectorXi phase = Eigen::VectorXi::Ones(1);
    this->copy_initializer(z, x, begin, phase);
}

Pauli Pauli::identity(int length, int size, int begin) {
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(size, length);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(size, length);
    return Pauli(z, x, begin);
}

Pauli Pauli::random(int length, int size, int begin) {
    Eigen::MatrixXi z = randomint2(size, length);
    Eigen::MatrixXi x = randomint2(size, length);
    Eigen::VectorXi phase = -randomint2(size, 1).array() * 2 + 1;
    return Pauli(z, x, begin, phase);
}

Pauli Pauli::from_array(const Eigen::MatrixXi& array, int begin, const Eigen::VectorXi& phase) {
    int nqubits = array.cols() / 2;
    Eigen::MatrixXi z = array(Eigen::all, Eigen::seqN(0, nqubits, 2));
    Eigen::MatrixXi x = array(Eigen::all, Eigen::seqN(1, nqubits, 2));
    return Pauli(z, x, begin, phase);
}

Pauli Pauli::from_array(const Eigen::MatrixXi& array, int begin) {
    int nqubits = array.cols() / 2;
    Eigen::MatrixXi z = array(Eigen::all, Eigen::seqN(0, nqubits, 2));
    Eigen::MatrixXi x = array(Eigen::all, Eigen::seqN(1, nqubits, 2));
    return Pauli(z, x, begin);
}

int Pauli::end() const {
    return this->begin + this->length;
}
Eigen::VectorXi Pauli::nY() const {
    return (this->z.array() * this->x.array()).rowwise().sum();
}

Eigen::VectorXi Pauli::group_phase() const {
    return mod((this->ZX_phase + this->nY()).array(), 4);
}

void Pauli::set_group_phase(Eigen::VectorXi value) {
    this->ZX_phase = mod((value - this->nY()).array(), 4);
}

Eigen::VectorXi Pauli::phase() const {
    Eigen::VectorXi p = -this->group_phase().array() + 1;
    assert((p.array().abs() == 1).all());
    return p;
}

Eigen::MatrixXi Pauli::array() const {
    Eigen::MatrixXi result(this->size, this->length * 2);
    result(Eigen::all, Eigen::seqN(0, this->length, 2)) = this->z;
    result(Eigen::all, Eigen::seqN(1, this->length, 2)) = this->x;
    return result;
}

Eigen::MatrixXi Pauli::array4() const {
    return this->z * 2 + this->x;
}

Pauli Pauli::operator*(int c) const {
    Pauli result = this->copy();
    assert(abs(c) == 1);
    int add_phase = 1 - c;
    Eigen::VectorXi new_group_phase = mod((result.group_phase().array() + add_phase).array(), 4);
    result.set_group_phase(new_group_phase);
    return result;
}

Pauli Pauli::operator*(Eigen::VectorXi c) const {
    Pauli result = this->copy();
    assert((c.array().abs() == 1).all());
    Eigen::VectorXi add_phase = -c.array() + 1;
    Eigen::VectorXi new_group_phase = mod((result.group_phase().array() + add_phase.array()).array(), 4);
    result.set_group_phase(new_group_phase);
    return result;
}

Pauli Pauli::operator[](int key) const {
    return Pauli(this->z.row(key), this->x.row(key), this->begin, this->phase().row(key));
}

Pauli Pauli::operator[](const Eigen::VectorXi& indices) const {
    return Pauli(this->z(indices, Eigen::all), this->x(indices, Eigen::all), this->begin, this->phase()(indices));
}

void Pauli::set_item(int key, Pauli value) {
    assert(value.size == 1);
    this->z.row(key) = value.z;
    this->x.row(key) = value.x;
    this->ZX_phase(key) = get_item<int>(value.ZX_phase);
}

Pauli Pauli::range(int begin, int end, bool allow_truncation) const {
    if (!allow_truncation) {
        assert(begin <= this->begin);
        assert(end >= this->end());
    }
    int length_begin = std::max(this->begin - begin, 0);
    int length_end = std::max(end - this->end(), 0);
    int start = std::max(begin - this->begin, 0);
    int stop = std::min(this->end() - this->begin, end - this->begin);
    Eigen::MatrixXi zeros_begin(this->size, length_begin);
    Eigen::MatrixXi zeros_end(this->size, length_end);
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(this->size, end - begin);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(this->size, end - begin);
    z.block(0, length_begin, this->size, stop - start) = this->z.block(0, start, this->size, stop - start);
    x.block(0, length_begin, this->size, stop - start) = this->x.block(0, start, this->size, stop - start);
    return Pauli(z, x, begin, this->phase());
}

Pauli Pauli::simplify() const {
    Eigen::MatrixXi array4 = this->array4();
    int begin = this->begin;
    int end = this->end();
    Eigen::VectorXb allzero = (array4.colwise().sum()).array() == 0;
    while (allzero(0)) {
        begin += 1;
        allzero = allzero(Eigen::seqN(1, allzero.size() - 1)).eval();
    }
    while (allzero(allzero.size() - 1)) {
        end -= 1;
        allzero = allzero(Eigen::seqN(0, allzero.size() - 1)).eval();
    }
    Pauli P = this->range(begin, end, true);
    return P;
}

std::pair<Pauli, Pauli> Pauli::extend(const Pauli& first, const Pauli& second) {
    int begin = std::min(first.begin, second.begin);
    int end = std::max(first.end(), second.end());
    return {first.range(begin, end), second.range(begin, end)};
}

std::vector<Pauli> Pauli::extend(std::vector<Pauli> paulis) {
    int begin = std::min_element(paulis.begin(), paulis.end(), [](const Pauli& a, const Pauli& b) { return a.begin < b.begin; })->begin;
    int end = std::max_element(paulis.begin(), paulis.end(), [](const Pauli& a, const Pauli& b) { return a.end() < b.end(); })->end();
    auto result = std::vector<Pauli>();
    for (auto& pauli : paulis) {
        result.push_back(pauli.range(begin, end));
    }
    return result;
}

Pauli Pauli::concatenate(const std::vector<Pauli>& paulis) {
    if (paulis.size() == 0) {
        return Pauli::identity(0, 0, 0);
    }
    auto paulis_extended = Pauli::extend(paulis);
    // total_size = sum([p.size for p in paulis_extended])
    int total_size = 0;
    for (auto& pauli : paulis_extended) {
        total_size += pauli.size;
    }
    int begin = paulis_extended[0].begin;
    int length = paulis_extended[0].length;
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(total_size, length);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(total_size, length);
    Eigen::VectorXi phase = Eigen::VectorXi::Zero(total_size);
    int index = 0;
    for (auto& pauli : paulis_extended) {
        z.block(index, 0, pauli.size, length) = pauli.z;
        x.block(index, 0, pauli.size, length) = pauli.x;
        phase.segment(index, pauli.size) = pauli.phase();
        index += pauli.size;
    }
    return Pauli(z, x, begin, phase);
}

Eigen::VectorXb Pauli::commute(const Pauli& second_input) const {
    auto [first, second] = Pauli::extend(*this, second_input);
    Eigen::VectorXi count = (first.z.array() * second.x.array()).rowwise().sum() + (second.z.array() * first.x.array()).rowwise().sum();
    return mod(count.array(), 2) == 0;
}

Pauli Pauli::dot(const Pauli& second_input) const {
    assert(this->commute(second_input).all());
    auto [first, second] = Pauli::extend(*this, second_input);
    Eigen::MatrixXi z = mod((first.z + second.z).array(), 2);
    Eigen::MatrixXi x = mod((first.x + second.x).array(), 2);
    Eigen::VectorXi add_phase = (first.x.array() * second.z.array()).rowwise().sum() * 2;
    Eigen::VectorXi ZX_phase = mod((first.ZX_phase + second.ZX_phase + add_phase).array(), 4);
    Pauli result = Pauli(z, x, first.begin);
    result.ZX_phase = ZX_phase;
    return result;
}

std::vector<Pauli> Pauli::to_vector() const {
    std::vector<Pauli> result;
    for (int i = 0; i < this->size; i++) {
        result.push_back((*this)[i]);
    }
    return result;
}

Pauli Pauli::reduce_dot() const {
    std::vector<Pauli> vec = this->to_vector();
    auto dot = [](Pauli first, Pauli second) {return first.dot(second);};
    auto I = Pauli::identity(this->length, 1, this->begin);
    return std::reduce(vec.begin(), vec.end(), I, dot);
}

std::string Pauli::to_string(bool add_header) const {
    std::string result;
    if (add_header) {
        result = "qubit " + std::to_string(this->begin) + "-" + std::to_string(this->end() - 1) + " size " + std::to_string(this->size) + " ";
    }
    for (int i = 0; i < this->size; i++) {
        // phase = 1: "+", phase = -1: "-"
        // for each element in array4, 0: "I", 1: "X", 2: "Z", 3: "Y"
        // example: "+XZIY"
        std::string pauli_string = this->phase()[i] == 1 ? "+" : "-";
        auto array4 = this->array4();
        for (int j = 0; j < this->length; j++) {
            pauli_string += std::string("IXZY")[array4(i, j)];
        }
        result += pauli_string + " ";
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Pauli& pauli) {
    os << pauli.to_string();
    return os;
}

// implement hash
size_t Pauli::hash_value(bool add_position) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, hash_matrix(this->z));
    boost::hash_combine(seed, hash_matrix(this->x));
    if (add_position) {
        boost::hash_combine(seed, this->begin);
    }
    boost::hash_combine(seed, hash_matrix(this->group_phase()));
    return seed;
}


// StabilityGroup
// StabilityGroup
// StabilityGroup

void Stab::initialize() {
    this->m = this->Ps.array().rows();
    this->dim = this->Ps.array().cols();
    this->nqubits = this->dim / 2;
}

void Stab::copy_initializer(const Pauli& Ps) {
    this->Ps = Ps;
    this->initialize();
}

Stab::Stab() {
}

Stab::Stab(const Pauli& Ps, bool canonicalized) {
    this->copy_initializer(Ps);
    if (!canonicalized) {
        this->canonicalize();
    }
}

Stab::Stab(const Stab& other) {
    this->copy_initializer(other.Ps);
}

Stab& Stab::operator=(const Stab& other) {
    if (this != &other) {
        this->copy_initializer(other.Ps);
    }
    return *this;
}

int Stab::begin() const {
    return this->Ps.begin;
}

int Stab::end() const {
    return this->Ps.end();
}

Stab Stab::copy() const {
    return Stab(this->Ps.copy(), true);
}

Stab Stab::add(const Pauli& P) const {
    Pauli Ps = Pauli::concatenate({this->Ps, P});
    return Stab(Ps);
}

void Stab::canonicalize() {
    auto [tmp, U] = linear::Gauss_elimination_full(this->Ps.array(), 2, true);
    this->transform(U);
}

void Stab::transform(const Eigen::MatrixXi& U) {
    Pauli new_Ps = this->Ps.copy();
    for (int i = 0; i < this->m; i++) {
        Pauli new_P = this->Ps[get_where(U.row(i).array() == 1)].reduce_dot();
        new_Ps.set_item(i, new_P);
    }
    this->Ps = new_Ps;
}

Stab Stab::empty(int nqubits, int begin) {
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(0, nqubits);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(0, nqubits);
    Pauli Ps = Pauli(z, x, begin);
    return Stab(Ps);
}

Stab Stab::random_stab(int nqubits, int m) {
    assert(nqubits >= m);
    Eigen::MatrixXi array = Eigen::MatrixXi::Zero(0, nqubits * 2);
    Pauli Ps = Pauli::from_array(array, 0);
    for (int i = 0; i < m; i++) {
        Pauli Ps_reverse = Pauli(Ps.x, Ps.z, 0);
        Eigen::MatrixXi full = linear::Gauss_elimination_Aonly(Ps_reverse.array(), 2, true);
        assert((full.rowwise().sum().array() > 0).all());
        Eigen::MatrixXi kernel = linear::get_kernel_reordered(full, 2);
        auto create_random = [&kernel]() {
            return (Eigen::VectorXi::Random(kernel.cols()).array() >= 0).cast<int>();
        };
        Pauli P = Pauli::identity(nqubits, 1, 0);
        while (true) {
            Eigen::VectorXi x = create_random();
            Eigen::VectorXi phase = -randomint2(1, 1).array() * 2 + 1;
            Eigen::MatrixXi array_P = mod((kernel*x).array(), 2).transpose();
            Pauli tmp_P = Pauli::from_array(array_P, 0, phase.transpose());
            bool included = Stab(Ps).include(tmp_P);
            if (!included) {
                P = tmp_P;
                break;
            }
        }
        Ps = Pauli::concatenate({Ps, P});
        array = Ps.array();
    }
    return Stab(Ps);
}

Stab Stab::range(int begin, int end, bool allow_truncation) const {
    Pauli Ps_new_range = this->Ps.range(begin, end, allow_truncation);
    return Stab(Ps_new_range, !allow_truncation);
}

std::tuple<bool, Eigen::VectorXi, int> Stab::decompose(const Pauli& input_P) const {
    assert(input_P.size == 1);
    auto [Ps, P] = Pauli::extend(this->Ps, input_P);
    Eigen::MatrixXi A = Ps.array().transpose();
    Eigen::MatrixXi b = P.array().transpose();
    auto [is_over, is_under, x] = linear::solve_modN(A, b, 2);
    assert(!is_under);
    Eigen::VectorXi nonzero = get_where(x.array() > 0);
    Pauli product = Ps[nonzero].reduce_dot();
    //assert((product.array() == P.array()).all());
    int phase = get_item<int>(P.phase() * product.phase());
    return std::make_tuple(get_item<bool>(is_over), x, phase);
}

bool Stab::include(const Pauli& P) const {
    assert(P.size == 1);
    auto [is_over, is_under, x] = this->decompose(P);
    return !is_over;
}

bool Stab::commute(const Pauli& Q) const {
    assert(Q.size == 1);
    auto vec = Ps.to_vector();
    for (auto &P : vec) {
        if (!get_item<bool>(P.commute(Q))) {
            return false;
        }
    }
    return true;
}

Stab Stab::project(int begin, int end) const {
    /*
    converty the following code to C++

    assert begin >= self.Ps.begin
    assert end <= self.Ps.end
    Ps = self.Ps
    n_left = begin - Ps.begin
    n_middle = end - begin
    n_right = Ps.end - end
    assert len(Ps.batch_size) == 1

    x = Ps.x
    z = Ps.z
    x_out = np.concatenate([x[:, :n_left], x[:, n_left + n_middle:]], axis=-1)
    z_out = np.concatenate([z[:, :n_left], z[:, n_left + n_middle:]], axis=-1)
    array_out = Pauli(z_out, x_out).array.T

    kernel = linear.get_kernel(array_out, 2)
    if kernel.shape[1] == 0:
        return StabilizerGroup.empty(n_middle, begin);
    new_Ps = Pauli.stack([Ps[include > 0].reduce_dot() for include in kernel.T])
    return StabilizerGroup(new_Ps.range(begin, end, allow_truncation=True))
    */
    assert(begin >= this->begin());
    assert(end <= this->end());
    Pauli Ps = this->Ps;
    int n_left = begin - Ps.begin;
    int n_middle = end - begin;
    int n_right = Ps.end() - end;
    Eigen::MatrixXi x = Ps.x;
    Eigen::MatrixXi z = Ps.z;
    Eigen::MatrixXi x_out = Eigen::MatrixXi::Zero(Ps.size, n_left + n_right);
    Eigen::MatrixXi z_out = Eigen::MatrixXi::Zero(Ps.size, n_left + n_right);
    x_out.block(0, 0, Ps.size, n_left) = x.block(0, 0, Ps.size, n_left);
    x_out.block(0, n_left, Ps.size, n_right) = x.block(0, n_left + n_middle, Ps.size, n_right);
    z_out.block(0, 0, Ps.size, n_left) = z.block(0, 0, Ps.size, n_left);
    z_out.block(0, n_left, Ps.size, n_right) = z.block(0, n_left + n_middle, Ps.size, n_right);
    Eigen::MatrixXi array_out = Pauli(z_out, x_out, 0).array().transpose();

    Eigen::MatrixXi kernel = linear::get_kernel(array_out, 2);
    if (kernel.cols() == 0) {
        return Stab::empty(n_middle, begin);
    }

    std::vector<Pauli> vec;
    for (int i = 0; i < kernel.cols(); i++) {
        Eigen::VectorXi include = kernel.col(i);
        Pauli P = Ps[get_where(include.array() > 0)].reduce_dot();
        vec.push_back(P);
    }
    Pauli new_Ps = Pauli::concatenate(vec);
    return Stab(new_Ps.range(begin, end, true));
}

int Stab::energy(const Pauli& Q) const {
    assert(Q.size == 1);
    auto [is_over, x, phase] = this->decompose(Q);
    if (is_over) {
        return 0;
    } else {
        return phase;
    }
}

std::string Stab::to_string(bool align) const {
    if (align) {
        std::string result = "qubit " + std::to_string(this->begin()) + "-" + std::to_string(this->end() - 1) + " size " + std::to_string(this->Ps.size) + "\n";
        for (int i = 0; i < this->Ps.size; i++) {
            result += this->Ps[i].to_string(false) + "\n";
        }
        return result;
    } else {
        return this->Ps.to_string();
    }
}

size_t Stab::hash_value(bool add_position) const {
    return this->Ps.hash_value(add_position);
}

// StabilizerGroupEnergies
// StabilizerGroupEnergies
// StabilizerGroupEnergies

void StabEnergies::copy_initializer(const Pauli& Ps, const Eigen::VectorXd& Es, const Eigen::VectorXi& nPs, const Eigen::MatrixXi& Ps_indices, const Eigen::MatrixXi& Ps_signs) {
    this->Ps = Ps;
    this->Es = Es;
    this->nPs = nPs;
    this->Ps_indices = Ps_indices;
    this->Ps_signs = Ps_signs;
    this->initialize();
}

StabEnergies::StabEnergies() {
}

StabEnergies::StabEnergies(const Pauli& Ps, const Eigen::VectorXd& Es, const Eigen::VectorXi& nPs, const Eigen::MatrixXi& Ps_indices, const Eigen::MatrixXi& Ps_signs, bool canonicalized){
    this->copy_initializer(Ps, Es, nPs, Ps_indices, Ps_signs);
    if (!canonicalized) {
        this->canonicalize();
    }
    assert((this->Ps.phase().array() == 1).all());
}

StabEnergies::StabEnergies(const StabEnergies& other) {
    this->copy_initializer(other.Ps, other.Es, other.nPs, other.Ps_indices, other.Ps_signs);
}

StabEnergies StabEnergies::copy() const {
    return StabEnergies(this->Ps.copy(), this->Es, this->nPs, this->Ps_indices, this->Ps_signs, true);
}

StabEnergies& StabEnergies::operator=(const StabEnergies& other) {
    if (this != &other) {
        this->copy_initializer(other.Ps, other.Es, other.nPs, other.Ps_indices, other.Ps_signs);
    }
    return *this;
}

StabEnergies StabEnergies::empty(int nqubits, int begin, int nmax) {
    Eigen::MatrixXi z = Eigen::MatrixXi::Zero(0, nqubits);
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(0, nqubits);
    Pauli Ps = Pauli(z, x, begin);
    Eigen::VectorXd Es = Eigen::VectorXd::Zero(1);
    Eigen::VectorXi nPs = Eigen::VectorXi::Zero(1);
    Eigen::MatrixXi Ps_indices = Eigen::MatrixXi::Zero(1, nmax);
    Eigen::MatrixXi Ps_signs = Eigen::MatrixXi::Zero(1, nmax);
    return StabEnergies(Ps, Es, nPs, Ps_indices, Ps_signs);
}

StabEnergies StabEnergies::range(int begin, int end, bool allow_truncation) const {
    if (!allow_truncation) {
        if (begin > this->begin()) {
            assert((this->Ps.array4().block(0, 0, this->Ps.size, begin - this->begin()).array() == 0).all());
        }
        if (end < this->end()) {
            assert((this->Ps.array4().block(0, end - this->begin(), this->Ps.size, this->end() - end).array() = 0).all());
        }
    }
    return StabEnergies(this->Ps.range(begin, end, allow_truncation), this->Es, this->nPs, this->Ps_indices, this->Ps_signs, true);
}

void StabEnergies::flip_sign(int index) {
    this->Ps.ZX_phase[index] = mod(this->Ps.ZX_phase[index] + 2, 4);
    Eigen::VectorXi indices = get_reshape_indices(this->m, index, false);
    int sizem1 = pow(2, this->m - 1);
    Eigen::VectorXi new_indices = indices.reshaped<Eigen::RowMajor>(2, sizem1)
                                        (arange(0, 2).reverse(), Eigen::all)
                                        .reshaped<Eigen::RowMajor>().eval();
    /*
    this->Es(indices) = this->Es(indices).reshaped<Eigen::RowMajor>(2, sizem1)
                                            (arange(0, 2).reverse(), Eigen::all)
                                            .reshaped<Eigen::RowMajor>().eval();
    */
    this->Es(indices) = this->Es(new_indices).eval();
    this->nPs(indices) = this->nPs(new_indices).eval();
    this->Ps_indices(indices, Eigen::all) = this->Ps_indices(new_indices, Eigen::all).eval();
    this->Ps_signs(indices, Eigen::all) = this->Ps_signs(new_indices, Eigen::all).eval();
}

void StabEnergies::canonicalize_signs() {
    for (int i = 0; i < this->m; i++) {
        if (this->Ps.phase()[i] == -1) {
            this->flip_sign(i);
        }
    }
}

void StabEnergies::canonicalize() {
    auto [tmp, U] = linear::Gauss_elimination_full(this->Ps.array(), 2, true);
    this->transform(U);
    this->canonicalize_signs();
}

void StabEnergies::transform(const Eigen::MatrixXi& U) {
    /*
    Eigen::MatrixXi z = mod((U * this->Ps.z).array(), 2);
    Eigen::MatrixXi x = mod((U * this->Ps.x).array(), 2);
    Pauli Ps(z, x, this->Ps.begin);
    Eigen::VectorXi signs = Eigen::VectorXi::Zero(this->m);
    for (int i = 0; i < this->m; i++) {
        auto [is_over, x, sign] = this->decompose(Ps[i]);
        assert(!is_over);
        signs[i] = sign;
    }
    Ps = Ps * signs;
    assert((Ps.array4().array() == new_Ps.array4().array()).all());
    assert((Ps.phase().array() == new_Ps.phase().array()).all());
    */
    Pauli new_Ps = this->Ps.copy();
    for (int i = 0; i < this->m; i++) {
        Pauli new_P = this->Ps[get_where(U.row(i).array() == 1)].reduce_dot();
        new_Ps.set_item(i, new_P);
    }
    this->Ps = new_Ps;
    Eigen::VectorXd Es = Eigen::VectorXd::Zero(this->Es.rows(), this->Es.cols());
    Eigen::MatrixXi indices = get_indices(this->m);
    Eigen::MatrixXi indices_transform = mod((indices * U.transpose()).array(), 2);
    Eigen::VectorXi indices_transform_flatten = indices_transform * get_weights(this->m);
    Eigen::VectorXi indices_transform_flatten_reverse = reverse_order(indices_transform_flatten);
    /*
    Es(indices_transform_flatten) = this->Es;
    this->Es = Es;
    */
    this->Es = this->Es(indices_transform_flatten_reverse).eval();
    this->nPs = this->nPs(indices_transform_flatten_reverse).eval();
    this->Ps_indices = this->Ps_indices(indices_transform_flatten_reverse, Eigen::all).eval();
    this->Ps_signs = this->Ps_signs(indices_transform_flatten_reverse, Eigen::all).eval();
}

template <typename Scalar>
Eigen::MatrixX<Scalar> concat_last(const Eigen::MatrixX<Scalar>& M1, const Eigen::MatrixX<Scalar>& M2) {
    assert(M1.rows() == M2.rows());
    assert(M1.cols() == M2.cols());
    int rows = M1.rows();
    int cols = M1.cols();
    Eigen::MatrixX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(rows * 2, cols);
    result(Eigen::seqN(0, rows, 2), Eigen::all) = M1;
    result(Eigen::seqN(1, rows, 2), Eigen::all) = M2;
    return result;
};

StabEnergies StabEnergies::add(const Pauli& P, bool canonicalize, int index) const {
    Pauli Ps = Pauli::concatenate({this->Ps, P});
    int rows = Es.rows();
    Eigen::VectorXd Es = concat_last<double>(this->Es, this->Es);
    Eigen::VectorXi nPs = concat_last<int>(this->nPs, this->nPs);
    Eigen::MatrixXi Ps_indices_tmp = this->Ps_indices;
    set_matrix_values<int>(Ps_indices_tmp, arange(0, rows), this->nPs, Eigen::VectorXi::Constant(rows, index));
    Eigen::MatrixXi Ps_indices = concat_last<int>(Ps_indices_tmp, Ps_indices_tmp);
    Eigen::MatrixXi Ps_signs1 = this->Ps_signs;
    Eigen::MatrixXi Ps_signs2 = this->Ps_signs;
    set_matrix_values<int>(Ps_signs1, arange(0, rows), this->nPs, Eigen::VectorXi::Constant(rows, 1));
    set_matrix_values<int>(Ps_signs2, arange(0, rows), this->nPs, Eigen::VectorXi::Constant(rows, -1));
    Eigen::MatrixXi Ps_signs = concat_last<int>(Ps_signs1, Ps_signs2);
    return StabEnergies(Ps, Es, nPs.array() + 1, Ps_indices, Ps_signs, !canonicalize);
}

template <typename Scalar>
Eigen::MatrixX<Scalar> merge_first(const Eigen::MatrixX<Scalar>& M1, const Eigen::MatrixX<Scalar>& M2, const Eigen::MatrixXb& keep1) {
    Eigen::MatrixX<Scalar> result = M2;
    Eigen::VectorXi arg = get_where(keep1);
    result(arg, Eigen::all) = M1(arg, Eigen::all);
    return result;
}

template <typename Scalar>
Eigen::MatrixX<Scalar> merge_first(const Eigen::MatrixX<Scalar>& M, const Eigen::MatrixXb& keep1) {
    int rows = M.rows();
    int cols = M.cols();
    Eigen::MatrixX<Scalar> M1 = M(Eigen::seqN(0, rows/2), Eigen::all);
    Eigen::MatrixX<Scalar> M2 = M(Eigen::seqN(rows/2, rows/2), Eigen::all);
    return merge_first<Scalar>(M1, M2, keep1);
    //Eigen::MatrixX<Scalar> result = M1.array() * keep1.array().cast<Scalar>() + M2.array() * keep2.array().cast<Scalar>();
    //return result;
}

StabEnergies StabEnergies::project_to_next() const {
    StabEnergies result = this->copy();
    int sizem1 = pow(2, this->m - 1);
    while ((result.Ps.array4().col(0).array() > 0).any()) {
        result.Ps = result.Ps[arange(1, result.Ps.size)];
        Eigen::VectorXd Es1 = result.Es.segment(0, sizem1);
        Eigen::VectorXd Es2 = result.Es.segment(sizem1, sizem1);
        Eigen::VectorXb keep1 = Es1.array() < Es2.array();
        //result.Es = Es1.array() * keep1.array().cast<double>() + Es2.array() * keep2.array().cast<double>();
        result.Es = merge_first<double>(result.Es, keep1);
        result.nPs = merge_first<int>(result.nPs, keep1);
        result.Ps_indices = merge_first<int>(result.Ps_indices, keep1);
        result.Ps_signs = merge_first<int>(result.Ps_signs, keep1);
        //result.Es = result.Es.reshaped<Eigen::RowMajor>(2, sizem1).colwise().minCoeff().eval();
        sizem1 /= 2;
    }
    result.initialize();
    result = result.range(result.Ps.begin + 1, result.Ps.end(), true);
    return result;
}

StabEnergies StabEnergies::project_to(int k1) const {
   StabEnergies self = this->copy();
   for (int i = this->begin(); i < this->end() - k1; i++) {
       self = self.project_to_next();
   }
   return self;
}

Pauli StabEnergies::simplify_coset_representative(const Pauli& input_P) const {
    auto [Ps, P] = Pauli::extend(this->Ps, input_P);
    Eigen::MatrixXi A2 = Ps.array();
    Eigen::VectorXi x = P.array().transpose();
    Eigen::VectorXi array = linear::simplify_coset_representative(A2, x, 2).transpose();
    return Pauli::from_array(array.transpose(), P.begin);
}

void StabEnergies::add_energy(const Pauli& P, double w) {
    auto [is_over, x, sign] = this->decompose(P);
    assert(!is_over);
    Eigen::MatrixXi indices = get_indices(this->m);
    Eigen::VectorXi signs = (1 - mod((indices * x).array(), 2) * 2).array() * sign;
    this->Es = this->Es.array() + signs.array().cast<double>() * w;
}

StabEnergies StabEnergies::merge_gs(const StabEnergies& stab1, const StabEnergies& stab2) {
    assert((stab1.Ps.phase().array() == 1).all());
    assert((stab2.Ps.phase().array() == 1).all());
    assert((stab1.Ps.array4().array() == stab2.Ps.array4().array()).all());
    Pauli Ps = stab1.Ps;
    Eigen::VectorXb keep1 = stab1.Es.array() < stab2.Es.array();
    //Eigen::MatrixXd Es = stab1.Es.cwiseMin(stab2.Es);
    Eigen::VectorXd Es = merge_first<double>(stab1.Es, stab2.Es, keep1);
    Eigen::VectorXi nPs = merge_first<int>(stab1.nPs, stab2.nPs, keep1);
    Eigen::MatrixXi Ps_indices = merge_first<int>(stab1.Ps_indices, stab2.Ps_indices, keep1);
    Eigen::MatrixXi Ps_signs = merge_first<int>(stab1.Ps_signs, stab2.Ps_signs, keep1);
    //Eigen::VectorXi nPs = stab1.nPs;
    //Eigen::MatrixXi Ps_indices = stab1.Ps_indices;
    //Eigen::MatrixXi Ps_signs = stab1.Ps_signs;
    StabEnergies result = StabEnergies(Ps, Es, nPs, Ps_indices, Ps_signs);
    return result;
}

/*
convert to C++
def is_in_group(Ps: Pauli, P: Pauli):
    assert Ps.begin == P.begin
    assert Ps.end == P.end
    assert len(Ps.batch_size) == 1
    assert len(P.batch_size) == 0
    is_over, is_under, x = linear.solve_modN(Ps.array.T, P.array, 2)
    return (not is_over)
*/

bool is_in_group(const Pauli& Ps, const Pauli& P) {
    assert(Ps.begin == P.begin);
    assert(Ps.end() == P.end());
    assert(P.size == 1);
    auto [is_over, is_under, x] = linear::solve_modN(Ps.array().transpose(), P.array().transpose(), 2);
    return !get_item<bool>(is_over);
}