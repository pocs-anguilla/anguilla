
//
// This source code is supplementary material of the research paper
// "Challenges of Convex Quadratic Bi-objective Benchmark Problems"
// by Tobias Glasmachers. It is provided under the MIT license:
//
// Copyright 2018 Tobias Glasmachers
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "moq/benchmarks.h"

#include <cmath>
#include <stdexcept>

using namespace shark;
using namespace remora;
using namespace std;

tuple<RealMatrix, RealVector> MOBenchmark::eig(RealMatrix const& U1, RealVector const& D1, RealMatrix const& U2, RealVector const& D2) {
    size_t n = D1.size();
    RealVector invsqrtD2(n);
    for (size_t i = 0; i < n; i++) invsqrtD2(i) = 1.0 / sqrt(D2(i));
    RealMatrix H1 = U1 % to_diagonal(D1) % trans(U1);
    RealMatrix invA2 = U2 % to_diagonal(invsqrtD2) % trans(U2);
    RealMatrix M = invA2 % H1 % invA2;

    RealMatrix X(n, n);
    RealVector lambda(n);
    symm_eigenvalue_decomposition<matrix<double>> solver(M);
    RealMatrix V = invA2 % solver.Q();
    return make_tuple(V, solver.D());
}

RealMatrix MOBenchmark::eye(unsigned int n) {
    RealMatrix ret(n, n, 0.0);
    for (unsigned int i = 0; i < n; i++) ret(i, i) = 1;
    return ret;
}

RealVector MOBenchmark::gauss(unsigned int n) {
    RealVector ret(n);
    normal_distribution<double> normal;
    for (unsigned int i = 0; i < n; i++) ret(i) = normal(m_rng);
    return ret;
}

RealMatrix MOBenchmark::gauss(unsigned int n, unsigned int m) {
    RealMatrix ret(n, m);
    normal_distribution<double> normal;
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < m; j++)
            ret(i, j) = normal(m_rng);
    return ret;
}

RealMatrix MOBenchmark::sampleU() {
    RealMatrix ret = gauss(m_dimension, m_dimension);
    for (unsigned int i = 0; i < m_dimension; i++) {
        for (unsigned int j = 0; j < i; j++) {
            column(ret, i) -= inner_prod(column(ret, i), column(ret, j)) * column(ret, j);
        }
        column(ret, i) /= norm_2(column(ret, i));
    }
    return ret;
}

RealMatrix MOBenchmark::sampleUfix(unsigned int u) {
    RealMatrix ret = gauss(m_dimension, m_dimension);
    for (unsigned int i = 0; i < m_dimension; i++) {
        ret(i, u) = 0;
        ret(u, i) = 0;
    }
    ret(u, u) = 1;
    for (unsigned int i = 0; i < m_dimension; i++) {
        for (unsigned int j = 0; j < i; j++) {
            column(ret, i) -= inner_prod(column(ret, i), column(ret, j)) * column(ret, j);
        }
        column(ret, i) /= norm_2(column(ret, i));
    }
    return ret;
}

RealMatrix MOBenchmark::sampleUTdelta(RealVector const& delta) {
    RealMatrix ret = gauss(m_dimension, m_dimension);
    column(ret, 0) = delta;
    for (unsigned int i = 0; i < m_dimension; i++) {
        for (unsigned int j = 0; j < i; j++) {
            column(ret, i) -= inner_prod(column(ret, i), column(ret, j)) * column(ret, j);
        }
        column(ret, i) /= norm_2(column(ret, i));
    }
    uniform_int_distribution<unsigned int> dist(0, m_dimension - 1);
    unsigned int i = dist(m_rng);
    if (i != 0) {
        RealVector tmp = column(ret, 0);
        column(ret, 0) = column(ret, i);
        column(ret, i) = tmp;
    }
    return trans(ret);
}

RealVector MOBenchmark::createD() {
    RealVector ret(m_dimension);
    for (unsigned int i = 0; i < m_dimension; i++) ret(i) = pow(m_kappa, i / (m_dimension - 1.0));
    shuffle(ret.begin(), ret.end(), m_rng);
    return ret;
}

RealVector MOBenchmark::createDdup(unsigned int u, unsigned int v) {
    if (v < u) swap(u, v);
    unsigned int z = m_dimension - 1;
    RealVector ret(m_dimension);
    for (unsigned int i = 0; i < z; i++) ret(i) = pow(m_kappa, i / (z - 1.0));
    shuffle(ret.begin(), ret.begin() + z, m_rng);
    ret(z) = ret[u];
    if (v != z) swap(ret(v), ret(z));
    return ret;
}

MOBenchmark::MOBenchmark(string const& name, unsigned int dimension, unsigned int instance, double kappa)
    : m_name(name), m_dimension(dimension), m_instance(instance), m_kappa(kappa), m_a1(1), m_b1(0), m_x1(dimension, 0.0), m_U1(dimension, dimension, 0.0), m_D1(dimension, 0.0), m_A1(dimension, dimension, 0.0), m_H1(dimension, dimension, 0.0), m_a2(1), m_b2(0), m_x2(dimension, 0.0), m_U2(dimension, dimension, 0.0), m_D2(dimension, 0.0), m_A2(dimension, dimension, 0.0), m_H2(dimension, dimension, 0.0), m_delta(dimension, 0.0), m_s(1), m_handler(SearchPointType(dimension, -5.0), SearchPointType(dimension, 5.0)), m_rng(instance) {
    announceConstraintHandler(&m_handler);
    m_features |= CAN_PROPOSE_STARTING_POINT;

    if (name.size() != 3) throw runtime_error("invalid problem name: " + name);
    unsigned int category = name[0] - '0';
    if (category < 1 || category > 9) throw runtime_error("invalid problem name: " + name);
    if (name[1] != '|' && name[1] != '/') throw runtime_error("invalid problem name: " + name);
    bool aligned = (name[1] == '|');
    if (name[2] == 'C')
        m_s = 1.0;
    else if (name[2] == 'I')
        m_s = 0.5;
    else if (name[2] == 'J')
        m_s = 0.25;
    else
        throw runtime_error("invalid problem name: " + name);

    // create the problem instance
    bool deltaFromGEV = false;
    uniform_real_distribution<double> uni(0, 1);
    uniform_int_distribution<unsigned int> uniDim(0, m_dimension - 1);
    if (category == 1) {
        m_U1 = eye(m_dimension);
        m_U2 = eye(m_dimension);
        m_D1 = RealVector(m_dimension, 1.0);
        m_D2 = RealVector(m_dimension, 1.0);
        if (aligned) {
            m_delta(uniDim(m_rng)) = 1;
        } else {
            m_delta = gauss(m_dimension);
            m_delta /= norm_2(m_delta);
        }
    } else if (category == 2) {
        m_U1 = eye(m_dimension);
        m_U2 = eye(m_dimension);
        m_D1 = RealVector(m_dimension, 1.0);
        if (aligned) {
            m_D2 = createD();
            m_delta(uniDim(m_rng)) = 1;
        } else {
            unsigned int i = uniDim(m_rng);
            unsigned int j = uniform_int_distribution<unsigned int>(0, m_dimension - 2)(m_rng);
            if (j >= i) j++;
            m_D2 = createDdup(i, j);
            double angle = 2 * M_PI * uni(m_rng);
            m_delta(i) = cos(angle);
            m_delta(j) = sin(angle);
        }
    } else if (category == 3) {
        m_U1 = eye(m_dimension);
        m_U2 = eye(m_dimension);
        if (aligned) {
            m_D1 = createD();
            m_delta(uniDim(m_rng)) = 1;
        } else {
            unsigned int i = uniDim(m_rng);
            unsigned int j = uniform_int_distribution<unsigned int>(0, m_dimension - 2)(m_rng);
            if (j >= i) j++;
            m_D1 = createDdup(i, j);
            double angle = 2 * M_PI * uni(m_rng);
            m_delta(i) = cos(angle);
            m_delta(j) = sin(angle);
        }
        m_D2 = m_D1;
    } else if (category == 4) {
        m_U1 = eye(m_dimension);
        m_U2 = eye(m_dimension);
        if (aligned) {
            m_D1 = createD();
            m_D2 = createD();
            m_delta(uniDim(m_rng)) = 1;
        } else {
            unsigned int i = uniDim(m_rng);
            unsigned int j = uniform_int_distribution<unsigned int>(0, m_dimension - 2)(m_rng);
            if (j >= i) j++;
            m_D1 = createDdup(i, j);
            m_D2 = createDdup(i, j);
            double angle = 2 * M_PI * uni(m_rng);
            m_delta(i) = cos(angle);
            m_delta(j) = sin(angle);
        }
    } else if (category == 5) {
        m_U1 = eye(m_dimension);
        m_D1 = RealVector(m_dimension, 1.0);
        m_D2 = createD();
        if (aligned) {
            unsigned int i = uniDim(m_rng);
            m_U2 = sampleUfix(i);
            m_delta(i) = 1.0;
        } else {
            m_U2 = sampleU();
            deltaFromGEV = true;
        }
    } else if (category == 6) {
        m_U1 = eye(m_dimension);
        m_D1 = createD();
        m_D2 = createD();
        if (aligned) {
            unsigned int i = uniDim(m_rng);
            m_U2 = sampleUfix(i);
            m_delta(i) = 1.0;
        } else {
            m_U2 = sampleU();
            deltaFromGEV = true;
        }
    } else if (category == 7) {
        m_D1 = createD();
        m_D2 = m_D1;
        if (aligned) {
            unsigned int i = uniDim(m_rng);
            m_U1 = sampleUfix(i);
            m_delta(i) = 1.0;
        } else {
            m_U1 = sampleU();
            deltaFromGEV = true;
        }
        m_U2 = m_U1;
    } else if (category == 8) {
        m_D1 = createD();
        m_D2 = createD();
        if (aligned) {
            unsigned int i = uniDim(m_rng);
            m_U1 = sampleUfix(i);
            m_delta(i) = 1.0;
        } else {
            m_U1 = sampleU();
            deltaFromGEV = true;
        }
        m_U2 = m_U1;
    } else if (category == 9) {
        m_U1 = sampleU();
        m_U2 = sampleU();
        m_D1 = createD();
        m_D2 = createD();
        if (aligned) {
            RealMatrix V;
            RealVector W;
            tie(V, W) = eig(m_U1, m_D1, m_U2, m_D2);
            unsigned int i = uniDim(m_rng);
            RealVector delta = column(V, i);
            delta /= norm_2(delta);
            RealMatrix UT = sampleUTdelta(delta);
            m_U1 = UT % m_U1;
            m_U2 = UT % m_U2;
            m_delta = UT % delta;
        } else {
            deltaFromGEV = true;
        }
    }

    m_A1 = m_U1 % to_diagonal(sqrt(m_D1));
    m_A2 = m_U2 % to_diagonal(sqrt(m_D2));
    m_H1 = m_A1 % trans(m_A1);
    m_H2 = m_A2 % trans(m_A2);

    if (deltaFromGEV) {
        RealMatrix V;
        RealVector W;
        tie(V, W) = eig(m_U1, m_D1, m_U2, m_D2);
        unsigned int i = uniDim(m_rng);
        m_delta = column(V, i);
        m_delta /= norm_2(m_delta);
    }

    // sample single-objective optima in the range [-5, 5]^n
    RealVector center = gauss(dimension);
    while (norm_inf(center) >= 4.5) center = gauss(dimension);
    m_x1 = center - 0.5 * m_delta;
    m_x2 = center + 0.5 * m_delta;

    m_a1 = pow(10.0, 6 * uni(m_rng));
    m_a2 = pow(10.0, 6 * uni(m_rng));
    m_b1 = 2 * m_a1 * uni(m_rng) - m_a1;
    m_b2 = 2 * m_a2 * uni(m_rng) - m_a2;
}
