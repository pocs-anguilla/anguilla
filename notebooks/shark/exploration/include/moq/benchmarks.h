
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

#pragma once

#include <shark/LinAlg/Base.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

#include <random>
#include <string>
#include <tuple>

// class representing the 108 multi-objective problems
class MOBenchmark : public shark::MultiObjectiveFunction {
   private:
    std::string m_name;
    unsigned int m_dimension;
    unsigned int m_instance;
    double m_kappa;
    double m_a1;
    double m_b1;
    shark::RealVector m_x1;
    shark::RealMatrix m_U1;
    shark::RealVector m_D1;
    shark::RealMatrix m_A1;
    shark::RealMatrix m_H1;
    double m_a2;
    double m_b2;
    shark::RealVector m_x2;
    shark::RealMatrix m_U2;
    shark::RealVector m_D2;
    shark::RealMatrix m_A2;
    shark::RealMatrix m_H2;
    shark::RealVector m_delta;
    double m_s;
    shark::BoxConstraintHandler<SearchPointType> m_handler;
    std::mt19937 m_rng;

    // solve U_1 diag(D_1) U_1^T x = \lambda U_2 diag(D_2) U_2^T x for x and \lambda
    static std::tuple<shark::RealMatrix, shark::RealVector> eig(shark::RealMatrix const& U1, shark::RealVector const& D1, shark::RealMatrix const& U2, shark::RealVector const& D2);

    // construct a unit (identity) matrix
    static shark::RealMatrix eye(unsigned int n);

    // sample a vector with i.i.d. standard normal entries
    shark::RealVector gauss(unsigned int n);

    // sample a matrix with i.i.d. standard normal entries
    shark::RealMatrix gauss(unsigned int n, unsigned int m);

    // sample uniformly from the orthogonal group
    shark::RealMatrix sampleU();

    // sample uniformly from the orthogonal group but leave row/column u fixed at the identity
    shark::RealMatrix sampleUfix(unsigned int u = 0);

    // sample uniformly from the orthogonal group under the constraint that delta is a column of the matrix
    shark::RealMatrix sampleUTdelta(shark::RealVector const& delta);

    // create ellipsoid diagonal
    shark::RealVector createD();

    // create ellipsoid diagonal with duplicate entry at u and v
    shark::RealVector createDdup(unsigned int u, unsigned int v);

   public:
    MOBenchmark(std::string const& name, unsigned int dimension, unsigned int instance, double kappa = 1e3);

    // Shark objective function interface
    std::string name() const override { return m_name; }
    std::size_t numberOfVariables() const override { return m_dimension; }
    bool hasScalableDimensionality() const override { return false; }
    std::size_t numberOfObjectives() const override { return 2; }
    bool hasScalableObjectives() const override { return false; }
    SearchPointType proposeStartingPoint() const override { return shark::RealVector(m_dimension, 0.0); }

    // additional properties
    unsigned int instance() const { return m_instance; }
    double kappa() const { return m_kappa; }

    // central evaluation interface
    ResultType eval(SearchPointType const& x) const override {
        m_evaluationCounter++;
        shark::RealVector d1 = trans(m_A1) % (x - m_x1);
        shark::RealVector d2 = trans(m_A2) % (x - m_x2);
        return shark::RealVector{
            0.5 * m_a1 * pow(remora::inner_prod(d1, d1), m_s) + m_b1,
            0.5 * m_a2 * pow(remora::inner_prod(d2, d2), m_s) + m_b2};
    }

    // Added for an experiment in mocma.cpp
    std::tuple<std::vector<double>, std::vector<double>> paretoFront(int num) {
        std::vector<double> ts(num);
        ts[0] = 0.0;
        float step = 1.0 / (num - 1.0);
        for (int i = 1; i != num; i++) {
            ts[i] = ts[i - 1] + step;
        }
        std::vector<double> y1(num);
        std::vector<double> y2(num);
        for (int i = 0; i != num; i++) {
            auto value = eval(m_x1 * (1.0 - ts[i]) + m_x2 * (ts[i]));
            y1[i] = value(0);
            y2[i] = value(1);
        }
        return std::make_tuple(y1, y2);
    }

    // The following data is provided only for evaluation purposes.
    // It must not be used by a black-box optimization algorithm.
    shark::RealVector const& x1star() const { return m_x1; }
    shark::RealVector const& x2star() const { return m_x2; }
    shark::RealMatrix const& U1() const { return m_U1; }
    shark::RealMatrix const& U2() const { return m_U2; }
    shark::RealVector const& D1() const { return m_D1; }
    shark::RealVector const& D2() const { return m_D2; }
    shark::RealMatrix const& A1() const { return m_A1; }
    shark::RealMatrix const& A2() const { return m_A2; }
    shark::RealMatrix const& H1() const { return m_H1; }
    shark::RealMatrix const& H2() const { return m_H2; }
    shark::RealVector utopian() const { return shark::RealVector{m_b1, m_b2}; }
    shark::RealVector nadir() const {
        shark::RealVector d1 = trans(m_A1) % m_delta;
        shark::RealVector d2 = trans(m_A2) % m_delta;
        return shark::RealVector{
            0.5 * m_a1 * pow(remora::inner_prod(d1, d1), m_s) + m_b1,
            0.5 * m_a2 * pow(remora::inner_prod(d2, d2), m_s) + m_b2};
    }
};
