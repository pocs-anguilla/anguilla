
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

#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/RealCodedNSGAII.h>
#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>
#include <shark/Core/Random.h>

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "moq/benchmarks.h"

using namespace shark;
using namespace remora;
using namespace std;

// normalized dominated hypervolume
// [problem, align, shape, instance, algo, "time"]
// array is too large for a local variable
constexpr auto RUNS = 101;
double result[9][2][3][RUNS][3][100];

template <typename Solution>
double hypervolume(Solution const& solution, RealVector const& reference) {
    HypervolumeCalculator hv;
    std::vector<RealVector> front;
    for (size_t i = 0; i < solution.size(); i++) {
        RealVector v = solution[i].value;
        if (v(0) < reference(0) && v(1) < reference(1)) front.push_back(v);
    }
    return hv(front, reference);
}

int main(int argc, char** argv) {
    auto dim = 10;
    auto mu = 20;
    auto budget = 100000;

    random::globalRng().seed(42);  // (the answer)
    cout << setprecision(20);

    // instantiate solvers
    MOCMA mocma;
    mocma.initialSigma() = 3.0;
    mocma.mu() = mu;
    SMSEMOA smsemoa;
    smsemoa.mu() = mu;
    RealCodedNSGAII nsga2;
    nsga2.mu() = mu;
    std::vector<AbstractMultiObjectiveOptimizer<RealVector>*> algos{&mocma, &smsemoa, &nsga2};

    string problemchar = "123456789";
    string alignchar = "|/";
    string shapechar = "CIJ";
    for (int problem = 0; problem < 9; problem++) {
        for (int align = 0; align < 2; align++) {
            for (int shape = 0; shape < 3; shape++) {
                string name;
                name += problemchar[problem];
                name += alignchar[align];
                name += shapechar[shape];

                for (int instance = 0; instance < RUNS; instance++) {
                    // problem and reference point
                    cout << name << " " << instance << endl;
                    MOBenchmark f(name, dim, instance);
                    RealVector utopian = f.utopian();
                    RealVector nadir = f.nadir();
                    RealVector ref = nadir;
                    ref(0) += 0.1 * (nadir(0) - utopian(0));
                    ref(1) += 0.1 * (nadir(1) - utopian(1));

                    // reference volume
                    double refvol = (nadir(0) - utopian(0)) * (nadir(1) - utopian(1));

                    // run all algorithms
                    for (int algo = 0; algo < 3; algo++) {
                        auto& a = *algos[algo];
                        f.init();
                        a.init(f);
                        for (int t = 0; t < 100; t++) {
                            while (f.evaluationCounter() < budget * (t + 1) / 100) a.step(f);
                            double hv = hypervolume(a.solution(), nadir);
                            result[problem][align][shape][instance][algo][t] = hv / refvol;
                        }
                        cout << "  [" << algo << "]: " << result[problem][align][shape][instance][algo][99] << endl;
                    }
                }
            }
        }
    }

    // store the results for later processing
    FILE* file = fopen("results", "wb+");
    fwrite(result, sizeof(double), sizeof(result) / sizeof(double), file);
    fclose(file);
}
