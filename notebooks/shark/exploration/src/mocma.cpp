/* The following code was adapted from [2008:shark].
   See https://git.io/JIKs7 and https://git.io/JIKHB */
#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/Core/Random.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include <stdlib.h>

#include <algorithm>
#include <boost/format.hpp>
#include <ios>
#include <iostream>

#include "matplotlibcpp/matplotlibcpp.h"
#include "moq/benchmarks.h"

using namespace shark;

namespace plt = matplotlibcpp;

std::string safe_name(std::string name) {
    std::replace(name.begin(), name.end(), '/', 'N');
    std::replace(name.begin(), name.end(), '|', 'A');
    return name;
}

struct PointExtractor {
    template <class T>
    RealVector const &operator()(T const &arg) const {
        return arg.value;
    }
};

template <typename Optimizer = MOCMA>
class PopulationPlotExperiment {
   public:
    static void run(int seed, int mu, int n, int maxEvaluations, RealVector *reference = nullptr, bool individual = false, std::string extra = "1|C", int instance = 1, std::size_t maxTrials = 3) {
        random::globalRng().seed(seed);
        std::cout.setf(std::ios_base::scientific);
        std::cout.precision(10);
        std::vector<std::string> formats = {"xb", "xg", "xy"};
        plt::figure_size(500, 350);

        std::string st = individual ? "I" : "P";
        auto nTrials = std::min(formats.size(), maxTrials);
        for (int i = 0; i < nTrials; i++) {
            benchmarks::CIGTAB2 fn(n);
            //MOBenchmark fn(extra, n, instance);
            if (fn.hasScalableObjectives()) {
                fn.setNumberOfObjectives(2);
            }
            if (fn.hasScalableDimensionality()) {
                fn.setNumberOfVariables(n);
            }

            /*if (i == 0) {
                auto [frontY1, frontY2] = fn.paretoFront(50);
                plt::plot(frontY1, frontY2, "r-");
            }*/

            Optimizer optimizer;
            if (individual) {
                optimizer.notionOfSuccess() = Optimizer::NotionOfSuccess::IndividualBased;
            } else {
                optimizer.notionOfSuccess() = Optimizer::NotionOfSuccess::PopulationBased;
            }
            optimizer.mu() = mu;
            optimizer.initialSigma() = 1.0;
            if (reference != nullptr) {
                optimizer.indicator().setReference(*reference);
            }

            fn.init();
            optimizer.init(fn);

            while (fn.evaluationCounter() < maxEvaluations) {
                optimizer.step(fn);
            }

            auto solution = optimizer.solution();
            int size = solution.size();
            std::vector<double> x(size), y(size);
            for (int i = 0; i != size; i++) {
                x[i] = solution[i].value[0];
                y[i] = solution[i].value[1];
            }
            plt::plot(x, y, formats[i]);

            if (reference != nullptr) {
                HypervolumeCalculator hyp;
                double volume = hyp(boost::adaptors::transform(solution, PointExtractor()), *reference);
                std::cout << "Trial " << i << ": " << volume << std::endl;
            }
            if (i + 1 == nTrials) {
                plt::title(boost::str(boost::format("%1%(n=%2%), %3%-%4%\nmu=%5%,evals=%6%") % fn.name() % n % optimizer.name() % st % mu % fn.evaluationCounter()));
                plt::save(boost::str(boost::format("./%1%n%2%-%3%-%4%-mu%5%-fe%6%-seed%7%.png") % safe_name(fn.name()) % n % optimizer.name() % st % mu % fn.evaluationCounter() % seed));
            }
        }
    }
};

int main(int argc, char *argv[]) {
    int seed = argc > 1 ? std::atoi(argv[1]) : 0;
    int mu = argc > 2 ? std::atoi(argv[2]) : 10;
    int n = argc > 3 ? std::atoi(argv[3]) : 5;
    int maxEvaluations = argc > 4 ? std::atoi(argv[4]) : 5000;
    bool useSteadyState = argc > 5 ? std::strcmp("steady", argv[5]) == 0 : false;
    bool individual = argc > 6 ? std::strcmp("individual", argv[6]) == 0 : false;
    std::string extra = argc > 7 ? std::string(argv[7]) : "1|C";
    int instance = argc > 8 ? std::atoi(argv[8]) : rand();

    RealVector reference = {11.0, 11.0};
    RealVector *reference_ptr = nullptr;
    std::size_t maxTrials = 1;
    if (useSteadyState) {
        PopulationPlotExperiment<SteadyStateMOCMA> experiment;
        experiment.run(seed, mu, n, maxEvaluations, reference_ptr, individual, extra, instance, maxTrials);
    } else {
        PopulationPlotExperiment<MOCMA> experiment;
        experiment.run(seed, mu, n, maxEvaluations, reference_ptr, individual, extra, instance, maxTrials);
    }
}
