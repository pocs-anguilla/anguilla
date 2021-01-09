/* The following code was adapted from [2008:shark].
   See https://git.io/JIKs7 and https://git.io/JIKHB */
#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <boost/format.hpp>
#include <ios>
#include <iostream>

#include "matplotlibcpp.h"

using namespace shark;

namespace plt = matplotlibcpp;

struct PointExtractor {
    template <class T>
    RealVector const &operator()(T const &arg) const {
        return arg.value;
    }
};

template <typename Optimizer = MOCMA, typename ObjectiveFunction = benchmarks::CIGTAB2>
class PopulationPlotExperiment {
   public:
    static void run(int mu, int n, int maxEvaluations, RealVector *reference = nullptr, bool individual = false) {
        std::cout.setf(std::ios_base::scientific);
        std::cout.precision(10);
        std::vector<std::string> formats = {"rx", "bx", "gx"};
        plt::figure_size(500, 500);

        std::string st = individual ? "I" : "P";
        for (int i = 0; i < formats.size(); i++) {
            ObjectiveFunction fn;
            if (fn.hasScalableObjectives()) {
                fn.setNumberOfObjectives(2);
            }
            fn.setNumberOfVariables(n);

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
            if (i + 1 == formats.size()) {
                plt::title(boost::str(boost::format("%1%(n=%2%), %3%-%4%\nmu=%5%,evals=%6%") % fn.name() % n % optimizer.name() % st % mu % fn.evaluationCounter()));
                plt::save(boost::str(boost::format("./%1%n%2%-%3%-%4%-mu%5%-fe%6%.png") % fn.name() % n % optimizer.name() % st % mu % fn.evaluationCounter()));
            }
        }
    }
};

int main(int argc, char *argv[]) {
    int mu = argc > 1 ? std::atoi(argv[1]) : 10;
    int n = argc > 2 ? std::atoi(argv[2]) : 5;
    int maxEvaluations = argc > 3 ? std::atoi(argv[3]) : 1000;
    bool useSteadyState = argc > 4 ? std::strcmp("steady", argv[4]) == 0 : false;
    bool individual = argc > 5 ? std::strcmp("individual", argv[5]) == 0 : false;

    RealVector reference = {11.0, 11.0};
    RealVector *reference_ptr = nullptr;

    if (useSteadyState) {
        PopulationPlotExperiment<SteadyStateMOCMA, benchmarks::ZDT1> experiment;
        experiment.run(mu, n, maxEvaluations, reference_ptr, individual);
    } else {
        PopulationPlotExperiment<MOCMA, benchmarks::ZDT1> experiment;
        experiment.run(mu, n, maxEvaluations, reference_ptr, individual);
    }
}
