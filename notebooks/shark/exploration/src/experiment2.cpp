/* The following code was adapted from [2008:shark].
 * See https://git.io/JIKs7 and https://git.io/JIKHB
 */

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

// Shark
#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
using namespace shark;

// STL
#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>

// Boost
#include <boost/format.hpp>

std::string name(std::string name, int mu, bool individualBased) {
    std::string suffix = individualBased ? "I" : "P";
    if (name == "SteadyStateMOCMA") {
        return boost::str(boost::format("(%1%+1)-MO-CMA-ES-%2%") % mu % suffix);
    } else {
        return boost::str(boost::format("(%1%+%1%)-MO-CMA-ES-%2%") % mu % suffix);
    }
}

template <typename ObjectiveFunction, typename Optimizer, bool individualBased>
class RunTrials {
   public:
    static void run(int mu, double initialSigma, int nObjectives, int nVariables, int nTrials, RealVector *reference = nullptr) {
        for (auto t = 0; t < nTrials; ++t) {
            Optimizer opt;
            ObjectiveFunction fn(nVariables);

            if (fn.hasScalableObjectives()) {
                fn.setNumberOfObjectives(nObjectives);
            }
            if (fn.numberOfObjectives() != nObjectives) {
                throw std::runtime_error("Could not set target value for number of objectives.");
            }
            if (fn.hasScalableDimensionality()) {
                fn.setNumberOfVariables(nVariables);
            }
            if (individualBased) {
                opt.notionOfSuccess() = Optimizer::NotionOfSuccess::IndividualBased;
            } else {
                opt.notionOfSuccess() = Optimizer::NotionOfSuccess::PopulationBased;
            }

            opt.mu() = mu;
            opt.initialSigma() = initialSigma;

            if (reference != nullptr) {
                opt.indicator().setReference(*reference);
            }
            fn.init();
            opt.init(fn);

            int nextEvaluationsLimit = 5000;
            while (nextEvaluationsLimit < 50001) {
                while (fn.evaluationCounter() < nextEvaluationsLimit) {
                    opt.step(fn);
                }
                const auto optName = name(opt.name(), mu, individualBased);
                auto filename = boost::str(boost::format("%1%_%2%_%3%_%4%.fitness.csv") % fn.name() % optName % (t + 1) % nextEvaluationsLimit);
                std::cout << "Writing file: " << filename << std::endl;
                std::ofstream logfile;
                logfile.open(filename);
                logfile << std::setprecision(10);
                logfile << "# Generated with Shark 4.1.x"
                        << "\n";
                logfile << "# Function: " << fn.name() << ": " << fn.numberOfVariables() << " -> " << fn.numberOfObjectives() << "\n";
                logfile << "# Optimizer: " << optName << "\n";
                logfile << "# Trial: " << (t + 1) % nextEvaluationsLimit << "\n";
                logfile << "# Evaluations: " << fn.evaluationCounter() << "\n";
                logfile << "# Observation: fitness\n";

                const auto solution = opt.solution();
                const auto size = solution.size();
                for (auto i = 0; i < size; ++i) {
                    const auto &value = solution[i].value;
                    for (auto j = 0; j < value.size(); ++j) {
                        logfile << value[j];
                        if (j != value.size() - 1) {
                            logfile << ",";
                        }
                    }
                    logfile << "\n";
                }
                logfile.close();
                nextEvaluationsLimit += 5000;
            }
        }
    }
};

/* 
 * Create the experiment data according to sec. 4.1 of [2010:mo-cma-es].
 */
int main() {
    RealVector reference = {11.0, 11.0};
    RealVector *referencePtr = nullptr;

    constexpr int nVariables_dConstrainedNonRotated = 30;
    constexpr int nVariables_dRotated = 10;
    constexpr int nTrials = 25;
    constexpr int mu = 100;

    // Two objectives.
    /*{
        constexpr auto initialSigma = 0.6;
        constexpr auto nVariables = nVariables_dConstrainedNonRotated;
        RunTrials<benchmarks::ZDT1, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ZDT1, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ZDT1, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ZDT1, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6;
        constexpr auto nVariables = nVariables_dConstrainedNonRotated;
        RunTrials<benchmarks::ZDT2, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ZDT2, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ZDT2, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ZDT2, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6;
        constexpr auto nVariables = nVariables_dConstrainedNonRotated;
        RunTrials<benchmarks::ZDT3, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ZDT3, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ZDT3, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ZDT3, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6;
        constexpr auto nVariables = nVariables_dConstrainedNonRotated;
        RunTrials<benchmarks::ZDT4, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ZDT4, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ZDT4, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ZDT4, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6;
        constexpr auto nVariables = nVariables_dConstrainedNonRotated;
        RunTrials<benchmarks::ZDT6, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ZDT6, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ZDT6, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ZDT6, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6 * (1.0 - -1.0);
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::IHR1, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::IHR1, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::IHR1, MOCMA, true> indOpt2;
        RunTrials<benchmarks::IHR1, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6 * (1.0 - -1.0);
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::IHR2, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::IHR2, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::IHR2, MOCMA, true> indOpt2;
        RunTrials<benchmarks::IHR2, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6 * (1.0 - -1.0);
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::IHR3, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::IHR3, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::IHR3, MOCMA, true> indOpt2;
        RunTrials<benchmarks::IHR3, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6 * (5.0 - -5.0);
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::IHR4, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::IHR4, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::IHR4, MOCMA, true> indOpt2;
        RunTrials<benchmarks::IHR4, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.6 * (5.0 - -5.0);
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::IHR6, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::IHR6, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::IHR6, MOCMA, true> indOpt2;
        RunTrials<benchmarks::IHR6, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::ELLI1, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ELLI1, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ELLI1, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ELLI1, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::ELLI2, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::ELLI2, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::ELLI2, MOCMA, true> indOpt2;
        RunTrials<benchmarks::ELLI2, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::CIGTAB1, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::CIGTAB1, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::CIGTAB1, MOCMA, true> indOpt2;
        RunTrials<benchmarks::CIGTAB1, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::CIGTAB2, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::CIGTAB2, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::CIGTAB2, MOCMA, true> indOpt2;
        RunTrials<benchmarks::CIGTAB2, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 2, nVariables, nTrials, referencePtr);
    }
    */
    // Three objectives.
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ1, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ1, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ1, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ1, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ2, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ2, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ2, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ2, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ3, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ3, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ3, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ3, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ4, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ4, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ4, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ4, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ5, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ5, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ5, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ5, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ6, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ6, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ6, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ6, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
    {
        constexpr auto initialSigma = 0.1;
        constexpr auto nVariables = nVariables_dRotated;
        RunTrials<benchmarks::DTLZ7, SteadyStateMOCMA, true> indOpt1;
        RunTrials<benchmarks::DTLZ7, SteadyStateMOCMA, false> popOpt1;
        indOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt1.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        RunTrials<benchmarks::DTLZ7, MOCMA, true> indOpt2;
        RunTrials<benchmarks::DTLZ7, MOCMA, false> popOpt2;
        indOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
        popOpt2.run(mu, initialSigma, 3, nVariables, nTrials, referencePtr);
    }
}
