/* References:
 * https://en.cppreference.com/w/cpp/numeric/random
 * http://www.cplusplus.com/reference/ostream/ostream/operator%3C%3C/
 */

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <random>

using namespace shark;

void writeVector(std::ofstream &out, RealVector &value, int maxSize) {
    for (auto i = 0; i < value.size(); i++) {
        out << "," << value[i];
    }
    if (value.size() < maxSize) {
        for (auto i = 0; i < maxSize - value.size(); i++) {
            out << "," << 0.0;
        }
    }
}

void writeVector(std::ofstream &out, double fitness, int maxSize) {
    out << "," << fitness;
    for (auto i = 0; i < maxSize - 1; i++) {
        out << "," << 0.0;
    }
}

int main() {
    benchmarks::Rastrigin fn;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> nObjectivesDist(2, 4);
    std::uniform_int_distribution<> nDimensionsDist(2, 10);
    // Indicate if the number of dimensions > number of objectives
    auto restrictedDimensions = true;
    RealVector point;
    auto outputFilename = fn.name().append(".csv");
    std::ofstream outputFile;
    outputFile.open(outputFilename);
    outputFile << std::setprecision(10) << std::scientific;
    int index;
    for (int i = 0; i < 10; i++) {
        int maxNObjectives = fn.numberOfObjectives();
        int maxNDimensions = fn.numberOfObjectives();
        // The first 5 samples are random combinations of dimensions
        // and objectives
        // The last 5 samples all have the same dimensions and objectives
        if (i < 6) {
            if (fn.hasScalableObjectives()) {
                fn.setNumberOfObjectives(nObjectivesDist(rng));
            }
            if (fn.hasScalableDimensionality()) {
                do {
                    fn.setNumberOfVariables(nDimensionsDist(rng));
                } while (restrictedDimensions && fn.numberOfVariables() < fn.numberOfObjectives());
            }
        }
        maxNObjectives = fn.hasScalableObjectives() ? 4 : fn.numberOfObjectives();
        maxNDimensions = fn.hasScalableDimensionality() ? 10 : fn.numberOfVariables();
        fn.init();
        point = fn.proposeStartingPoint();
        auto fitness = fn(point);
        outputFile << 4 << "," << 4 + fn.numberOfVariables() << "," << 4 + maxNDimensions << "," << 4 + maxNDimensions + fn.numberOfObjectives();
        writeVector(outputFile, point, maxNDimensions);
        writeVector(outputFile, fitness, maxNObjectives);
        outputFile << "\n";
    }
    std::cout << outputFilename << std::endl;
    outputFile.close();
}
