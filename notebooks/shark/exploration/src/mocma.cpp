/* The following code was adapted from [2008:shark].
   See https://git.io/JIKs7 and https://git.io/JIKHB */
#include <ios>
#include <iostream>
#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>

using namespace shark;

struct PointExtractor
{

    template <class T>
    RealVector const &operator()(T const &arg) const
    {
        return arg.value;
    }
};

int main()
{
    std::cout.setf(std::ios_base::scientific);
    std::cout.precision(10);

    benchmarks::DTLZ2 fn;
    fn.setNumberOfObjectives(2); // 3
    fn.setNumberOfVariables(5);  // 7

    MOCMA optimizer;
    optimizer.mu() = 10;

    for (int i = 0; i < 100; i++)
    {
        fn.init();
        optimizer.init(fn);

        while (fn.evaluationCounter() < 1000) // 25000
        {
            optimizer.step(fn);
        }

        /*for (auto solution : optimizer.solution())
        {
            for (auto j = 0; j < fn.numberOfObjectives(); j++)
            {
                std::cout << solution.value[j] << " ";
            }
            std::cout << std::endl;
        }*/

        RealVector reference = {11.0, 11.0};
        HypervolumeCalculator hyp;
        double volume = hyp(boost::adaptors::transform(optimizer.solution(), PointExtractor()), reference);
        std::cout << "Volume: " << volume << std::endl;
    }
}
