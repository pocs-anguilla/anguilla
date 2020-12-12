#include <ios>
#include <iostream>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

int main()
{
    std::cout.setf(std::ios_base::scientific);
    std::cout.precision(10);

    benchmarks::DTLZ2 fn;
    fn.setNumberOfObjectives(2);
    fn.setNumberOfVariables(5);
    fn.init();

    for (int i = 0; i < 3; i++)
    {
        auto point = fn.proposeStartingPoint();
        std::cout << point << fn(point) << std::endl;
    }
}
