
#include <ios>
#include <iostream>
#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

int main()
{
  std::cout.setf(std::ios_base::scientific);
  std::cout.precision(10);

  benchmarks::Sphere sphere(4);
  CMA cma;
  cma.setInitialSigma(0.3);
  sphere.init();
  cma.init(sphere, sphere.proposeStartingPoint());

  do
  {
    cma.step(sphere);
    std::cout << sphere.evaluationCounter() << " " << cma.solution().value << " " << cma.solution().point << " " << cma.sigma() << std::endl;
  } while (cma.solution().value > 1E-14);
}
