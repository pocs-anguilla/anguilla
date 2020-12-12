#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <functional>

#include <sys/time.h>
#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pyshark/LinAlg/Base.h>

#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  unsigned int resolution = 1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff < 0);
}

using namespace shark;

static double REF_POINT_A[] = {1.1, 1.1, 1.1};

static double POINTS_A[][3] = {
    {6.56039859404455e-2, 0.4474014917277, 0.891923776019316},
    {3.74945443950542e-2, 3.1364039802686e-2, 0.998804513479922},
    {0.271275894554688, 0.962356894778677, 1.66911984440026e-2},
    {0.237023460537611, 0.468951509833942, 0.850825693417425},
    {8.35813910785332e-2, 0.199763306732937, 0.97627289850149},
    {1.99072649788403e-2, 0.433909411793732, 0.900736544810901},
    {9.60698311356187e-2, 0.977187045721721, 0.18940978121319},
    {2.68052822856208e-2, 2.30651870780559e-2, 0.999374541394087},
    {0.356223209018184, 0.309633114503212, 0.881607826507812},
    {0.127964409429531, 0.73123479272024, 0.670015513129912},
    {0.695473366395562, 0.588939459338073, 0.411663831140169},
    {0.930735605917613, 0.11813654121718, 0.346085234453039},
    {0.774030940645471, 2.83363630460836e-2, 0.632513362272141},
    {0.882561783965009, 4.80931050853475e-3, 0.470171849451808},
    {4.92340623346446e-3, 0.493836329534438, 0.869540936185878},
    {0.305054163869799, 0.219367569077876, 0.926725324323535},
    {0.575227233936948, 0.395585597387712, 0.715978815661927},
    {0.914091673974525, 0.168988399705031, 0.368618138912863},
    {0.225088318852838, 0.796785147906617, 0.560775067911755},
    {0.306941172015014, 0.203530333828304, 0.929710987422322},
    {0.185344081015371, 0.590388202293731, 0.785550343533082},
    {0.177181921358634, 0.67105558509432, 0.719924279669315},
    {0.668494587335475, 0.22012845825454, 0.710393164782469},
    {0.768639363955671, 0.256541291890516, 0.585986427942633},
    {0.403457020846225, 0.744309886218013, 0.532189088208334},
    {0.659545359568811, 0.641205442223306, 0.39224418355721},
    {0.156141960251846, 8.36191498217669e-2, 0.984188765446851},
    {0.246039496399399, 0.954377757574506, 0.16919711007753},
    {3.02243260456876e-2, 0.43842801306405, 0.898257962656493},
    {0.243139979715573, 0.104253945099703, 0.96437236853565},
    {0.343877707314699, 0.539556201272222, 0.768522757034998},
    {0.715293885551218, 0.689330705208567, 0.114794756629825},
    {1.27610149409238e-2, 9.47996983636579e-2, 0.995414573777096},
    {0.30565381275615, 0.792827267212719, 0.527257689476066},
    {0.43864576057661, 3.10389339442242e-2, 0.8981238674636}};

constexpr double VOL_A = 0.60496383631719475;

int main()
{
  std::cout.setf(std::ios_base::scientific);
  std::cout.precision(10);

  double elapsed;
  struct timeval t_start, t_end, t_diff;

  gettimeofday(&t_start, NULL);

  // Read input from standard input
  std::string line;
  while (std::getline(std::cin, line))
  {
    const double number = std::stod(line);
    std::cout << "read: " << number << std::endl;
  }

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
  std::cout << "Took: " << elapsed << std::endl;

  return 0;

  std::vector<RealVector> points;
  for (auto it = std::begin(POINTS_A); it != std::end(POINTS_A); it++)
  {
    RealVector point = remora::dense_vector_adaptor<double>(*it, 3, 1);
    points.push_back(point);
    std::cout << point << "\n";
  }
  std::cout << std::endl;
}
