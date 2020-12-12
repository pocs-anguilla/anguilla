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
#include "pyshark.h"

#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>

// The timing code comes from PMPH's starter code
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution = 1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff < 0);
}

double shark_hv_f8(const PySharkVectorList<double> &ps, const PySharkVector<double> &ref_p)
{
    shark::HypervolumeCalculator hv;
    return hv(ps, ref_p);
}

double benchmark_shark_hv_contributions(const PySharkVectorList<double> &ps, const PySharkVector<double> &ref_p, const int runs)
{
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    shark::HypervolumeContribution3D hvc3d;
    auto result = hvc3d.largest(ps, ps.size(), ref_p);
    gettimeofday(&t_start, NULL);
    for (auto i = 0; i < runs; ++i)
    {
        auto result = hvc3d.largest(ps, ps.size(), ref_p);
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
    return elapsed / float(runs);
}

PYBIND11_MODULE(hv_comparison, m)
{
    m.doc() = "Experiment comparing 3-D HV implementations.";
    m.def("shark_hv_f8", &shark_hv_f8);
    m.def("benchmark_shark_hv_contributions", &benchmark_shark_hv_contributions);
}
