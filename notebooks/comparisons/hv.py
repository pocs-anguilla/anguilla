# Compares the Python hypervolume implementation with
# Shark's and the "chimeric solution" obtained using
# a C++ implementation with Python bindings.

# The benchmark experiment is adapted from section 5 of
# [2011:hypervolume-3d] and uses the CLIFF3D function
# (problem 8).

import os
import sys

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("build/"))

import math
import numpy as np
import time
import matplotlib.pyplot as plt
import anguilla.hypervolume as hv
import hv_comparison as hvc


def cliff_3d(n, rng=np.random.default_rng()):
    """Generat n Cliff3D points."""
    vs = rng.standard_normal(size=(n, 2))
    ys = np.zeros((n, 3))
    for i in range(n):
        c = np.linalg.norm(vs[i])
        ys[i, 0:2] = 10.0 * np.abs(vs[i]) / c
    ys[:, 2] = rng.uniform(low=0.0, high=10.0, size=n)
    return ys


ref_p_3d = np.array([1.1, 1.1, 1.1], dtype=float)
ps_3d = np.array(
    [
        [6.56039859404455e-2, 0.4474014917277, 0.891923776019316],
        [3.74945443950542e-2, 3.1364039802686e-2, 0.998804513479922],
        [0.271275894554688, 0.962356894778677, 1.66911984440026e-2],
        [0.237023460537611, 0.468951509833942, 0.850825693417425],
        [8.35813910785332e-2, 0.199763306732937, 0.97627289850149],
        [1.99072649788403e-2, 0.433909411793732, 0.900736544810901],
        [9.60698311356187e-2, 0.977187045721721, 0.18940978121319],
        [2.68052822856208e-2, 2.30651870780559e-2, 0.999374541394087],
        [0.356223209018184, 0.309633114503212, 0.881607826507812],
        [0.127964409429531, 0.73123479272024, 0.670015513129912],
        [0.695473366395562, 0.588939459338073, 0.411663831140169],
        [0.930735605917613, 0.11813654121718, 0.346085234453039],
        [0.774030940645471, 2.83363630460836e-2, 0.632513362272141],
        [0.882561783965009, 4.80931050853475e-3, 0.470171849451808],
        [4.92340623346446e-3, 0.493836329534438, 0.869540936185878],
        [0.305054163869799, 0.219367569077876, 0.926725324323535],
        [0.575227233936948, 0.395585597387712, 0.715978815661927],
        [0.914091673974525, 0.168988399705031, 0.368618138912863],
        [0.225088318852838, 0.796785147906617, 0.560775067911755],
        [0.306941172015014, 0.203530333828304, 0.929710987422322],
        [0.185344081015371, 0.590388202293731, 0.785550343533082],
        [0.177181921358634, 0.67105558509432, 0.719924279669315],
        [0.668494587335475, 0.22012845825454, 0.710393164782469],
        [0.768639363955671, 0.256541291890516, 0.585986427942633],
        [0.403457020846225, 0.744309886218013, 0.532189088208334],
        [0.659545359568811, 0.641205442223306, 0.39224418355721],
        [0.156141960251846, 8.36191498217669e-2, 0.984188765446851],
        [0.246039496399399, 0.954377757574506, 0.16919711007753],
        [3.02243260456876e-2, 0.43842801306405, 0.898257962656493],
        [0.243139979715573, 0.104253945099703, 0.96437236853565],
        [0.343877707314699, 0.539556201272222, 0.768522757034998],
        [0.715293885551218, 0.689330705208567, 0.114794756629825],
        [1.27610149409238e-2, 9.47996983636579e-2, 0.995414573777096],
        [0.30565381275615, 0.792827267212719, 0.527257689476066],
        [0.43864576057661, 3.10389339442242e-2, 0.8981238674636],
    ]
)


if __name__ == "__main__":
    # Checking that the bindings are passing correct data around
    vol = hvc.shark_hv_f8([p for p in ps_3d], ref_p_3d)
    assert math.isclose(vol, 0.60496383631719475)

    # Running the benchmark
    ns = np.arange(start=1, stop=11, step=1) * 100
    elapsed_shark = np.zeros(len(ns))
    elapsed_py = np.zeros(len(ns))
    m_fronts = 100
    n_runs = 10

    for i in range(len(ns)):
        n = ns[i]
        for _ in range(m_fronts):
            ys = cliff_3d(n)
            ref_y = np.max(ys, axis=0) + 1e-4
            elapsed_shark[i] += (
                hvc.benchmark_shark_hv_contributions(ys, ref_y, n_runs) / 1e6
            )  # in seconds

            result = hv.contributions(ys, ref_y)
            t_start = time.perf_counter()
            for _ in range(n_runs):
                result = hv.contributions(ys, ref_y)
            t_end = time.perf_counter()
            t_diff = t_end - t_start
            elapsed_py[i] += t_diff / float(n_runs)  # in seconds

    elapsed_shark /= float(m_fronts)
    elapsed_py /= float(m_fronts)

    # Plotting the results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].plot(ns, elapsed_shark, label="Shark")
    axs[0].plot(ns, elapsed_py, label="Python")
    axs[0].set_xlabel("Pareto front size")
    axs[0].set_ylabel(
        "Avg. running time ({} runs and {} fronts) in seconds".format(
            n_runs, m_fronts
        )
    )
    axs[0].legend()
    axs[1].plot(ns, elapsed_shark, label="Shark")
    axs[1].set_xlabel("Pareto front size")
    axs[1].set_ylabel(
        "Avg. running time ({} runs and {} fronts) in seconds".format(
            n_runs, m_fronts
        )
    )
    axs[1].legend()
    fig.savefig("hv_comparison.pdf", bbox_inches="tight")
