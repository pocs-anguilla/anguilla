"""Testsuite for the :py:mod:`archive` package."""
import math
import unittest

import numpy as np

from anguilla.archive import UPMOArchive, UPMOParameters, UPMOIndividual
from anguilla.dominance import non_dominated_sort
from anguilla.util import random_2d_3d_front
import anguilla.hypervolume as hv


class TestParameters(unittest.TestCase):
    def test_default_values(self) -> None:
        """Test that the default values are those defined in the literature."""
        n = 3
        n_f = float(3)
        initial_step_size = 3.5
        parameters = UPMOParameters(n, initial_step_size)
        self.assertTrue(
            parameters.initial_step_size == initial_step_size,
            "initial_step_size",
        )
        self.assertTrue(parameters.d == 1.0 + n_f / 2.0, "d")
        self.assertTrue(parameters.p_target_succ == 0.5, "p_target_succ")
        self.assertTrue(parameters.c_p == 0.5 / 2.5, "c_p")
        self.assertTrue(parameters.p_threshold == 0.44, "p_threshold")
        self.assertTrue(
            math.isclose(parameters.c_cov, 2.0 / (n_f ** 2.1 + 3.0)), "c_cov"
        )
        self.assertTrue(parameters.p_extreme == 1.0 / 5.0, "p_extreme")
        self.assertTrue(parameters.sigma_min == 1e-15, "sigma_min")
        self.assertTrue(parameters.alpha == 3.0, "alpha")
        self.assertTrue(math.isclose(parameters.c_r, parameters.c_cov / 2.0))


class TestIndividual(unittest.TestCase):
    def test_creation(self) -> None:
        """Test creating an individual."""
        point = np.array([1.0, 2.0, 3.0, 4.0])
        fitness = np.array([3.0, 4.0])
        individual = UPMOIndividual(point, fitness)
        self.assertTrue(np.allclose(individual.point, point))
        self.assertTrue(np.allclose(individual.fitness, fitness))
        self.assertTrue(np.allclose(individual.cov, np.eye(4)))
        tmp = np.random.uniform(size=(4, 4))
        individual.cov[:, :] = tmp
        self.assertTrue(np.allclose(individual.cov, tmp))
        self.assertTrue(individual.step_size == 1.0)
        self.assertTrue(individual.p_succ == 0.5)


class TestArchive(unittest.TestCase):
    def test_creation(self) -> None:
        """Test creating an empty archive."""
        parameters = UPMOParameters(10, 1.0)
        archive = UPMOArchive(parameters)
        self.assertTrue(archive.empty, "Empty")
        self.assertTrue(archive.left_exterior is None, "Left exterior")
        self.assertTrue(archive.right_exterior is None, "Right exterior")
        self.assertTrue(archive.size == 0, "Size")

    def test_sampling(self) -> None:
        """Test sampling exterior and interior points."""
        # m: number of samples used to determine the empirical probabilities
        m = 100000
        # n: number of points
        n = np.random.default_rng().integers(7, 11, 1, dtype=int)[0]
        n_f = float(n)
        # points = {(1, n), (2, n-1), ..., (n, 1)}
        points = np.array(
            list(zip(range(1, n + 1), range(n, 0, -1))), dtype=float
        )
        np.random.shuffle(points)

        parameters = UPMOParameters(points.shape[1], 1.0)
        archive = UPMOArchive(parameters)
        for point in points:
            archive.insert(point, point)

        self.assertTrue(
            np.all(
                np.allclose(
                    archive.left_exterior.fitness, np.array([1.0, n_f])
                )
            ),
            "Left exterior: {}".format(archive.left_exterior.fitness),
        )
        self.assertTrue(
            np.allclose(archive.right_exterior.fitness, np.array([n_f, 1.0])),
            "Right exterior {}".format(archive.right_exterior.fitness),
        )
        sorted_contributions = np.ones(n)
        sorted_contributions[-1] = np.finfo(float).max
        sorted_contributions[-2] = np.finfo(float).max
        output_contributions = np.array(
            sorted(map(lambda individual: individual.contribution, archive))
        )
        self.assertTrue(
            np.allclose(sorted_contributions, output_contributions),
            "Sorted contributions: Got\n{}\nExpected:{}\n".format(
                output_contributions,
                sorted_contributions,
            ),
        )
        max_acc_contributions = max(
            map(lambda individual: individual.acc_contribution, archive)
        )
        self.assertTrue(
            max_acc_contributions == (n_f - 2.0),
            "Max. cumulative contribution",
        )

        # Exterior empirical probabilities
        counts = np.zeros(n)
        for p in np.random.default_rng().uniform(size=m):
            sample = archive.sample_exterior(p)
            i = int(sample.coord(0)) - 1
            counts[i] += 1.0
        empirical_ps = np.round(counts / float(m), decimals=1)
        ps = np.zeros(n)
        ps[0] = 0.5
        ps[-1] = 0.5
        self.assertTrue(
            np.allclose(ps, empirical_ps, atol=1e-1),
            "Exterior empiricals, got: {}, expected: {}".format(
                empirical_ps, ps
            ),
        )

        # Interior empirical probabilities
        counts = np.zeros(n)
        for p in np.random.default_rng().uniform(size=m):
            sample = archive.sample_interior(p)
            i = int(sample.coord(0)) - 1
            counts[i] += 1.0
        empirical_ps = np.round(counts / float(m), decimals=3)
        ps = np.full((n,), 1.0 / (n_f - 2.0))
        ps[0] = 0.0
        ps[-1] = 0.0
        ps = np.round(ps, decimals=3)
        self.assertTrue(
            np.allclose(ps, empirical_ps, atol=1e-2),
            "Interior empiricals, got: {}, expected: {}".format(
                empirical_ps, ps
            ),
        )

    def test_dominated_points_ref_1(self) -> None:
        """Test input data with dominated points and reference."""
        # Random points generated using one of the utility
        # functions (random_2d_3d_front).
        points = np.array(
            [
                [1.07525383, 9.9420234],
                [9.0063025, 4.34586186],
                [1.07525383, 9.9520234],
                [5.21288155, 8.53380723],
                [4.56317607, 8.90816971],
                [8.01491032, 5.98006794],
                [3.24097153, 9.46023803],
                [8.02491032, 5.98006794],
                [4.56317607, 8.89816971],
                [8.09812306, 5.8668904],
                [9.47977929, 3.18336057],
                [8.15916972, 5.78169088],
                [9.93329032, 1.15314504],
            ]
        )
        ranks, _ = non_dominated_sort(points)
        nadir = np.array([10.0, 10.0])
        # The contributions were generated using the reference implementation
        # by A.P. Guerreiro, available at (https://github.com/apguerreiro/HVC).
        # For example: ```./hvc -P 1 -f 0 -r "10. 10. 1" | sort```
        # In the 2-D case, the z-component of the points is set to zero
        # and the z-component of the reference point is set to one.
        sorted_contribs = np.array(
            [
                0.00690911080401627,
                0.0721753062322654,
                0.12556094880582,
                0.135435028337332,
                0.212503643566556,
                0.365178867638393,
                0.527207157404229,
                0.637018803519581,
                0.679831715378445,
                1.02095415166855,
            ]
        )
        parameters = UPMOParameters(points.shape[1], 1.0)
        archive = UPMOArchive(parameters, nadir)
        for point in points:
            archive.insert(point, point)
        self.assertTrue(archive.size == np.sum(ranks == 1), "Size")
        output_contribs = np.array(
            sorted(map(lambda individual: individual.contribution, archive))
        )
        self.assertTrue(
            np.allclose(sorted_contribs, output_contribs), "Contributions"
        )

    def test_nondominated_points_ref_1(self):
        """Test input data with non-dominated points and reference."""
        _, _, points, reference = random_2d_3d_front(1000)
        sorted_contribs = np.array(sorted(hv.contributions(points, reference)))
        parameters = UPMOParameters(points.shape[1], 1.0)
        archive = UPMOArchive(parameters, reference)
        for point in points:
            archive.insert(point, point)
        self.assertTrue(archive.size == len(points), "Size")
        output_contribs = np.array(
            sorted(map(lambda individual: individual.contribution, archive))
        )
        self.assertTrue(
            np.allclose(sorted_contribs, output_contribs), "Contributions"
        )

    def test_nondominated_points_noref_1(self):
        """Test input data with non-dominated points and no reference."""
        _, _, points, _ = random_2d_3d_front(1000)
        sorted_contribs = np.array(sorted(hv.contributions(points)))
        parameters = UPMOParameters(points.shape[1], 1.0)
        archive = UPMOArchive(parameters)
        for point in points:
            archive.insert(point, point)
        self.assertTrue(archive.size == len(points), "Size")
        output_contribs = np.array(
            sorted(map(lambda individual: individual.contribution, archive))
        )
        self.assertTrue(
            np.allclose(sorted_contribs, output_contribs), "Contributions"
        )

    def test_merge(self):
        """Test merging of archives."""
        _, _, points1, _ = random_2d_3d_front(50)
        _, _, points2, _ = random_2d_3d_front(40)
        _, _, points3, _ = random_2d_3d_front(60)
        parameters = UPMOParameters(points1.shape[1], 1.0)
        archive0 = UPMOArchive(parameters)
        archive1 = UPMOArchive(parameters)
        archive2 = UPMOArchive(parameters)
        archive3 = UPMOArchive(parameters)
        for point in points1:
            archive0.insert(point, point)
            archive1.insert(point, point)
        for point in points2:
            archive0.insert(point, point)
            archive2.insert(point, point)
        for point in points3:
            archive0.insert(point, point)
            archive3.insert(point, point)
        size1 = archive1.size
        archive1.merge(archive2)
        del archive2
        self.assertTrue(archive1.size > size1)
        size1 = archive1.size
        archive1.merge(archive3)
        del archive3
        self.assertTrue(archive1.size > size1)
        for x0, x1 in zip(archive0, archive1):
            p0 = x0.fitness
            p1 = x1.fitness
            c0 = x0.contribution
            c1 = x1.contribution
            self.assertTrue(np.allclose(p0, p1), "{}, {}".format(p0, p1))
            self.assertTrue(math.isclose(c0, c1), "{}, {}".format(c0, c1))
