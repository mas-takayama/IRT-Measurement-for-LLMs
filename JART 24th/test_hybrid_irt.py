"""Run with: python -m unittest -v test_hybrid_irt.py (from this directory)."""
import unittest

import numpy as np
from scipy.optimize import check_grad

import hybrid_irt as h


class HybridIRTTests(unittest.TestCase):
    def setUp(self):
        self.sim = h.simulate_itemwise_mixed_dataframes(
            n_full_probability=8, n_full_binary=9, n_mixed=10,
            n_items=12, R_l=5, R_b=5, seed=42,
        )

    def test_shared_theta_objective_is_sum_of_both_channels(self):
        s = self.sim
        a, b, th = s.a_true, s.b_true, s.theta_mix_true[0]
        p = s.p_mix_df.to_numpy()[0]
        v = h.make_variance_from_s2(s.s2_mix_df.to_numpy(), 5)[0]
        r = s.r_mix_df.to_numpy()[0]
        prob = h.probability_theta_objective(th, p, v, a, b)
        binary = h.binary_theta_objective(th, r, 5, a, b)
        both = h.mixed_theta_objective(th, p, v, r, np.ones(12), 5, a, b)
        self.assertAlmostEqual(both, prob + binary, places=10)
        self.assertAlmostEqual(h.mixed_theta_objective(th, p, v, r, np.zeros(12), 5, a, b), binary)
        self.assertAlmostEqual(h.mixed_theta_objective(th, p, v, np.full(12, np.nan), np.ones(12), 5, a, b), prob)

    def test_item_gradient_includes_both_channels(self):
        s = self.sim
        j = 3
        args = (s.p_full_df.to_numpy()[:, j], h.make_variance_from_s2(s.s2_full_df.to_numpy(), 5)[:, j],
                s.r_bin_df.to_numpy()[:, j], s.p_mix_df.to_numpy()[:, j],
                h.make_variance_from_s2(s.s2_mix_df.to_numpy(), 5)[:, j],
                s.r_mix_df.to_numpy()[:, j], np.ones(10), 5,
                s.theta_full_true, s.theta_bin_true, s.theta_mix_true)
        z = np.array([np.log(s.a_true[j]), s.b_true[j]])
        error = check_grad(lambda x: h.item_objective_and_grad_mixed(x, *args)[0],
                           lambda x: h.item_objective_and_grad_mixed(x, *args)[1], z)
        self.assertLess(error, 1e-4)

    def test_shared_theta_information_adds(self):
        s = self.sim
        empty = np.empty((0, 12))
        p, r = s.p_mix_df.to_numpy(), s.r_mix_df.to_numpy()
        v = h.make_variance_from_s2(s.s2_mix_df.to_numpy(), 5)
        args = (empty, empty, empty, p, v, r, np.ones_like(p), 5,
                s.a_true, s.b_true, np.array([]), np.array([]), s.theta_mix_true)
        both = h.compute_se_mixed(*args)[-1]
        prob_args = list(args)
        prob_args[5] = np.full_like(r, np.nan, dtype=float)
        prob = h.compute_se_mixed(*prob_args)[-1]
        binary_args = list(args)
        binary_args[6] = np.zeros_like(p)
        binary = h.compute_se_mixed(*binary_args)[-1]
        np.testing.assert_allclose(1 / both**2, 1 / prob**2 + 1 / binary**2)

    def test_generator_endpoints_and_common_random_numbers(self):
        kwargs = dict(n_full_probability=0, n_full_binary=0, n_mixed=10, n_items=12, seed=8)
        first = h.simulate_itemwise_mixed_dataframes(binary_fraction=0, **kwargs)
        last = h.simulate_itemwise_mixed_dataframes(binary_fraction=1, **kwargs)
        self.assertTrue((first.mask_mix_df.to_numpy() == 1).all())
        self.assertTrue((last.mask_mix_df.to_numpy() == 0).all())
        np.testing.assert_array_equal(first.r_mix_df, last.r_mix_df)
        np.testing.assert_array_equal(first.p_mix_df, last.p_mix_df)

    def test_four_scenarios_and_fitted_group_alignment(self):
        for scenario, flags in h.SCENARIOS.items():
            with self.subTest(scenario=scenario):
                fit = h.fit_simulation(self.sim, scenario, outer_iter=3, item_maxiter=30)
                expected = [n if use else 0 for n, use in zip((8, 9, 10), flags)]
                truth = h.fitted_truth(fit, self.sim)
                self.assertEqual([len(x) for x in truth[2:]], expected)
                np.testing.assert_allclose(np.concatenate(truth[2:]).mean(), 0, atol=1e-14)
                np.testing.assert_allclose(np.concatenate(truth[2:]).std(), 1)
                summary = h.summarize_result(fit, self.sim)
                self.assertTrue(np.isfinite(summary['rmse_a']))
                self.assertTrue(np.isfinite(summary['rmsse_a']))
                items, people = h.parameter_tables(fit, self.sim)
                self.assertEqual(len(items), 12)
                self.assertEqual(len(people), sum(expected))
                self.assertFalse(fit['converged'])
                self.assertEqual(fit['stop_reason'], 'max_iter_reached')

    def test_identification_preserves_item_response_function(self):
        s = self.sim
        a, b, tp, tb, tm = h.standardize_params_mixed(
            s.a_true, s.b_true, s.theta_full_true, s.theta_bin_true, s.theta_mix_true)
        for before, after in zip((s.theta_full_true, s.theta_bin_true, s.theta_mix_true), (tp, tb, tm)):
            np.testing.assert_allclose(h.D*s.a_true*(before[:, None]-s.b_true), h.D*a*(after[:, None]-b))

    def test_missing_probability_falls_back_to_binomial(self):
        s = self.sim
        p = np.full((10, 12), np.nan)
        fit = h.fit_hybrid_irt_with_itemwise_mixed(
            p_mix=p, s2_mix=p, r_mix=s.r_mix_df, mask_mix=np.ones_like(p),
            outer_iter=3, verbose=False,
        )
        self.assertTrue(np.isfinite(fit['a']).all())
        self.assertTrue(np.isfinite(fit['theta_mixed']).all())

    def test_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            h.simulate_itemwise_mixed_dataframes(R_l=1)
        with self.assertRaises(ValueError):
            h.fit_hybrid_irt_with_itemwise_mixed(r_bin=np.full((3, 2), np.nan), verbose=False)
        with self.assertRaises(ValueError):
            h.fit_hybrid_irt_with_itemwise_mixed(r_bin=np.full((3, 2), 5.5), verbose=False)
        with self.assertRaises(ValueError):
            h.fit_hybrid_irt_with_itemwise_mixed(p_full=np.full((3, 2), .5), s2_full=np.zeros((3, 2)), c_tau=0, verbose=False)
        with self.assertRaises(ValueError):
            h.fit_hybrid_irt_with_itemwise_mixed(r_bin=np.ones((3, 2)), outer_iter=0, verbose=False)

    def test_mixed_item_experiment_really_contains_mixed_group(self):
        raw, summary = h.run_simulation_grid(
            'items_mixed', values=[10], seeds=[42],
            fit_options={'outer_iter': 2, 'item_maxiter': 20}, progress=False,
        )
        self.assertEqual(raw.loc[0, 'n_fitted_mixed'], 50)
        self.assertGreater(raw.loc[0, 'objective_mix_prob'], 0)
        self.assertGreater(raw.loc[0, 'objective_mix_bin'], 0)
        self.assertEqual(summary.loc[0, 'n_rep'], 1)
        self.assertTrue(np.isnan(summary.loc[0, 'rmsse_a_se']))


if __name__ == '__main__':
    unittest.main()
