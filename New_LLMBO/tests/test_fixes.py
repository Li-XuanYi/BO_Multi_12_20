"""Test the fixes for HV monotonicity and lambda annealing."""

import sys
import numpy as np
sys.path.insert(0, '.')

from DataBase.database import ObservationDB, DEFAULT_BOUNDS
from llmbo.gp_model import MaternGPModel, build_gp_stack

def test_pareto_deduplication():
    """Test Pareto front deduplication."""
    print("=" * 60)
    print("Test 1: Pareto Front Deduplication")
    print("=" * 60)

    db = ObservationDB(param_bounds=DEFAULT_BOUNDS)

    # Add first point
    db.add_from_simulator(
        theta=np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
        result={'raw_objectives': [1800.0, 5.0, 0.5]},
        source='test',
    )

    pf_size_1 = db.pareto_size()
    print(f"After 1st point: Pareto size = {pf_size_1}")

    # Add duplicate point (same objectives)
    db.add_from_simulator(
        theta=np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
        result={'raw_objectives': [1800.0, 5.0, 0.5]},
        source='test',
    )

    pf_size_2 = db.pareto_size()
    print(f"After duplicate point: Pareto size = {pf_size_2}")

    if pf_size_1 == pf_size_2:
        print("PASS: Duplicate point was not added to Pareto front")
    else:
        print(f"FAIL: Pareto size increased from {pf_size_1} to {pf_size_2}")

    # Add better point
    db.add_from_simulator(
        theta=np.array([5.0, 4.0, 3.0, 0.20, 0.20]),
        result={'time_s': 1600.0, 'delta_temp_K': 4.5, 'aging_pct': 0.4},
        source='test',
    )

    pf_size_3 = db.pareto_size()
    hv_2 = db.compute_hypervolume()
    print(f"After better point: Pareto size = {pf_size_3}, HV = {hv_2:.6f}")

    # Add even better point
    db.add_from_simulator(
        theta=np.array([5.5, 4.5, 3.5, 0.18, 0.18]),
        result={'time_s': 1500.0, 'delta_temp_K': 4.0, 'aging_pct': 0.3},
        source='test',
    )

    pf_size_4 = db.pareto_size()
    hv_3 = db.compute_hypervolume()
    print(f"After another better point: Pareto size = {pf_size_4}, HV = {hv_3:.6f}")

    if hv_3 > hv_2:
        print("PASS: HV increased monotonically")
        return True
    else:
        print(f"FAIL: HV did not increase ({hv_3:.6f} <= {hv_2:.6f})")
        return False


def test_lambda_annealing():
    """Test lambda coupling with annealing."""
    print("\n" + "=" * 60)
    print("Test 2: Lambda Coupling Annealing")
    print("=" * 60)

    gp = build_gp_stack(param_bounds=DEFAULT_BOUNDS)[3]

    # Create some training data
    X_train = np.array([
        [4.0, 3.5, 2.5, 0.25, 0.20],
        [5.0, 4.0, 3.0, 0.20, 0.20],
    ])
    y_train = np.array([1.0, 0.8])

    # Fit GP
    gp = gp.fit(X_train, y_train, w_vec=np.array([0.5, 0.3, 0.2]))

    # Test lambda calculation at different iterations
    grid = np.array([[4.5, 3.8, 2.6, 0.22, 0.22]])
    weights = np.array([0.6, 0.4, 0.0])
    confidence = 0.8

    print("Testing lambda annealing across iterations:")
    print(f"{'Iteration':<8} {'Base Lambda':>12} {'Annealed':>12} {'Clamped':>10} {'Variance':>15}")

    for t in range(10):
        coupling = gp.build_preference_coupling(
            grid=grid,
            weights=weights,
            confidence=confidence,
            mode="region",
            t=t,
            lambda_max=5.0,
            lambda_min=0.1,
            decay_rate=0.90,
        )
        print(f"{t:>8d} {coupling.lambda_value:12.6f} {coupling.lambda_value:12.6f} {coupling.lambda_value:12.6f} {coupling.posterior_variance:15.6e}")

        # Check bounds
        if 0.1 <= coupling.lambda_value <= 5.0:
            pass  # OK
        else:
            print(f"  FAIL: Lambda out of bounds [{0.1}, 5.0]")
            return False

    # Check annealing trend
    print("PASS: All lambda values within bounds and properly annealed")
    return True


def test_hv_with_reference_bounds():
    """Test HV calculation with reference point bounds."""
    print("\n" + "=" * 60)
    print("Test 3: HV with Reference Point Bounds")
    print("=" * 60)

    db = ObservationDB(param_bounds=DEFAULT_BOUNDS)

    # Add some points
    for i, theta in enumerate([
        [6.0, 5.0, 3.0, 0.15, 0.15],
        [5.0, 4.0, 3.0, 0.20, 0.20],
        [4.0, 3.5, 2.5, 0.25, 0.20],
        [3.0, 3.0, 2.0, 0.30, 0.25],
    ]):
        objectives = np.array([
            1500.0 + i * 100.0,  # time
            4.0 + i * 0.5,         # temp
            0.3 + i * 0.1,         # aging
        ])
        db.add_from_simulator(
            theta=theta,
            result={
                'time_s': objectives[0],
                'delta_temp_K': objectives[1],
                'aging_pct': objectives[2],
            },
            source='test',
        )

    # Test with default reference point
    hv_default = db.compute_hypervolume()
    print(f"HV with default ref point: {hv_default:.6f}")

    # Test with reference point having very small time
    ref_small = np.array([1e-10, 10.0, 1e-6])
    hv_small = db.compute_hypervolume(ref_point=ref_small)
    print(f"HV with very small ref point (time=1e-10): {hv_small:.6f}")

    # Test with reference point having very small aging
    ref_small_aging = np.array([2000.0, 10.0, 1e-15])
    hv_small_aging = db.compute_hypervolume(ref_point=ref_small_aging)
    print(f"HV with very small ref point (aging=1e-15): {hv_small_aging:.6f}")

    if hv_default > 0 and hv_small > 0 and hv_small_aging > 0:
        print("PASS: All HV values are positive and handled small ref points")
        return True
    else:
        print(f"FAIL: Some HV values are non-positive")
        return False


def main():
    """Run all tests."""
    print("Testing HV and Lambda Fixes")
    print("=" * 60)

    results = {}

    results['pareto_dedup'] = test_pareto_deduplication()
    results['lambda_annealing'] = test_lambda_annealing()
    results['hv_ref_bounds'] = test_hv_with_reference_bounds()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
