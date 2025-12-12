import pytest
import jax.numpy as jnp
import jax
import flowgym.flow.consensus.admm as consensus_mod


@pytest.fixture
def dummy_solvers(monkeypatch):
    # Dummy "solvers" just pass through the first argument
    def dummy_optimize_flows(
        flows, consensus_flow, consensus_dual, obj_fn, rho, num_iters
    ):
        # Simply increment all elements to test the loop
        return flows + 1

    def dummy_optimize_consensus(
        flows, consensus_flow, consensus_dual, obj_fn, rho, num_iters
    ):
        # Return the mean across agent dimension
        return jnp.mean(flows, axis=0)

    # Patch the global mappings
    monkeypatch.setattr(
        "flowgym.flow.consensus.admm.SOLVER_FLOWS_FACTORY",
        {"dummy": lambda lr: dummy_optimize_flows},
    )
    monkeypatch.setattr(
        "flowgym.flow.consensus.admm.SOLVER_CONSENSUS_FACTORY",
        {"dummy": lambda lr: dummy_optimize_consensus},
    )


def dummy_obj(*args, **kwargs):
    # Always returns a scalar, doesn't matter for this test
    return 0.0


@pytest.mark.usefixtures("dummy_solvers")
@pytest.mark.parametrize("N, H, W, C", [(2, 3, 4, 1)])
def test_run_admm_shape_and_basic_flow(N, H, W, C):
    """Test that run_admm outputs the expected shape and calls the solvers."""
    flows = jnp.ones((N, H, W, C))
    consensus, _ = consensus_mod.run_admm(
        flows=flows,
        rho=1.0,
        objective_fn_flows=dummy_obj,
        num_iterations_flows=2,
        solver_flows="dummy",
        objective_fn_z=dummy_obj,
        num_iterations_consensus=1,
        solver_consensus="dummy",
        max_admm_iterations=4,
    )
    # The result shape should match the initial consensus_flow: (H, W, C)
    assert consensus.shape == (H, W, C)


@pytest.mark.parametrize("max_iters", [1, 2, 5])
@pytest.mark.parametrize("N, H, W, C", [(2, 2, 2, 1)])
def test_run_admm_runs_correct_number_of_iterations(max_iters, N, H, W, C):
    """Test that run_admm performs exactly max_admm_iterations."""
    flows = jnp.ones((N, H, W, C))

    # We'll count how many times the solvers are called via closure
    flow_calls = []
    consensus_calls = []

    def dummy_flows_factory(lr):
        def dummy_flows_solver(flows_in, *args, **kwargs):
            flow_calls.append(1)
            return flows_in

        return dummy_flows_solver

    def dummy_consensus_factory(lr):
        def dummy_consensus_solver(flows, *args, **kwargs):
            consensus_calls.append(1)
            return jnp.mean(flows, axis=1)

        return dummy_consensus_solver

    consensus_mod.SOLVER_FLOWS_FACTORY["dummy"] = dummy_flows_factory
    consensus_mod.SOLVER_CONSENSUS_FACTORY["dummy"] = dummy_consensus_factory

    # Jit is disabled to ensure we can count calls in a straightforward way
    with jax.disable_jit():
        consensus_mod.run_admm(
            flows=flows,
            rho=1.0,
            objective_fn_flows=dummy_obj,
            num_iterations_flows=1,
            solver_flows="dummy",
            objective_fn_z=dummy_obj,
            num_iterations_consensus=1,
            solver_consensus="dummy",
            max_admm_iterations=max_iters,
            eps_abs=None,
            eps_rel=None,
        )

    # We expect flow_calls and consensus_calls == max_iters
    assert len(flow_calls) == max_iters
    assert len(consensus_calls) == max_iters


@pytest.mark.usefixtures("dummy_solvers")
@pytest.mark.parametrize("N, H, W, C", [(3, 2, 2, 1)])
def test_run_admm_returns_consensus_mean_when_solvers_do_nothing(N, H, W, C):
    """solvers return unchanged input, consensus_flow is the mean of original flows."""
    flows = jnp.arange(N * H * W * C).reshape(N, H, W, C).astype(jnp.float32)

    # Solvers just return inputs, no changes
    def no_op_solver(*args, **kwargs):
        return args[0]

    consensus_mod.SOLVER_FLOWS_FACTORY["noop"] = lambda lr: no_op_solver
    consensus_mod.SOLVER_CONSENSUS_FACTORY["noop"] = (
        lambda lr: lambda flows, *a, **k: jnp.mean(flows, axis=0)
    )
    consensus, _ = consensus_mod.run_admm(
        flows=flows,
        rho=1.0,
        objective_fn_flows=dummy_obj,
        num_iterations_flows=1,
        solver_flows="noop",
        objective_fn_z=dummy_obj,
        num_iterations_consensus=1,
        solver_consensus="noop",
        max_admm_iterations=2,
    )
    assert jnp.allclose(consensus, jnp.mean(flows, axis=0))
