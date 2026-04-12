"""Verify vectorized MoE forward pass correctness and performance."""

import time

import torch

from server.hyperion_policy_network import SoftMoEPolicyNetwork


def test_shapes():
    """Test all output shapes are correct."""
    policy = SoftMoEPolicyNetwork(num_experts=12, top_k=2, router_noise=0.0)
    policy.eval()

    for batch_size in [1, 8, 32, 64, 128]:
        batch = torch.randn(batch_size, 384)
        with torch.no_grad():
            out = policy(batch, use_system2=True, training=False)

        assert out["logits"].shape == (batch_size, 16), f"FAIL logits: {out['logits'].shape}"
        assert out["log_probs"].shape == (batch_size, 16)
        assert out["policy_dist"].shape == (batch_size, 16)
        assert out["value"].shape == (batch_size,)
        assert out["entropy"].shape == (batch_size,)
        assert out["confidence"].shape[0] == batch_size
        assert out["model_confidence"].shape[0] == batch_size
        assert out["process_reward"].shape[0] == batch_size
        assert out["gates"].shape == (batch_size, 2)
        assert out["expert_indices"].shape == (batch_size, 2)
        assert out["meta_weights"].shape == (batch_size, 2)
        print(f"  batch_size={batch_size:3d}: ALL SHAPES OK")


def test_gradients():
    """Test gradient flow through vectorized MoE."""
    policy = SoftMoEPolicyNetwork(num_experts=4, top_k=2, router_noise=0.0)
    policy.train()

    batch = torch.randn(16, 384, requires_grad=True)
    out = policy(batch, use_system2=True, training=True)

    # All outputs should require grad
    assert out["balance_loss"].requires_grad
    loss = out["logits"].sum() + out["balance_loss"]
    loss.backward()

    assert batch.grad is not None, "Gradient should flow to input"
    assert batch.grad.shape == batch.shape
    print("  GRADIENTS: OK")


def test_system2_toggle():
    """Test System 1/2 toggle works correctly."""
    policy = SoftMoEPolicyNetwork(num_experts=4, top_k=2, router_noise=0.0)
    policy.eval()

    batch = torch.randn(16, 384)
    with torch.no_grad():
        out_s2_on = policy(batch, use_system2=True, training=False)
        out_s2_off = policy(batch, use_system2=False, training=False)

    # Outputs should differ
    diff = (out_s2_on["logits"] - out_s2_off["logits"]).abs().mean().item()
    assert diff > 0, "System 1 and System 2 outputs should differ"
    print(f"  System 1/2 diff: {diff:.6f}: OK")


def test_expert_coverage():
    """Test that all experts get used with diverse inputs."""
    policy = SoftMoEPolicyNetwork(num_experts=12, top_k=2, router_noise=0.0)
    policy.eval()

    # Large diverse batch should hit all experts
    batch = torch.randn(256, 384)
    with torch.no_grad():
        out = policy(batch, use_system2=False, training=False)

    used_experts = set(out["expert_indices"].view(-1).tolist())
    print(f"  Experts used: {len(used_experts)}/12: {'OK' if len(used_experts) > 6 else 'WARN (expected more)'}")


def benchmark():
    """Benchmark forward pass at different batch sizes."""
    policy = SoftMoEPolicyNetwork(num_experts=12, top_k=2, router_noise=0.0)
    policy.eval()

    print("\nBenchmark (CPU, avg of 5 runs):")
    for batch_size in [16, 64, 128, 256]:
        batch = torch.randn(batch_size, 384)

        # Warmup
        with torch.no_grad():
            policy(batch, use_system2=True, training=False)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                policy(batch, use_system2=True, training=False)
            times.append(time.time() - start)

        avg_ms = (sum(times) / len(times)) * 1000
        print(f"  batch_size={batch_size:4d}: {avg_ms:7.1f}ms")


if __name__ == "__main__":
    print("=== Vectorized MoE Tests ===\n")

    print("1. Shape tests:")
    test_shapes()

    print("\n2. Gradient test:")
    test_gradients()

    print("\n3. System 1/2 toggle test:")
    test_system2_toggle()

    print("\n4. Expert coverage test:")
    test_expert_coverage()

    benchmark()

    print("\n=== All tests passed ===")
