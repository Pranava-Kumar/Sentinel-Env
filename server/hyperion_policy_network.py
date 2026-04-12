"""HyperionRL Soft MoE Policy Network.

Implements the 12-expert soft mixture-of-experts architecture with dual-process
(System 1 / System 2) reasoning, auxiliary heads, and SCALE selective compute.

Based on:
- Soft MoE (arXiv 2402.08609) - Soft routing with harmonious balance loss
- SCALE (arXiv 2512.00466) - Dual-process compute allocation
- MoW (arXiv 2602.01270) - Mixture-of-World Models routing
"""

from typing import Any, ClassVar

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = structlog.get_logger()

# 16 threat categories from models.py
NUM_CLASSES = 16


class SoftRouter(nn.Module):
    """Soft top-2 routing for MoE experts.

    Uses softmax with temperature scheduling and auxiliary balance loss
    to prevent expert collapse. Inspired by Soft MoE (arXiv 2402.08609).
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 12,
        top_k: int = 2,
        noise_scale: float = 0.5,
    ):
        """Initialize soft router.

        Args:
            input_dim: Input feature dimension.
            num_experts: Total number of experts.
            top_k: Number of top experts to combine (soft routing).
            noise_scale: Initial exploration noise scale (decays during training).
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_scale = noise_scale

        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts),
        )

    def set_noise_scale(self, scale: float) -> None:
        """Update exploration noise scale.

        Args:
            scale: New noise scale (0.05 to 0.5 recommended range).
        """
        self.noise_scale = scale

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute soft routing weights.

        Args:
            x: Input features of shape (batch, input_dim).
            training: Whether in training mode (adds noise).

        Returns:
            Tuple of (gates, top_k_indices, balance_loss).
            - gates: Soft routing weights (batch, top_k).
            - top_k_indices: Expert indices (batch, top_k).
            - balance_loss: Auxiliary loss to prevent expert collapse.
        """
        logits = self.router(x)  # (batch, num_experts)

        if training and self.noise_scale > 0.01:
            # Add Gaussian noise for exploration
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise

        # Soft top-k routing
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(top_k_values, dim=-1)  # (batch, top_k)

        # Expert balance loss (prevent collapse)
        # Mean routing probability per expert should be 1/num_experts
        full_probs = F.softmax(logits, dim=-1)  # (batch, num_experts)
        mean_probs = full_probs.mean(dim=0)  # (num_experts,)
        target_prob = 1.0 / self.num_experts
        balance_loss = F.mse_loss(mean_probs, torch.full_like(mean_probs, target_prob))

        return gates, top_k_indices, balance_loss


class MoEExpert(nn.Module):
    """Single MoE expert: 3-layer MLP with LayerNorm + GELU.

    Each expert specializes in a different threat pattern.
    Architecture: 256 -> 128 -> 64 -> num_classes
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = NUM_CLASSES,
        expert_id: int = 0,
    ):
        """Initialize MoE expert.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (num_classes).
            expert_id: Expert identifier for logging.
        """
        super().__init__()
        self.expert_id = expert_id
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert.

        Args:
            x: Input features.

        Returns:
            Expert output logits.
        """
        return self.network(x)


class System1Fast(nn.Module):
    """System 1: Fast Intuition - lightweight feedforward.

    Pattern matching based on MoE features. Provides rapid threat assessment.
    Architecture: 256 -> 128 -> 16 classes.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        """Initialize System 1.

        Args:
            input_dim: Input feature dimension (MoE output).
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fast classification.

        Args:
            x: Input features from MoE layer.

        Returns:
            Dict with logits, features, and confidence.
        """
        features = self.network[:3](x)
        logits = self.network[3:](features)
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        return {
            "logits": logits,
            "features": features,
            "confidence": confidence,
        }


class System2Slow(nn.Module):
    """System 2: Slow Deliberation - GRU with 3 thought steps.

    Deeper reasoning through recurrent "thought steps" that verify
    the initial classification. Activates for complex/hard cases.

    Architecture: GRU(256+16, 256, 2 layers) -> verification head.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_thoughts: int = 3,
    ):
        """Initialize System 2.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: GRU hidden dimension.
            num_thoughts: Number of deliberation steps.
        """
        super().__init__()
        self.num_thoughts = num_thoughts
        self.hidden_dim = hidden_dim

        # GRU takes concatenated features + initial prediction
        self.thought_rnn = nn.GRU(
            input_size=input_dim + NUM_CLASSES,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Verification head
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_CLASSES),
        )

    def forward(
        self,
        x: torch.Tensor,
        initial_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Deliberative reasoning through thought steps.

        Args:
            x: Input features from MoE layer.
            initial_logits: Initial classification from System 1.

        Returns:
            Dict with verified_logits, thoughts, and confidence.
        """
        _batch_size = x.shape[0]

        # Initialize thought sequence with embedding + initial prediction
        initial_probs = F.softmax(initial_logits, dim=-1)
        thought_input = torch.cat([x, initial_probs], dim=-1)  # (batch, 256+16)
        thought_input = thought_input.unsqueeze(1).expand(-1, self.num_thoughts, -1)

        # RNN reasoning
        thoughts, _ = self.thought_rnn(thought_input)  # (batch, num_thoughts, hidden)

        # Take last thought
        final_thought = thoughts[:, -1, :]  # (batch, hidden)

        # Verification
        verified_logits = self.verification_head(final_thought)
        confidence = F.softmax(verified_logits, dim=-1).max(dim=-1).values

        return {
            "verified_logits": verified_logits,
            "thoughts": thoughts,
            "confidence": confidence,
        }


class SoftMoEPolicyNetwork(nn.Module):
    """HyperionRL Policy Network with Soft MoE and Dual-Process Architecture.

    Combines 12 specialized experts with soft routing, System 1 fast intuition,
    System 2 slow deliberation, and auxiliary heads for value, process reward,
    and confidence estimation.

    Architecture:
        Input (384-dim) -> Input Proj (256-dim) -> Soft MoE (12 experts, top-2)
        -> System 1 (fast) + System 2 (slow) -> Meta-weighting -> Output

    Auxiliary Outputs:
        - Value network (PPO baseline)
        - Process reward head (step-level quality)
        - Confidence head (prediction uncertainty)
    """

    # Expert specialization names (for interpretability)
    EXPERT_NAMES: ClassVar[list[str]] = [
        "injection-pattern",
        "social-engineering",
        "stealth-exfiltration",
        "reasoning-quality",
        "calibration",
        "curiosity-novelty",
        "command-injection",
        "jailbreak-detection",
        "authority-impersonation",
        "urgency-manipulation",
        "context-manipulation",
        "encoded-payload",
    ]

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        num_experts: int = 12,
        top_k: int = 2,
        num_thoughts: int = 3,
        router_noise: float = 0.5,
    ):
        """Initialize SoftMoE policy network.

        Args:
            embedding_dim: Input text embedding dimension (384).
            hidden_dim: Hidden layer dimension.
            num_experts: Number of MoE experts.
            top_k: Number of top experts for soft routing.
            num_thoughts: System 2 deliberation steps.
            router_noise: Initial router exploration noise.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        # Input projection: 384 -> 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Soft MoE layer: 12 experts with soft routing
        self.router = SoftRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            noise_scale=router_noise,
        )
        self.experts = nn.ModuleList(
            [
                MoEExpert(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim // 2,
                    output_dim=hidden_dim // 4,
                    expert_id=i,
                )
                for i in range(num_experts)
            ]
        )

        # MoE output projection (64 -> 256)
        self.moe_output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # System 1: Fast intuition
        self.system1 = System1Fast(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
        )

        # System 2: Slow deliberation
        self.system2 = System2Slow(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_thoughts=num_thoughts,
        )

        # Meta-weighting: learns to balance System 1 vs System 2
        self.meta_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

        # Auxiliary heads (on top of MoE features)
        # Value network for PPO baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Process reward head (sigmoid, step-level quality)
        self.process_reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Confidence head (prediction uncertainty)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + NUM_CLASSES, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def set_router_noise(self, scale: float) -> None:
        """Update router exploration noise.

        Args:
            scale: New noise scale.
        """
        self.router.set_noise_scale(scale)

    def forward(
        self,
        x: torch.Tensor,
        use_system2: bool = True,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through SoftMoE policy.

        Args:
            x: Input embeddings of shape (batch, 384).
            use_system2: Whether to run System 2 deliberation.
            training: Whether in training mode (adds noise).

        Returns:
            Dict containing:
                - logits: Final 16-class prediction logits.
                - log_probs: Log probabilities of predictions.
                - policy_dist: Full softmax distribution.
                - value: Value estimate for PPO baseline.
                - entropy: Policy entropy for regularization.
                - confidence: Prediction confidence (0-1).
                - process_reward: Process reward estimate (0-1).
                - gates: MoE routing weights.
                - expert_indices: Selected expert indices.
                - balance_loss: Auxiliary expert balance loss.
                - meta_weights: System 1 vs System 2 weights.
        """
        batch_size = x.shape[0]

        # Input projection
        h = self.input_projection(x)  # (batch, 256)

        # Soft MoE routing
        gates, expert_indices, balance_loss = self.router(h, training=training)

        # =========================================================================
        # Vectorized MoE forward pass using batch-by-expert strategy.
        #
        # Instead of looping over batch samples (defeating GPU parallelism),
        # we group samples by their assigned experts and process all samples
        # for each expert in a single batched operation. This enables full
        # GPU utilization through parallel matrix multiplications.
        #
        # Strategy:
        # 1. Initialize output tensor
        # 2. For each expert, find all samples routed to it (across all top-k positions)
        # 3. Process all those samples in one batched forward pass through the expert
        # 4. Weight by corresponding gate values and scatter back to correct positions
        #
        # This reduces Python loop iterations from (batch_size * top_k) to num_experts,
        # typically yielding 10-50x speedup on GPU for batch_size=64, top_k=2.
        # Numerically equivalent to the sequential version within floating-point tolerance.
        # =========================================================================
        moe_output = torch.zeros(batch_size, self.hidden_dim // 4, device=x.device)

        for expert_id in range(self.num_experts):
            # Find all samples that use this expert (at any top-k position)
            # expert_mask shape: (batch, top_k) where True means this expert is selected
            expert_mask = expert_indices == expert_id

            # Get sample indices that use this expert
            sample_mask = expert_mask.any(dim=1)  # (batch,)
            if not sample_mask.any():
                continue  # No samples use this expert, skip

            # Get the actual indices of selected samples
            selected_indices = sample_mask.nonzero(as_tuple=False).squeeze(1)  # (num_selected,)

            # For each selected sample, determine which top-k position this expert was at
            # and get the corresponding gate value
            selected_masks = expert_mask[selected_indices]  # (num_selected, top_k)

            # Get input features for selected samples
            selected_h = h[selected_indices]  # (num_selected, hidden_dim)

            # Process all selected samples through this expert in a single batched operation
            # This is the key vectorization: instead of batch_size * top_k sequential calls,
            # we have at most num_experts batched calls (12 vs 128 for batch=64, top_k=2)
            expert_output = self.experts[expert_id](selected_h)  # (num_selected, output_dim)

            # Get gate values for this expert at each top-k position
            # Sum gates if expert appears at multiple positions (unlikely with top-k, but safe)
            gate_values = (selected_masks.float() * gates[selected_indices]).sum(
                dim=1, keepdim=True
            )  # (num_selected, 1)

            # Weight expert output by gate values
            weighted_output = gate_values * expert_output  # (num_selected, output_dim)

            # Scatter back to the correct positions in moe_output
            moe_output.index_add_(0, selected_indices, weighted_output)

        # Project MoE output back to hidden_dim
        moe_features = self.moe_output_proj(moe_output)  # (batch, 256)

        # System 1: Fast intuition
        s1_output = self.system1(moe_features)

        # Meta-weighting
        weights = self.meta_weights(moe_features)  # (batch, 2)
        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]

        if use_system2:
            # System 2: Slow deliberation
            s2_output = self.system2(moe_features, s1_output["logits"])

            # Combine systems
            combined_logits = w1 * s1_output["logits"] + w2 * s2_output["verified_logits"]
            confidence = w1 * s1_output["confidence"] + w2 * s2_output["confidence"]
        else:
            combined_logits = s1_output["logits"]
            confidence = s1_output["confidence"]

        # Compute outputs
        log_probs = F.log_softmax(combined_logits, dim=-1)
        policy_dist = F.softmax(combined_logits, dim=-1)
        entropy = -(policy_dist * log_probs).sum(dim=-1)

        value = self.value_head(moe_features).squeeze(-1)
        process_reward = self.process_reward_head(moe_features).squeeze(-1)

        conf_input = torch.cat([moe_features, policy_dist], dim=-1)
        model_confidence = self.confidence_head(conf_input).squeeze(-1)

        return {
            "logits": combined_logits,
            "log_probs": log_probs,
            "policy_dist": policy_dist,
            "value": value,
            "entropy": entropy,
            "confidence": confidence,
            "model_confidence": model_confidence,
            "process_reward": process_reward,
            "gates": gates,
            "expert_indices": expert_indices,
            "balance_loss": balance_loss,
            "meta_weights": weights,
        }

    def get_expert_usage_stats(self) -> dict[str, Any]:
        """Return expert usage statistics for monitoring.

        Returns:
            Dict with expert names and their average routing weights.
        """
        with torch.no_grad():
            router_weights = self.router.router.weight.data  # type: ignore[attr-defined]  # (12, 256)
            norms = router_weights.norm(dim=1)  # type: ignore[operator]  # (12,)
            total = norms.sum()
            proportions: list[float] = (
                (norms / total).tolist() if total > 0 else [1.0 / self.num_experts] * self.num_experts
            )

        return {name: round(prop, 4) for name, prop in zip(self.EXPERT_NAMES, proportions, strict=False)}

    def get_scale_decision(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[bool]:
        """Determine whether to use System 2 per sample (SCALE).

        Args:
            x: Input embeddings.
            threshold: Meta-weight threshold for System 2 activation.

        Returns:
            List of booleans indicating System 2 usage.
        """
        with torch.no_grad():
            h = self.input_projection(x)
            weights = self.meta_weights(h)
            use_s2 = (weights[:, 1] > threshold).tolist()
        return use_s2
