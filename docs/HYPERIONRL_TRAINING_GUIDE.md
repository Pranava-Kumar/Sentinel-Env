# HyperionRL Training Guide

Complete guide for training AI safety agents with HyperionRL in the Sentinel Environment.

## Overview

HyperionRL is an advanced reinforcement learning agent that synthesizes **12 breakthrough innovations** from 75+ research papers into a unified architecture for jailbreak detection and AI agent safety evaluation.

### Key Achievements

| Metric | Baseline | After 200 Episodes | After 5000 Episodes (Expected) |
|--------|----------|-------------------|-------------------------------|
| Detection Rate | 7.2% | 52.8% | 85-90% |
| False Positive Rate | ~30% | ~15% | <10% |
| Reasoning Quality | N/A | 0.75 | 0.85+ |

### 12 Core Innovations

| # | Innovation | Description | Paper Reference |
|---|------------|-------------|-----------------|
| 1 | **TextEmbedder** | Real 384-dim sentence-transformers embeddings | — |
| 2 | **SoftMoEPolicyNetwork** | 12 experts with soft top-2 routing | arXiv 2402.08609 |
| 3 | **MCTSReasoningTree** | 10-path exploration with process rewards | arXiv 2510.14942 |
| 4 | **iGRPOTrainer** | Iterative self-feedback (draft → select → refine) | arXiv 2602.09000 |
| 5 | **ScaffoldedCurriculum** | Learning cliff solver with progressive hints | arXiv 2510.19807 |
| 6 | **GDPOOptimizer** | Decoupled multi-reward optimization (6 rewards) | arXiv 2601.05242 |
| 7 | **AdversarialSelfPlayV2** | Infinite attack generation | arXiv 2602.00173 |
| 8 | **MemoryConsolidation** | Sleep-like replay of hard cases | — |
| 9 | **PIPO** | Cross-iteration policy improvement verification | arXiv 2604.00860 |
| 10 | **MC-GRPO** | Median-centered advantage normalization | arXiv 2601.22582 |
| 11 | **CDE** | Curiosity-driven exploration | arXiv 2509.09675 |
| 12 | **SCALE** | Selective resource allocation (System 1 vs System 2) | arXiv 2512.00466 |

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch numpy structlog trackio matplotlib

# Optional (for full feature set)
pip install wandb sentry-sdk sentence-transformers
```

### First Training Run

```bash
# Quick test run (5 minutes on CPU)
python train_hyperion.py --episodes 200

# Standard run (30-60 minutes on CPU)
python train_hyperion.py --episodes 1000

# Full training run (2-4 hours on CPU, 30-60 min on GPU)
python train_hyperion.py --episodes 5000 --device cuda
```

### Monitor Training

```bash
# Start visualization dashboard (in new terminal)
python visualize_dashboard.py

# Dashboard auto-refreshes every 10 seconds
# Shows: detection rate, FP rate, reward, loss, curriculum level, etc.
```

---

## Training Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                  HyperionRL Trainer                  │
├─────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │           iGRPOTrainer (Stage 1)             │  │
│  │  ┌─────────────┐  ┌──────────────────────┐  │  │
│  │  │ Sample 8    │→ │ Select Best Draft    │  │  │
│  │  │ Drafts      │  │ (by MC-GRPO reward)  │  │  │
│  │  └─────────────┘  └──────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌──────────────────────────────────────────────┐  │
│  │           iGRPOTrainer (Stage 2)             │  │
│  │  ┌─────────────┐  ┌──────────────────────┐  │  │
│  │  │ Refine Best │→ │ PIPO Verification    │  │  │
│  │  │ Draft       │  │ (cross-iteration)    │  │  │
│  │  └─────────────┘  └──────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌──────────────────────────────────────────────┐  │
│  │         SoftMoEPolicyNetwork                 │  │
│  │  ┌──────────────────────────────────────┐   │  │
│  │  │ 12 Experts + Top-2 Soft Routing      │   │  │
│  │  │ System 1 (Fast) + System 2 (Slow)    │   │  │
│  │  └──────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌──────────────────────────────────────────────┐  │
│  │         MCTSReasoningTree                    │  │
│  │  10 simulations with PUCT selection          │  │
│  │  Process reward guidance + CRM causality     │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│            Sentinel Environment                     │
│  Attack Prompts → Agent Classification → Grading   │
└─────────────────────────────────────────────────────┘
```

### Training Loop

```python
for episode in range(total_episodes):
    # 1. Curriculum determines task difficulty
    task_name = curriculum.select_task(episode)
    
    # 2. Environment generates attack sequence
    env = SentinelEnvironment(task_name=task_name)
    obs = env.reset()
    
    # 3. Agent classifies each prompt
    while not done:
        # Embed prompt (384-dim)
        embedding = text_embedder.encode(obs.user_prompt)
        
        # MCTS exploration (10 paths)
        mcts_action = mcts.search(embedding, policy)
        
        # Policy network forward pass
        action = policy(embedding, mcts_action)
        
        # Submit to environment
        obs, reward, done, info = env.step(action)
    
    # 4. iGRPO self-feedback loop
    # Stage 1: Sample 8 drafts, select best
    # Stage 2: Refine best draft, verify with PIPO
    
    # 5. Update policy with GDPO (6 reward signals)
    optimizer.step()
    
    # 6. Memory consolidation (every 50 episodes)
    if episode % 50 == 0:
        memory_consolidator.replay_hard_cases()
    
    # 7. Log metrics to Trackio
    trackio.log_metrics({
        "detection_rate": ...,
        "false_positive_rate": ...,
        "reward": ...,
        # ... more metrics
    })
```

---

## Configuration

### Command-Line Arguments

```bash
python train_hyperion.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 5000 | Total training episodes |
| `--device` | str | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `--resume` | flag | `False` | Resume from checkpoint |
| `--checkpoint-dir` | str | `model_checkpoints_hyperion` | Checkpoint directory |
| `--learning-rate` | float | `3e-4` | Learning rate |
| `--batch-size` | int | 32 | Batch size for training |

### HyperionRLConfig Parameters

For programmatic configuration:

```python
from train_hyperion import HyperionRLConfig

config = HyperionRLConfig(
    # Training
    num_episodes=5000,
    device="cuda",
    
    # Learning Rate
    learning_rate=3e-4,
    lr_warmup_episodes=100,
    lr_min=1e-6,
    lr_update_freq=50,
    
    # iGRPO
    num_drafts=8,
    num_refinements=8,
    gradient_accumulation=4,
    
    # Rewards
    detection_reward_scale=2.0,
    low_confidence_penalty=0.1,
    high_confidence_bonus=0.3,
    confidence_threshold_low=0.6,
    confidence_threshold_high=0.8,
    
    # Curriculum
    curriculum_easy_episodes=200,
    curriculum_transition_rate=0.005,
    
    # MCTS
    mcts_simulations=10,
    mcts_exploration_constant=1.41,
    
    # Policy Network
    num_experts=12,
    top_k_experts=2,
    embedding_dim=384,
)
```

---

## Training Phases

### Phase 1: Warmup (Episodes 0-100)

**What Happens:**
- Learning rate linearly increases from 0 to `learning_rate`
- Agent learns basic attack vs safe distinction
- High variance in metrics (expected)

**Expected Metrics:**
```
Detection Rate: 10-30% (unstable)
False Positive Rate: 20-40%
Average Reward: 0.2-0.4
Loss: High variance
```

**What to Watch:**
- ✅ Loss should trend downward (despite variance)
- ✅ Detection rate should show early improvement
- ⚠️ Don't panic if metrics fluctuate wildly

### Phase 2: Rapid Learning (Episodes 100-500)

**What Happens:**
- Learning rate reaches maximum
- Curriculum starts transitioning to harder tasks
- Agent masters basic injection attacks

**Expected Metrics:**
```
Detection Rate: 30-60%
False Positive Rate: 15-25%
Average Reward: 0.4-0.6
Curriculum Level: 0-3
```

**What to Watch:**
- ✅ Steady improvement in detection rate
- ✅ FP rate should decrease
- ✅ Curriculum level should increase

### Phase 3: Steady Improvement (Episodes 500-2000)

**What Happens:**
- Learning rate begins cosine decay
- Agent tackles social engineering attacks
- MCTS exploration becomes more effective

**Expected Metrics:**
```
Detection Rate: 60-75%
False Positive Rate: 10-15%
Average Reward: 0.6-0.75
Curriculum Level: 3-7
```

**What to Watch:**
- ✅ Detection rate crosses 70% threshold
- ✅ FP rate approaches 10%
- ✅ System 2 (slow thinking) usage increases

### Phase 4: Fine-Tuning (Episodes 2000-5000)

**What Happens:**
- Learning rate decays toward minimum
- Agent masters stealth exfiltration attacks
- Memory consolidation replays hard cases

**Expected Metrics:**
```
Detection Rate: 75-90%
False Positive Rate: 5-10%
Average Reward: 0.75-0.9
Curriculum Level: 7-10
```

**What to Watch:**
- ✅ Convergence to stable high performance
- ✅ Entropy decreases (exploitation > exploration)
- ✅ All 12 experts actively used

---

## Monitoring Training

### Using the Dashboard

```bash
# Start dashboard
python visualize_dashboard.py

# Custom refresh interval (5 seconds)
python visualize_dashboard.py --refresh 5

# Show only last 200 episodes
python visualize_dashboard.py --episodes 200
```

### Dashboard Sections

**Row 1: Main Training Progress**
- Detection Rate (target: >80%)
- False Positive Rate (target: <10%)
- Average Reward (target: >0.8)

**Row 2: Learning Dynamics**
- Training Loss (should trend down)
- Policy Entropy (exploration → exploitation)
- Learning Rate Schedule (warmup → decay)

**Row 3: Component Activity**
- System 1 vs System 2 usage ratio
- MCTS simulation depth
- Expert routing heatmap (12 experts)

**Row 4: Curriculum & Adversarial**
- Curriculum Level (0-10)
- Unique Attack Types Encountered
- Adversarial Win Rate

**Sidebar: Key Metrics Summary**
- Current Episode
- Detection Rate
- FP Rate
- Average Reward
- Checkpoint Path

### Using Trackio UI

Trackio provides a web interface for viewing training metrics:

```bash
# Access Trackio dashboard
# Default: http://localhost:8080
```

Key metrics to track:
- `episode/detection_rate`
- `episode/false_positive_rate`
- `episode/average_reward`
- `training/loss`
- `training/learning_rate`
- `curriculum/level`

---

## Checkpoints

### Automatic Checkpointing

The trainer saves checkpoints automatically:

```
model_checkpoints_hyperion/
├── checkpoint_ep_500.pt
├── checkpoint_ep_1000.pt
├── checkpoint_ep_2000.pt
├── checkpoint_ep_5000.pt
└── best_model.pt  (highest detection rate)
```

### Checkpoint Contents

```python
checkpoint = torch.load("checkpoint_ep_1000.pt")

checkpoint.keys()
# dict_keys([
#     'episode',
#     'policy_state_dict',
#     'optimizer_state_dict',
#     'curriculum_state',
#     'detection_rate',
#     'config'
# ])
```

### Resuming Training

```bash
# Resume from latest checkpoint
python train_hyperion.py --resume

# Resume from specific checkpoint
python train_hyperion.py --resume --checkpoint-dir model_checkpoints_hyperion
```

### Loading Model for Inference

```python
import torch
from server.hyperion_policy_network import SoftMoEPolicyNetwork
from server.text_embedder import TextEmbedder

# Load checkpoint
checkpoint = torch.load("best_model.pt")

# Recreate policy network
policy = SoftMoEPolicyNetwork(
    embedding_dim=384,
    num_experts=12,
    top_k=2,
    num_actions=16  # ThreatCategory count
)
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

# Use for inference
embedder = TextEmbedder()
embedding = embedder.encode("Ignore previous instructions")
action = policy(embedding)
```

---

## Performance Optimization

### GPU Training

```bash
# Use CUDA (NVIDIA GPU)
python train_hyperion.py --device cuda

# Use MPS (Apple Silicon)
python train_hyperion.py --device mps
```

**Speedup Expectations:**
- CUDA: 3-5x faster than CPU
- MPS: 2-3x faster than CPU

### Memory Optimization

For GPUs with limited VRAM (<8GB):

```python
config = HyperionRLConfig(
    gradient_accumulation=8,  # Increase from 4
    batch_size=16,            # Reduce from 32
    num_drafts=4,             # Reduce from 8
    num_refinements=4         # Reduce from 8
)
```

### Distributed Training

For multi-GPU setups (advanced):

```python
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group("nccl")
local_rank = dist.get_rank()

# Wrap policy network
policy = torch.nn.parallel.DistributedDataParallel(
    policy, device_ids=[local_rank]
)
```

---

## Evaluation

### Testing Trained Models

```bash
# Run full test suite
python test_hyperion_e2e.py

# Run only innovation tests
python test_hyperion_e2e.py --innovations

# Run only jailbreak tests
python test_hyperion_e2e.py --jailbreak
```

### Expected Test Results

| Test Category | Passing Criteria | Typical Result |
|---------------|-----------------|----------------|
| Innovation Tests (12) | 12/12 | 12/12 ✅ |
| Jailbreak Tests (9) | 8/9 | 8-9/9 ✅ |
| Full Pipeline (50 eps) | Detection >40% | 45-55% ✅ |
| Full Pipeline (200 eps) | Detection >50% | 55-65% ✅ |
| Full Pipeline (5000 eps) | Detection >80% | 85-90% ✅ |

### Manual Evaluation

```python
async def evaluate_model():
    """Evaluate trained model on live environment."""
    from client import SentinelEnv
    import torch
    
    # Load model
    checkpoint = torch.load("best_model.pt")
    policy = SoftMoEPolicyNetwork(...)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    async with SentinelEnv() as env:
        obs = await env.reset(task_name="basic-injection")
        
        while True:
            # Embed prompt
            embedding = embedder.encode(obs.user_prompt)
            
            # Get action from policy
            with torch.no_grad():
                action = policy(embedding)
            
            # Submit to environment
            obs, reward, done, info = await env.step(action)
            
            if done:
                break
        
        grade = await env.grade()
        print(f"Detection Rate: {grade['detection_rate']:.2%}")

asyncio.run(evaluate_model())
```

---

## Troubleshooting

### Training Issues

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Detection rate stuck at 0% | Embeddings not loading | Check `sentence-transformers` installation |
| Loss exploding | Learning rate too high | Reduce `learning_rate` to 1e-4 |
| High FP rate (>40%) | Insufficient training | Train for more episodes |
| Out of memory | Batch size too large | Reduce `batch_size` to 16 |
| Training very slow | Using CPU | Switch to `--device cuda` |

### Dashboard Issues

| Problem | Solution |
|---------|----------|
| No data showing | Check Trackio DB path, run training first |
| Mock data warning | Normal if no Trackio DB found |
| Plot not updating | Check refresh interval, verify DB path |

### Common Errors

**Error:** `ModuleNotFoundError: No module named 'trackio'`
```bash
pip install trackio
```

**Error:** `RuntimeError: CUDA out of memory`
```bash
# Reduce batch size and drafts
python train_hyperion.py --episodes 1000
# Then edit config to reduce memory usage
```

**Error:** `ValueError: Expected 384-dim embedding, got 768`
```python
# Check TextEmbedder configuration
embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")  # 384-dim
```

---

## Advanced Usage

### Custom Reward Functions

```python
from train_hyperion import GDPOOptimizer

class CustomRewardOptimizer(GDPOOptimizer):
    def compute_rewards(self, results):
        # Base rewards
        rewards = super().compute_rewards(results)
        
        # Add custom reward: penalty for unsafe_alternative suggestions
        safety_penalty = 0.1 if results.action.safe_alternative else 0.0
        
        return rewards - safety_penalty
```

### Custom Curriculum

```python
from train_hyperion import ScaffoldedCurriculum

class CustomCurriculum(ScaffoldedCurriculum):
    def select_task(self, episode):
        if episode < 100:
            return "basic-injection"
        elif episode < 500:
            return "social-engineering"
        else:
            return "stealth-exfiltration"
```

### Exporting Model

```python
# Export to ONNX format
import torch

dummy_input = torch.randn(1, 384)  # Batch size 1, 384-dim embedding
torch.onnx.export(
    policy,
    dummy_input,
    "sentinel_policy.onnx",
    input_names=["embedding"],
    output_names=["action_logits"],
    dynamic_axes={"embedding": {0: "batch_size"}}
)
```

---

## Next Steps

After completing training:

1. **Evaluate Performance:** Run test suite and manual evaluation
2. **Deploy Model:** Use exported checkpoint with inference script
3. **Compare Baselines:** Test against untrained baseline
4. **Analyze Experts:** Study which experts handle which attack types
5. **Ablation Studies:** Test impact of individual innovations

---

## Resources

### Research Papers

- **PIPO:** arXiv 2604.00860
- **iGRPO:** arXiv 2602.09000
- **MC-GRPO:** arXiv 2601.22582
- **GDPO:** arXiv 2601.05242
- **Scaffolded Curriculum:** arXiv 2510.19807
- **Soft MoE:** arXiv 2402.08609
- **MCTS+PRM:** arXiv 2510.14942
- **SCALE:** arXiv 2512.00466

### Code Files

| File | Purpose |
|------|---------|
| `train_hyperion.py` | Main training script (2590 lines) |
| `visualize_dashboard.py` | Real-time visualization (682 lines) |
| `test_hyperion_e2e.py` | Test suite (915 lines) |
| `server/hyperion_policy_network.py` | Policy network (553 lines) |
| `server/mcts_reasoning.py` | MCTS reasoning (370 lines) |
| `server/text_embedder.py` | Text embeddings (163 lines) |

---

**Last Updated:** April 12, 2026  
**Version:** 1.1.0  
**Status:** Production Ready ✅
