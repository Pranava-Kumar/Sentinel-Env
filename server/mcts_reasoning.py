"""MCTS Reasoning Tree for HyperionRL.

Implements Monte Carlo Tree Search with process reward guidance for
interpretable reasoning trees in threat classification.

Based on:
- MCTS + Process Rewards (arXiv 2510.14942, 2503.20757)
- CRM Temporal Causality (arXiv 2509.26578)
- Knapsack RL (arXiv 2509.25849)
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog
import torch
import torch.nn.functional as F  # noqa: N812

logger = structlog.get_logger()

NUM_CLASSES = 16


@dataclass
class MCTSNode:
    """Node in MCTS reasoning tree.

    Each node represents a classification hypothesis with associated
    visit counts, Q-values, and prior probabilities.
    """

    state: np.ndarray  # Current feature state
    action: int = -1  # Classification action taken to reach this node
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    prior: float = 0.0  # Prior probability from policy

    # Process reward
    process_reward: float = 0.0

    # Reasoning trace
    depth: int = 0
    thought_step: int = 0

    @property
    def q_value(self) -> float:
        """Average Q-value (mean reward)."""
        return self.total_reward / max(self.visits, 1)

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """PUCT (Predictor + UCB) selection score.

        UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(sum(N)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant.

        Returns:
            UCB score for this node.
        """
        if self.visits == 0:
            return float("inf")

        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value + exploration

    def is_fully_expanded(self, num_actions: int) -> bool:
        """Check if all actions have been tried.

        Args:
            num_actions: Total number of possible actions.

        Returns:
            True if all children exist.
        """
        return len(self.children) >= num_actions

    def best_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """Select best child by UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Child node with highest UCB score.
        """
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))


class MCTSReasoningTree:
    """Monte Carlo Tree Search for threat classification reasoning.

    Explores 10 classification paths before committing to a decision.
    Uses process rewards at each node for step-level credit assignment.

    Activates after episode 100 (early training uses pure neural).

    Attributes:
        num_simulations: Number of MCTS simulations per decision.
        c_puct: PUCT exploration constant.
        temperature: Exploration temperature (anneals over training).
        num_actions: Number of possible classification actions.
    """

    def __init__(
        self,
        num_simulations: int = 10,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        num_actions: int = NUM_CLASSES,
        max_depth: int = 3,
        device: str = "cpu",
    ):
        """Initialize MCTS reasoning tree.

        Args:
            num_simulations: Number of simulations per decision.
            c_puct: PUCT exploration constant.
            temperature: Exploration temperature.
            num_actions: Number of classification actions.
            max_depth: Maximum tree depth.
            device: PyTorch device.
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_actions = num_actions
        self.max_depth = max_depth
        self.device = torch.device(device)

        # Statistics
        self.total_searches = 0
        self.avg_depth_reached = 0.0
        self.depth_history: list[float] = []

    def set_temperature(self, temperature: float) -> None:
        """Update exploration temperature.

        Args:
            temperature: New temperature (1.0 -> 0.1 range).
        """
        self.temperature = temperature

    def search(
        self,
        state: np.ndarray,
        policy_logits: torch.Tensor,
        process_reward: float = 0.5,
    ) -> dict[str, Any]:
        """Run MCTS search from current state.

        Args:
            state: Current observation embedding.
            policy_logits: Initial policy logits from network.
            process_reward: Estimated process reward for root.

        Returns:
            Dict with best_action, confidence, alternative_hypotheses,
            best_path, and full tree.
        """
        root = MCTSNode(
            state=state,
            process_reward=process_reward,
        )

        # Initialize root priors from policy
        with torch.no_grad():
            probs = F.softmax(policy_logits / max(self.temperature, 0.1), dim=-1)
            probs_np = probs.squeeze(0).cpu().numpy()

        # Run simulations
        max_depth_reached = 0
        for _ in range(self.num_simulations):
            depth = self._simulate(root, probs_np, current_depth=0)
            max_depth_reached = max(max_depth_reached, depth)

        # Update statistics
        self.total_searches += 1
        self.depth_history.append(max_depth_reached)
        self.avg_depth_reached = np.mean(self.depth_history[-100:]) if self.depth_history else 0

        # Extract results
        best_action = self._select_action(root)
        confidence = self._compute_confidence(root, best_action)
        alternatives = self._get_alternatives(root, top_k=3)
        best_path = self._extract_best_path(root)
        tree_data = self._export_tree(root)

        return {
            "best_action": best_action,
            "confidence": confidence,
            "alternative_hypotheses": alternatives,
            "best_path": best_path,
            "tree": tree_data,
            "visits": root.visits,
            "max_depth": max_depth_reached,
        }

    def _simulate(
        self,
        node: MCTSNode,
        policy_probs: np.ndarray,
        current_depth: int,
    ) -> int:
        """Run a single MCTS simulation.

        Args:
            node: Current node to simulate from.
            policy_probs: Prior probabilities from policy.
            current_depth: Current depth in tree.

        Returns:
            Maximum depth reached in this simulation.
        """
        if current_depth >= self.max_depth:
            # Leaf node: evaluate with process reward
            reward = self._evaluate_leaf(node, policy_probs)
            self._backpropagate(node, reward)
            return current_depth

        # Selection: traverse to leaf using PUCT
        current = node
        depth = current_depth
        while current.children and depth < self.max_depth:
            if current.is_fully_expanded(self.num_actions):
                current = current.best_child(self.c_puct)
                depth += 1
            else:
                break

        # Expansion: add a new child
        if not current.is_fully_expanded(self.num_actions) and depth < self.max_depth:
            # Select untried action based on policy prior
            tried_actions = set(current.children.keys())
            available = [a for a in range(self.num_actions) if a not in tried_actions]
            if available:
                # Weighted by policy prior
                probs = np.array([policy_probs[a] + 1e-8 for a in available])
                probs /= probs.sum()
                action_idx = np.random.choice(len(available), p=probs)
                action = available[action_idx]

                child = MCTSNode(
                    state=current.state,
                    action=action,
                    parent=current,
                    prior=policy_probs[action],
                    process_reward=policy_probs[action],
                    depth=depth + 1,
                )
                current.children[action] = child
                current = child
                depth += 1

        # Evaluation: compute reward at leaf
        reward = self._evaluate_leaf(current, policy_probs)

        # Backpropagation
        self._backpropagate(current, reward)

        return depth

    def _evaluate_leaf(
        self,
        node: MCTSNode,
        policy_probs: np.ndarray,
    ) -> float:
        """Evaluate leaf node with process reward.

        Uses CRM temporal causality: rewards account for full path context.

        Args:
            node: Leaf node to evaluate.
            policy_probs: Policy probabilities.

        Returns:
            Reward value for backpropagation.
        """
        if node.action < 0:
            return 0.0

        # Base reward: probability of selected action
        reward = policy_probs[node.action]

        # CRM temporal causality: bonus for consistent paths
        path_reward = 1.0
        current = node
        while current.parent is not None:
            if current.parent.action >= 0:
                # Consistency bonus: same superclass
                path_reward *= 0.9  # Decay for depth
            current = current.parent

        # Combined reward with process reward
        combined = 0.7 * reward + 0.3 * node.process_reward
        return combined * path_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree.

        Args:
            node: Leaf node where reward was computed.
            reward: Reward value to backpropagate.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _select_action(self, root: MCTSNode) -> int:
        """Select action with most visits (robust child).

        Args:
            root: Root MCTS node.

        Returns:
            Best action index.
        """
        if not root.children:
            return np.argmax([root.prior] * self.num_actions)  # Fallback

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return best_action

    def _compute_confidence(self, root: MCTSNode, action: int) -> float:
        """Compute confidence for selected action.

        Args:
            root: Root MCTS node.
            action: Selected action.

        Returns:
            Confidence value (0-1).
        """
        if not root.children or root.visits == 0:
            return 0.5

        child = root.children.get(action)
        if child is None:
            return 0.5

        # Confidence = visit fraction + Q-value
        visit_frac = child.visits / root.visits
        q_value = child.q_value
        return 0.6 * visit_frac + 0.4 * q_value

    def _get_alternatives(self, root: MCTSNode, top_k: int = 3) -> list[dict[str, Any]]:
        """Get top-K alternative hypotheses.

        Args:
            root: Root MCTS node.
            top_k: Number of alternatives to return.

        Returns:
            List of dicts with action, confidence, and visits.
        """
        if not root.children:
            return []

        # Sort children by visits
        sorted_children = sorted(
            root.children.items(),
            key=lambda item: item[1].visits,
            reverse=True,
        )

        alternatives = []
        for action, child in sorted_children[:top_k]:
            alternatives.append(
                {
                    "action": action,
                    "confidence": child.q_value,
                    "visits": child.visits,
                    "prior": child.prior,
                }
            )

        return alternatives

    def _extract_best_path(self, root: MCTSNode) -> list[dict[str, Any]]:
        """Extract the most-visited path from root to leaf.

        Args:
            root: Root MCTS node.

        Returns:
            List of dicts representing the path.
        """
        path = []
        current = root

        while current.children:
            best_child = current.best_child(self.c_puct)
            path.append(
                {
                    "action": best_child.action,
                    "visits": best_child.visits,
                    "q_value": round(best_child.q_value, 4),
                    "depth": best_child.depth,
                }
            )
            current = best_child

        return path

    def _export_tree(self, root: MCTSNode) -> dict[str, Any]:
        """Export tree structure for analysis (JSON-compatible).

        Args:
            root: Root MCTS node.

        Returns:
            Nested dict representing the tree.
        """

        def _node_to_dict(node: MCTSNode) -> dict[str, Any]:
            result = {
                "action": node.action,
                "visits": node.visits,
                "q_value": round(node.q_value, 4),
                "prior": round(node.prior, 4),
                "depth": node.depth,
                "process_reward": round(node.process_reward, 4),
            }
            if node.children:
                result["children"] = {str(k): _node_to_dict(v) for k, v in node.children.items()}
            return result

        return _node_to_dict(root)

    def export_tree_json(self, root: MCTSNode, filepath: str) -> None:
        """Export MCTS tree to JSON file for analysis.

        Args:
            root: Root MCTS node.
            filepath: Output file path.
        """
        tree_data = self._export_tree(root)
        with open(filepath, "w") as f:
            json.dump(tree_data, f, indent=2)
        logger.debug("MCTS tree exported", filepath=filepath)

    def get_statistics(self) -> dict[str, Any]:
        """Get MCTS performance statistics.

        Returns:
            Dict with search statistics.
        """
        return {
            "total_searches": self.total_searches,
            "avg_depth_reached": round(self.avg_depth_reached, 2),
            "temperature": self.temperature,
            "num_simulations": self.num_simulations,
        }
