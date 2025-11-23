# Algorithmic Market Audit

A multi-agent reinforcement learning simulation framework designed to audit pricing behaviors in a duopoly market. The system compares No-Swap-Regret (NSR) dynamics against Reinforcement Learning (RL) dynamics to detect and mitigate tacit collusion via Lagrangian penalties.

## Modules

- **MarketCore**: `DuopolyEnv` implementing Logit Demand Model and market mechanics.
- **AgentZoo**: Implementations of `NSRAgent`, `RLAgent` (DQN), and `ConstrainedRLAgent`.
- **AuditingEngine**: `ConformalAuditor` for detecting collusion.
- **SimulationController**: Manages scenarios and metrics collection.

## Usage

(Instructions to be added)
