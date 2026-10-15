# PRIS-R

PRIS-R stands for **Proactive-Reactive Intelligent Slicing in congested railways FRMCS network**. This repository supports the PRIS-R study on intelligent railway network slicing for FRMCS, where critical and performance railway applications share the same radio infrastructure under strict latency, reliability, and efficiency constraints.

The work is motivated by the fact that railway traffic is strongly shaped by train mobility, operational events, and rapidly changing radio conditions. In the paper, PRIS-R combines a lightweight spatio-temporal forecasting model with a deep reinforcement learning scheduler to jointly handle proactive resource reservation and reactive protection for critical traffic during congestion and emergency conditions.

This work is currently being improved and updated based on the review feedback  received  from the OJComs Journal.

## Scope

This repository is intended for simulation and evaluation work of the PRIS-R of the paper currently under review. In this paper we adressed:

- spatio-temporal traffic forecasting
- railway network slicing
- congestion-aware resource allocation
- emergency-aware prioritization of critical railway services

## Simulation

The simulation is provided in `simulations.ipynb`. It replays the mobility-driven FRMCS traffic scenarios in `STG_Datasets/` and schedules downlink and uplink demand for critical and performance slices at each simulation interval. A spatio-temporal forecaster supplies the next-step traffic estimate, which is used by the proactive approaches; the reactive approach instead responds to current demand only. Emergency events increase the priority of critical services and allow the scheduler to protect them when resources are congested.

The notebook compares a reactive baseline, a proactive heuristic, a pro-reactive heuristic, and the PRIS-R pro-reactive DRL policy. It reports throughput, delay, reliability, resource puncturing, and runtime-related metrics, with outputs written to `sim_outputs_v5_compare/<scenario>/`.

To run an evaluation, open the notebook from the repository root, choose `SCENARIO_NAME` (`STGdata_A`, `STGdata_B`, or `STGdata_C`) and the desired entries in `RUN_APPROACHES`, then run the notebook. The included checkpoints in `saved_models/` can be used for evaluation; set `IF_TRAINING = True` only when training a new DRL policy.

