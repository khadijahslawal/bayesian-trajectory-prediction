# Bayesian Pedestrian Trajectory Predction with Uncertainty Quantification
Applying bayesian trajectory prediction for autonomous driving safety

## Project Overview
This project explores how uncertainty quantification can improve the safety of pedestrian trajectory prediction in autonomous driving environments. Instead of predicting a single deterministic future path, the models produce a distribution over possible trajectories, allowing downstream safety systems to reason about risk and uncertainty.

For example, if a pedestrian could plausibly move in multiple directions, the model should express this uncertainty rather than committing to a single prediction. This enables planning systems to make safer decisions when the future behavior of agents is ambiguous.

## System Components and Key Deliverables 
The project evaluates several trajectory prediction approaches:

1. Deterministic LSTM baseline
A standard sequence model that predicts a single future trajectory.

2. MC Dropout (primary Bayesian method)
Uses dropout at inference time to approximate Bayesian uncertainty and generate multiple trajectory samples.

3. Variational Bayesian Neural Network (secondary Bayesian method)
Applies variational inference to learn a distribution over model weights, producing probabilistic trajectory predictions.

4. Safety evaluation framework
A framework for analyzing prediction uncertainty and assessing how probabilistic forecasts can improve downstream decision-making.
