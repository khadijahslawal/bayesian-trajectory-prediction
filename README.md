# Bayesian Pedestrian Trajectory Predction with Uncertainty Quantification
Applying bayesian trajectory prediction for autonomous driving safety

## Project Overview
This project investigates how uncertainty quantification can make trajectory prediction safer in autonomous driving scenarios. Rather than predicing a single path, our Bayesian models would output a distribution over possible futures. This allows for a downstram safety system to reason about risks given certain levels of uncertainity. 

For an end-to-end system the key deliverables are as follows:
1. A deterministic baseline LSTM model
2. MC Dropout (primary bayesian method)
3. Variational BNN (secondary bayesian method)
4. Safety frameowrk
