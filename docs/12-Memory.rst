Memory
===================================================

In many real-world scenarios we do not have access to the ground truth state of the environment. In these cases we only get observations that have some mutual information with the true state. In this case the Markov property is violated because we are dealing with a  `POMDP <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_ and hence information from past observations influences the estimate of the current state. Because the belief about the current state is dependent on previous observations, also the future behavior and action selection decision are dependent on them.

To respect this in our policy, it needs to be able to reason about sequences of observations. A natural way to implement this is using recurrent neural networs (RNNs) that carry an internal state that can transport information through time.