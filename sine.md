### 1. sines

We generate 10,000 samples of a 5-dimensional sinusoidal sequence with varying frequencies, amplitudes, and phases, where each feature is correlated with others.

For each dimension \( i \in \{1, 2, \ldots, 5\} \):

\[
x_i(t) = a \sin(2\pi \eta t + \theta)
\]

where:

- \( \eta \sim \mathcal{U}[0.1, 0.15] \)
- \( \theta \sim \mathcal{U}[0, 2\pi] \)
- \( a \sim \mathcal{U}[1.0, 3.0] \)