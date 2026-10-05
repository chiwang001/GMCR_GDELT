# Equilibrium-robustness evidence report

## Scope and interpretation

This analysis tests whether sampling stability of the February 2020 ensemble-based pairwise preference matrices propagates to probability-form GMCR stability margins and equilibrium classifications. The full set of 500 valid retained solutions is a finite reference ensemble, not a population truth. Pairwise entries are ranking frequencies across retained compatible structures, not action probabilities, posterior probabilities, or causal effects.

## Mandatory audit

- The solution-level utility file contains 500 distinct complete solutions; the driver forecast file contains 500; and 500 parameter files were found. Their seed sets match.
- Every forecast row is labeled 202002. The latest event-impact input and observed conflict state are 202001 and 202001, respectively. Thus the February 2020 extrapolation uses information through January 2020 under the archived workflow.
- Matrix orientation is `P[s,q] = fraction of retained solutions with u(s) > u(q)`. A move from `s` to `q` is a strict unilateral improvement exactly when `P[q,s] > tau`.
- Reconstructing both 76 by 76 matrices from solution-level utilities gives a maximum absolute discrepancy of 0 from the archived matrices.
- The archived probability-stability table lies on a 369-solution grid (maximum rounding residual 1.83e-08), while the current matrices use 500 solutions; its declared source paths resolve in the checkout. Applying the same code to the current matrices therefore does not reproduce the legacy archive: the maximum margin difference is 0.496, and 3 of 608 actor--state--concept classifications differ. The new experiment uses the exactly reconstructed current 500-solution matrices as its internally consistent reference; `archived_stability_comparison.csv` records every difference.
- The archived convergence columns named `squared_difference_sum` are unnormalized sums of squared entry differences. The `mean_squared_difference` columns are MSEs obtained by dividing by 76 squared entries per actor (or twice that count for the combined value).
- For state 54, the archived 369-solution US SMR margin is 0.1964769648, and the current 500-solution margin is 0.1880000000; both classify US as SMR-stable. The manuscript actor-specific cell omits this classification. Joint SMR remains false because the current CN SMR margin is -0.4700000000. The US cell should read `GMR, SMR, SEQ`; the common-concept cell remains `GMR, SEQ`.

## Methods

For each proper-subset size, independent subsets were sampled without replacement from the full finite ensemble. Each subset was propagated through the complete workflow: strict utility comparisons, CN and US pairwise matrices, strict-improvement sets at tau=0.50, and the archived probability-form Nash, GMR, SMR, and SEQ margin functions on the same directed state graph. We report matrix sup error and MSE, maximum margin error, actor-state-concept agreement, false-stable and false-unstable rates, concept-specific Jaccard similarity of joint equilibrium sets, and selected-state joint margins.

The empty-union Jaccard convention is 1.0 because both analyses then return the same empty joint equilibrium set. False-stable rates use full-reference unstable classifications as the denominator; false-unstable rates use full-reference stable classifications.

## Conditional robustness statement

Let epsilon be the sup-norm difference between a subset matrix and the reference matrix. The global reference separation of directed unilateral-move entries from tau is 0.0020000000. If epsilon is smaller than this separation, no unilateral-improvement set changes. Conditional on fixed improvement sets, the implemented finite minimum/maximum operators, including the finite missing-sanction convention, are non-expansive, so each probability-form stability margin changes by at most epsilon. A classification is therefore certified unchanged when its reference margin has absolute value greater than epsilon. The output also applies the sharper state-local separation condition and separately identifies states with no feasible unilateral moves, whose stability is graph-structural.

This is a conditional certificate. When an entry lies on or near the threshold, the conservative analytical condition can fail even though every sampled classification is empirically preserved.

## Results

At the largest proper-subset size R=490, the median pairwise-matrix sup error is 0.0088, the median maximum margin error is 0.0096, and median actor-state-concept agreement is 1.0000. Complete actor-state-concept agreement occurs in 89 of 100 replicates; the minimum agreement is 0.9967. The smallest concept-specific median Jaccard similarity of the joint equilibrium sets is 1.0000. All four joint equilibrium sets are reproduced in every R=490 replicate: yes.

The 95th percentile of the maximum margin error is 0.5000, with 6 of 100 replicates showing a maximum jump of at least 0.49. These jumps occur because a directed comparison close to tau changes the strict-improvement set; they confirm why an unconditional global Lipschitz claim is invalid. The affected actor-specific boundary classifications are reported rather than suppressed. They do not alter any joint equilibrium set at R=490.

Across the 16 selected state-concept combinations, the minimum empirical joint-classification preservation rate is 1.0000; 16 combinations satisfy the two-actor analytical certificate in every replicate at R=490. Their 5th--95th percentile joint-margin intervals do not cross zero from R=100 onward. For state 73, CN has no feasible unilateral move, so its four positive margins are graph-structural; US has a feasible move but no strict improvement. This distinction prevents structural stability from being attributed to matrix convergence.

The full numerical results are in the accompanying CSV files. The main convergence figure connects matrix error, margin error, classification agreement, and joint-set Jaccard similarity. The selected-state figure displays the joint margin (the smaller actor margin) and its 5th--95th percentile band, with zero marked as the classification boundary.

## Interpretation and limitations

The results quantify computational and finite-ensemble sampling stability conditional on the specified dynamic preference-update structure, parameter bounds, retained-solution protocol, directed graph, and threshold. They do not resolve parameter non-identification, model-form uncertainty, event-coding uncertainty, or graph misspecification. Stability caused by an absence of feasible moves is reported separately from preference-induced stability. Thresholds above 0.5 are stricter ensemble-support requirements, not alternative behavioral-probability estimates.

The strongest defensible conclusion is narrower than global margin robustness. Weak identification of individual preference-update structures does not destabilize the four selected-state conclusions or the joint equilibrium sets at near-full ensemble sizes, under the stated model and graph. A small number of actor-specific boundary margins remain threshold-sensitive. Analytical certification is therefore reported separately from empirical invariance, and neither is generalized beyond these conditions.
