# Review 1: Mohammed Arfan (PES1UG24AM163)
**Your part:** reference labels, feature engineering, training data and evaluation metrics, and two models (**Random Forest** and **Extra Trees**). You push **third**, after Sinan.

## Your files
| File | What it is |
|---|---|
| `src/labels.py` | Gives every GPS point its true arc by aligning the `.route` to the track |
| `src/features.py` | Candidate table: one row per (GPS point, nearby arc) with 20 features |
| `src/data.py` | Loads the processed tables; selects which rows are used for training and evaluation |
| `src/metrics.py` | Macro F1, precision, recall, log loss, Top-1 accuracy, candidate miss rate, route continuity, route F1 |
| `src/models/trees.py` | Random Forest and Extra Trees |

## 1. `labels.py`: the true arc for each GPS point
`.route` is the list of arcs driven, **in order but without times**. In trip 1 there are 1,315 GPS points but only 525 route arcs, so they don't line up one to one.

**Monotone alignment (dynamic programming):**
- Give each point i a route position j<sub>i</sub> with j<sub>0</sub> ≤ j<sub>1</sub> ≤ …, because the car only moves **forward** along its route.
- Minimise the total distance from the points to their assigned arcs, with the recurrence `dp[i][j] = dist(i, arc j) + min(dp[i-1][0..j])`, then backtrack.
- Label = `route[j_i]`, plus `align_dist`. If `align_dist > 50 m` the label is unreliable (the route doesn't cover that part of the trip; trip 00000054 is 89% uncovered), so those points aren't used.

## 2. `features.py`: 20 features per candidate
| Group | Features | Why |
|---|---|---|
| Distance | `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio` | The true road is close, and close **relative to the other candidates** |
| Ambiguity | `n_cand`, `cand_density_10m` | How many roads compete nearby |
| Direction | `cos_heading`, `abs_heading_diff`, `heading_valid` | Car heading (point i-2 → i+2) vs arc direction; separates the two directions of a two-way street |
| Motion | `speed` | Heading is unreliable when stopped (< 3 m of movement) |
| Position on arc | `arc_len`, `t_frac`, `dist_to_end` | Projecting onto the end of an arc often means the next arc |
| Context | `d_prev2`, `d_next2`, `d_prev5`, `d_next5` | The points 2 and 5 seconds before and after stay close to the true road |
| Topology | `out_degree`, `in_degree`, `is_two_way` | Junction complexity and whether a twin arc exists |

Label column `y = 1` if the arc is the point's true arc. No feature uses the answer.

## 3. `data.py` and `metrics.py`: how models are trained and scored
- **Training rows:** points with a reliable label (≤ 50 m) whose true arc is among the candidates.
- **Evaluation:** every reliably labelled point. If the true arc isn't among a point's candidates, that point counts as **wrong**.
- **Macro F1:** candidate-level F1 averaged over class 0 and class 1, at threshold 0.5.
- **Log loss:** how good the probabilities are; lower is better.
- **Top-1 accuracy:** per GPS point, take the highest-probability candidate and check whether it's the true arc. **This is the main metric for choosing the model**, because it's what map matching needs and it doesn't depend on a threshold.
- **Candidate miss rate:** the share of points whose true arc isn't in the candidate set (0.66% on reliable train points).
- **Route continuity:** the share of consecutive points whose predicted arcs are the same arc or directly connected.

## 4. Your models: Random Forest and Extra Trees
**Random Forest** (200 trees, `max_features = sqrt`, `min_samples_leaf = 5`, `random_state = 42`)
- **Why:** many decision trees trained on bootstrap samples. They learn non-linear rules and interactions (e.g. "heading matters only when moving") and need no feature scaling.
- **Training objective:** each split minimises Gini impurity; the forest averages the trees' probabilities.
- **Results:** CV macro F1 **0.929** (the best of all models), CV Top-1 **0.880**; student-test Top-1 **0.885**, macro F1 0.932, log loss **0.082** (also the best).
- **Fit:** train F1 0.971 vs validation 0.929. That gap of 0.04 is **mild overfitting** of deep trees, limited by `min_samples_leaf = 5`. It still generalises very well.
- **Weakness:** the slowest to train (122 s) and a large model file.

**Extra Trees** (same settings, but split thresholds are chosen **randomly** and there's no bootstrap)
- **Why:** more randomness makes the trees less similar, which can reduce overfitting.
- **Results:** CV macro F1 **0.925**, CV Top-1 0.874; student-test Top-1 **0.876**, log loss 0.089.
- **Fit:** smaller gap (0.950 vs 0.925), so less overfitting than RF, but slightly lower accuracy too. The random thresholds blur sharp boundaries such as a few-metre distance difference.
- Trains twice as fast as Random Forest (60 s).

## Likely viva questions
- **Why align the route?** It has no timestamps; without alignment there's no per-point label.
- **Why is distance alone not enough?** Two-way streets mean two arcs at the same distance. The nearest arc is correct only 56% of the time.
- **Why use Top-1 as the main metric instead of accuracy?** About 85% of candidate rows are negatives, so plain row accuracy looks high even for a bad model. Top-1 measures the real task.
- **What is the difference between Random Forest and Extra Trees?** RF uses bootstrap samples and the best split threshold; ET uses the whole data and random thresholds.
- **Is Random Forest overfitting?** Slightly (train 0.971 vs validation 0.929), but its validation score is still the best.

## How to push (after Sinan has pushed)
```bash
git config --global user.name "Mohammed Arfan"
git config --global user.email "<your GitHub email>"
git clone https://github.com/Bilal-biryani/Ml-hackathon-.git     # or: git pull
cd Ml-hackathon-
```
Copy everything inside `files/` from this zip into `Ml-hackathon-/`, keeping the folders. Then:
```bash
git add src/labels.py src/features.py src/data.py src/metrics.py src/models/trees.py
git status
git commit -m "Route-to-track label alignment, candidate features, evaluation metrics, tree ensembles"
git push
```
If the push is rejected, run `git pull --rebase` and then `git push`. Tell **Mirunjai** when you're done.
