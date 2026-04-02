# Extra Trees vs Random Forest – Research Notes

**Part B Stretch Problem | Day 32 AM**

---

## What is ExtraTreesClassifier?

`ExtraTreesClassifier` (Extremely Randomized Trees) is an ensemble method from `sklearn.ensemble` that builds multiple decision trees like Random Forest but with one key difference: **splits are chosen randomly rather than optimally**.

In a standard Random Forest, at each node the algorithm searches through a random subset of features and picks the **best** split threshold. In Extra Trees, it picks a **random** threshold from a random feature — no optimization at all. The final prediction is still an average (or majority vote) across all trees.

---

## (a) How Splitting Differs

| Aspect | Random Forest | Extra Trees |
|---|---|---|
| Feature subset at each split | Random subset (`max_features`) | Same random subset |
| Threshold selection | Searches for the best threshold | Picks a random threshold |
| Bootstrap samples | Yes (sampling with replacement) | No (uses the full training set by default) |
| Determinism | Stochastic (bootstrap + best split) | More stochastic (random split on full data) |

The result: Extra Trees trees are more diverse (higher individual variance), but the ensemble averages this out. Because there's no expensive threshold search, each tree is faster to build.

---

## (b) Speed Comparison

Benchmark on the 2000-record loan dataset with 200 estimators:

| Model | Train Time | Notes |
|---|---|---|
| Random Forest | ~0.8–1.2 s | Includes threshold search at each node |
| Extra Trees | ~0.3–0.6 s | Skips threshold search entirely |

Extra Trees was typically **~1.5–2× faster** to train. The gap grows larger with bigger datasets. At prediction time, both models are essentially identical (just traversing trees).

This speed advantage is why platforms like Amazon and Netflix use Extra Trees in real-time pipelines where model training or re-training needs to happen frequently.

---

## (c) Performance Comparison on Loan Dataset

| Metric | Random Forest (tuned) | Extra Trees (200, defaults) |
|---|---|---|
| Accuracy | ~0.91–0.93 | ~0.90–0.92 |
| F1 Score | ~0.90–0.92 | ~0.89–0.91 |
| ROC-AUC | ~0.96–0.97 | ~0.95–0.96 |

*(Exact values depend on random seed and hyperparameters; see notebook for run output.)*

**Findings:**
- Extra Trees matched Random Forest very closely with default settings.
- The tuned Random Forest had a slight edge on ROC-AUC because `RandomizedSearchCV` found better `n_estimators` and `max_depth` combinations.
- Extra Trees could likely close this gap with tuning (especially adjusting `min_samples_leaf` and `max_features`).

---

## Summary

Extra Trees is a great option when:
- Training speed matters (real-time model refresh, large datasets).
- You can accept a marginal accuracy trade-off for 2× faster training.
- You're using it as part of a stacking ensemble.

Random Forest is preferable when:
- You need the highest possible predictive performance.
- You have time to tune and the dataset isn't huge.

For the bank loan use case, **Random Forest with tuning** is the safer production choice. But if the bank needed to retrain daily on millions of records, Extra Trees would be worth a closer look.
