# MFD+PAIR: Interaction-Aware Tabular Retrieval

A standalone implementation of marginal feature discriminability (MFD) and query-conditioned pair retrieval for binary and multiclass classification. The package contains the method, a feature-budget adapter, class-balanced demonstration selection, and generic prompt construction. Dependencies are NumPy and pandas.

## Install

Use Python 3.10 or newer:

```bash
python -m pip install -e .
```

## Python API

Supply your own feature-only training frame, training labels, and held-out feature frame:

```python
from mfd_pair import MFDPairRetriever, save_corpus

retriever = MFDPairRetriever(max_features=15, demonstrations=16, seed=42)
retriever.fit(X_train, y_train, categorical_features=categorical_columns)

result = retriever.retrieve(X_query.iloc[0].to_dict())
# result.indices: positions in X_train, not pandas index labels
# result.scores: descending retrieval scores
# result.label_counts: demonstration counts by frozen class token
# result.pair_weights: active query-specific beta weights

bundle = retriever.build_corpus(X_query)
save_corpus(bundle, "run")
```

`categorical_features=None` infers categorical fields from nonnumeric and Boolean dtypes. Supply the list explicitly for integer-coded categories. Exclude target columns and identifiers from the feature frame yourself. Training row order is preserved; only training labels are used to fit the retriever. Query labels are neither required nor accepted by `score` or `retrieve`.

`build_corpus(..., query_labels=y_query)` optionally writes scoring targets to each record's `output`. These targets never enter retrieval, the task instruction, or the current-query prompt. Omit this argument for unlabeled inference. An optional `instruction` and `feature_descriptions` dictionary customize the generic text at runtime.

## CSV interface

The command reads caller-provided files; it performs no downloading or automatic data splitting:

```bash
python -m mfd_pair \
  --train-csv TRAIN.csv --query-csv QUERIES.csv \
  --target target --categorical feature_b feature_c \
  --max-features 15 --demonstrations 16 --output run
```

The training CSV must contain the target. A target column in the query CSV is optional and is used only as an output target for scoring. Feature names, task descriptions, and category types are provided by the caller. Pass `--instruction-file FILE.txt` for a custom task instruction. Existing corpus files are never overwritten.

Generated files:

- `corpus.json`: `instruction`, `input`, and optional scoring `output`.
- `retrieval.jsonl`: selected training positions, scores, class counts, and active pair weights.
- `metadata.json`: runtime field selection, label mapping, and fitted-index metadata.

Feed only `instruction` and `input` to your frozen language model. `output` is a scoring target. The package intentionally exposes a backend-neutral prompt interface rather than bundling a model runner. Save the model's actual response separately and parse it with:

```python
from mfd_pair import parse_prediction
label = parse_prediction(raw_response, n_classes=len(retriever.classes_))
```

The expected final format is `Prediction is class_0.` (or another valid class token). The parser returns the last valid explicit prediction, or `None`. This is an output-format parser, not a model-specific reasoning-channel filter; extract the final answer channel before calling it when your backend provides separate channels. Class tokens follow first occurrence in the training labels; `classes_[i]` gives the original value for `class_i`.

## Method and implementation

1. Estimate each field's MFD from training class-conditional distributions.
2. Rank the full candidate pool by MFD and retain at most 15 fields.
3. Mine unordered pairs only within the retained subset. Filter by incremental joint information and pooled joint-state support.
4. Estimate class-conditional lift contrasts and stratified-bootstrap stability.
5. For each query, combine MFD-weighted single-field similarities with reliability-weighted pair similarities.
6. Select class-balanced training examples and serialize them after the current query.

The feature budget is enforced **before** pair mining. Pair scores do not select fields in this version. This package does not implement a joint single/pair feature-selection objective or all-feature pair search outside the retained subset.

Numerical statistics use frozen training quartiles, with duplicate and endpoint boundaries removed. Missing values occupy a separate state. Numerical similarity uses continuous values standardized by the training mean and population standard deviation. Categorical similarity compares original identities; singleton pooling affects statistical estimation only.

For two classes, lift contrast has one component. For more classes, the method averages absolute pairwise class contrasts after stability weighting. Per-state raw pair weights are normalized as `beta = raw / (1 + sum(raw))`. The unit mass preserves reliability shrinkage.

The final score is:

```text
(single_score + lambda * pair_score)
/ (sum(single_weights) + lambda * sum(beta) + epsilon)
```

`lambda=1` is a unit multiplier, not a fixed 50:50 mixture. For weak interaction evidence, the pair branch remains attenuated.

Binary retrieval allocates half the slots to each class; an odd slot goes to class zero. Multiclass retrieval starts with `floor(K/C)` per class and assigns the remaining slots by the strongest next candidates. Insufficient class pools are backfilled by the highest-scoring unselected records. At most the available number of distinct training rows is returned. `exclude_position` can omit a training row for self-retrieval.

## Defaults

| Setting | Default | Meaning |
|---|---:|---|
| `max_features` | 15 | MFD-selected field budget |
| `demonstrations` | 16 | Class-balanced demonstration budget |
| `seed` | 42 | Shared bootstrap seed |
| Quantile bins | Up to 4 nonmissing bins | Fixed quartile probabilities, not a constructor argument |
| `alpha` | 1 | Additive pseudocount per discrete state or joint cell |
| `min_category_count` | 2 | Pool observed training categories with count below 2 |
| `gamma_tol` | 1e-6 bits | Near-zero positive incremental-information cutoff |
| `min_pair_support` | 5 | Minimum pooled joint-state count |
| `n_bootstrap` | 10 | Stratified training resamples |
| `support_shrinkage` | 10 | Scale in `n / (n + scale)` |
| `tau` | 1 | Numerical similarity temperature in standardized units |
| `epsilon` | 1e-12 | Numerical protection and negligible-weight tolerance |
| `interaction_weight` | 1 | Fusion coefficient lambda |

Pass core options as keyword arguments to `MFDPairRetriever`, for example `MFDPairRetriever(tau=1.0, interaction_weight=1.0)`. Support and bootstrap factors are reliability heuristics, not formal significance tests. Defaults are shared settings; no universal optimality is claimed.

## Code map

| File | Responsibility |
|---|---|
| `mfd_pair/interaction.py` | Binary MFD, joint information, lift, bootstrap, query weights, and scores |
| `mfd_pair/_multiclass.py` | Multiclass class-contrast extension |
| `mfd_pair/pipeline.py` | Training-only field budget, class-balanced selection, prompts, and parser |
| `mfd_pair/__main__.py` | Generic CSV-to-corpus command |
| `tests/test_method.py` | Synthetic correctness and interface checks |

The low-level `InteractionAwareRetriever.fit` uses every field passed to it. Use `MFDPairRetriever` for the default feature-budget and class-balanced pipeline. Low-level `retrieve` is an unrestricted top-k operation; the high-level wrapper applies the class quotas.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests construct artificial inputs in memory and create temporary outputs only. `VALIDATION.json` records the release checks, and `SHA256SUMS` covers the delivered files. Low-level pickle persistence is for trusted local artifacts and includes fitted training state; no fitted artifacts are distributed with the source package.
