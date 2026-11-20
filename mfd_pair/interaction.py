"""Train-only interaction-aware retrieval for binary and multiclass tabular prediction.

Implements the MFD, incremental pair information, class-conditional lift, and
query-specific soft binding for tabular classification.
Only numpy and pandas are required, and are imported when the retriever is used.
Support cutoffs and bootstrap sign agreement are practical reliability heuristics;
neither constitutes a formal hypothesis test or a statistical significance claim.
"""

from itertools import combinations
from pathlib import Path
import pickle


def _dependencies():
    import numpy as np
    import pandas as pd

    return np, pd


class InteractionAwareRetriever:
    """Fit frozen training statistics and rank historical rows for unlabeled queries.

    Parameters
    ----------
    alpha : float
        Positive Dirichlet pseudo-count per state/cell. Lift marginals are sums of
        the *same smoothed joint distribution*, rather than separately smoothed.
    min_category_count : int
        Training categories occurring fewer times are encoded as OTHER for
        statistics. Retrieval still compares their original category values.
    min_pair_support : int
        Minimum pooled training count for the query's joint state. A combination
        need not occur in both classes: that would discard deterministic XOR.
    support_shrinkage : float
        State reliability is n / (n + support_shrinkage), after the support cutoff.
    n_bootstrap : int
        Number of stratified training bootstrap resamples, using frozen encoders.
        Stability is the proportion retaining positive Gamma and the fitted lift
        contrast sign for that state. Zero disables this mechanism; one resample
        is a degraded diagnostic, not a meaningful stability assessment.
    gamma_tol : float
        Retain pairs only when empirical MI(Y; pair) - max(single MI) exceeds it.
        Every feature pair is considered, including zero-MFD features.
    interaction_weight : float
        Lambda in the final normalized marginal-plus-pair score.
    tau : float
        Positive scale for exp(-abs(z_query - z_train) / tau), where the mean and
        standard deviation used for z are learned exclusively during fit.
    epsilon : float
        Positive stabilizer for MFD weights and the final score denominator.

    ``feature_importances_`` contains raw equal-prior JSD in bits;
    ``feature_weights_`` contains (MFD + epsilon) normalized over all features.
    ``retrieve`` returns positional indices into the original training row order,
    together with descending scores. It never consumes a query label.
    """

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        *,
        alpha=1.0,
        min_category_count=2,
        min_pair_support=5,
        support_shrinkage=10.0,
        n_bootstrap=10,
        random_state=42,
        gamma_tol=1e-6,
        interaction_weight=1.0,
        tau=1.0,
        epsilon=1e-12,
    ):
        self.alpha = alpha
        self.min_category_count = min_category_count
        self.min_pair_support = min_pair_support
        self.support_shrinkage = support_shrinkage
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.gamma_tol = gamma_tol
        self.interaction_weight = interaction_weight
        self.tau = tau
        self.epsilon = epsilon

    def _validate_parameters(self):
        import math

        for name in ("alpha", "tau", "epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("support_shrinkage", "gamma_tol", "interaction_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, lower in (("min_category_count", 1), ("min_pair_support", 1), ("n_bootstrap", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError("random_state must be an integer seed")

    @staticmethod
    def _category_token(value):
        """Hashable exact-match token, with all scalar missing values equivalent."""
        np, pd = _dependencies()
        missing = pd.isna(value)
        if not np.isscalar(missing):
            raise ValueError("Categorical values must be scalar and hashable")
        if bool(missing):
            return ("missing",)
        if isinstance(value, np.generic):
            value = value.item()
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("Categorical values must be scalar and hashable") from exc
        return ("value", value)

    @staticmethod
    def _mutual_information(codes, labels, n_states):
        """Empirical MI in bits, sharing the same empirical sample distribution."""
        np, _ = _dependencies()
        counts = np.bincount(labels * n_states + codes, minlength=2 * n_states).reshape(2, n_states)
        probability = counts / counts.sum()
        expected = probability.sum(axis=1, keepdims=True) * probability.sum(axis=0, keepdims=True)
        occupied = probability > 0
        return float(np.sum(probability[occupied] * np.log2(probability[occupied] / expected[occupied])))

    def _joint_statistics(self, left, right, labels, left_size, right_size):
        np, _ = _dependencies()
        flat = left * right_size + right
        counts = np.bincount(
            labels * left_size * right_size + flat,
            minlength=2 * left_size * right_size,
        ).reshape(2, left_size, right_size)
        smoothed = counts.astype(float) + self.alpha
        joint = smoothed / smoothed.sum(axis=(1, 2), keepdims=True)
        # Consistent Dirichlet smoothing: derive both marginals from this joint.
        independent = joint.sum(axis=2, keepdims=True) * joint.sum(axis=1, keepdims=True)
        log_lift = np.log(joint / independent)
        delta = log_lift[1] - log_lift[0]
        gamma = self._mutual_information(flat, labels, left_size * right_size) - max(
            self._mutual_information(left, labels, left_size),
            self._mutual_information(right, labels, right_size),
        )
        # A tiny negative value can arise from floating-point MI arithmetic.
        return counts, joint, delta, max(0.0, gamma)

    def fit(self, X_train, y_train, categorical_cols=None):
        """Learn all encodings, probabilities, pairs and scales from training only.

        X_train may be a pandas DataFrame or an iterable of feature dictionaries.
        y_train must contain at least two nonmissing classes. Binary fitting retains
        the original arithmetic and RNG path; C>2 dispatches to multiclass_interaction.
        """
        np, pd = _dependencies()
        if len(pd.unique(np.asarray(y_train))) > 2:
            from ._multiclass import fit_multiclass
            return fit_multiclass(self, X_train, y_train, categorical_cols)
        self.n_classes_ = 2
        self._validate_parameters()
        frame = pd.DataFrame(X_train).copy(deep=True)
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("X_train must contain rows and at least one feature")
        if not frame.columns.is_unique:
            raise ValueError("Feature names must be unique")
        labels = np.asarray(y_train)
        if labels.ndim != 1 or len(labels) != len(frame):
            raise ValueError("y_train must be one-dimensional and aligned with X_train")
        if pd.isna(labels).any():
            raise ValueError("Binary labels must not be missing")
        classes = list(pd.unique(labels))
        if len(classes) != 2:
            raise ValueError("InteractionAwareRetriever currently requires exactly two classes")
        encoded_labels = (labels == classes[1]).astype(np.int64)
        feature_names = list(frame.columns)
        if categorical_cols is None:
            categorical = {
                name for name in feature_names
                if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])
            }
        else:
            categorical = set(categorical_cols)
            unknown = categorical.difference(feature_names)
            if unknown:
                raise ValueError(f"Unknown categorical columns: {sorted(unknown, key=str)}")

        # Clear fitted state before rebuilding, so a failed refit cannot expose a
        # partially rebuilt index as fitted.
        self._fitted_ = False
        self.feature_names_ = feature_names
        self.classes_ = classes
        self.categorical_cols_ = [name for name in feature_names if name in categorical]
        self.numeric_cols_ = [name for name in feature_names if name not in categorical]
        self.train_features_ = frame.reset_index(drop=True)
        self.train_index_ = frame.index.copy()
        self.train_labels_ = labels.copy()
        self.n_train_ = len(frame)
        self.encoders_ = {}
        self.numeric_stats_ = {}
        self._numeric_train_ = {}
        self._category_train_ = {}
        self.encoded_train_ = np.empty((len(frame), len(feature_names)), dtype=np.int64)
        self.feature_importances_ = {}
        self.feature_mi_ = {}
        state_sizes = []

        for column, name in enumerate(feature_names):
            if name in categorical:
                tokens = [self._category_token(value) for value in frame[name]]
                counts = {}
                for token in tokens:
                    if token != ("missing",):
                        counts[token] = counts.get(token, 0) + 1
                # Missing=0 and OTHER=1 are reserved even if absent in training.
                mapping = {token: index + 2 for index, token in enumerate(
                    token for token, count in counts.items() if count >= self.min_category_count
                )}
                codes = np.asarray([
                    0 if token == ("missing",) else mapping.get(token, 1)
                    for token in tokens
                ], dtype=np.int64)
                self.encoders_[name] = {
                    "kind": "categorical", "mapping": mapping,
                    "seen": set(counts), "n_states": len(mapping) + 2,
                }
                # Object arrays of tuple tokens can silently become 2-D; keep a list.
                self._category_train_[name] = tokens
            else:
                try:
                    values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Numeric feature {name!r} contains nonnumeric values") from exc
                valid = np.isfinite(values)
                observed = values[valid]
                if len(observed):
                    boundaries = np.unique(np.quantile(observed, [0.25, 0.5, 0.75]))
                    boundaries = boundaries[(boundaries > observed.min()) & (boundaries < observed.max())]
                    mean = float(observed.mean())
                    std = float(observed.std(ddof=0))
                    if not np.isfinite(std) or std <= 0:
                        std = 1.0
                else:
                    boundaries, mean, std = np.asarray([], dtype=float), 0.0, 1.0
                codes = np.zeros(len(frame), dtype=np.int64)
                codes[valid] = np.searchsorted(boundaries, observed, side="right") + 1
                normalized = np.full(len(frame), np.nan)
                normalized[valid] = (observed - mean) / std
                self.encoders_[name] = {
                    "kind": "numeric", "boundaries": boundaries,
                    "n_states": len(boundaries) + 2,
                }
                self.numeric_stats_[name] = {"mean": mean, "std": std}
                self._numeric_train_[name] = normalized
            n_states = self.encoders_[name]["n_states"]
            state_sizes.append(n_states)
            self.encoded_train_[:, column] = codes
            counts = np.bincount(encoded_labels * n_states + codes, minlength=2 * n_states).reshape(2, n_states)
            conditional = counts.astype(float) + self.alpha
            conditional /= conditional.sum(axis=1, keepdims=True)
            midpoint = conditional.mean(axis=0)
            mfd = 0.5 * np.sum(conditional * np.log2(conditional / midpoint))
            self.feature_importances_[name] = max(0.0, float(mfd))
            self.feature_mi_[name] = self._mutual_information(codes, encoded_labels, n_states)

        weights = np.asarray([self.feature_importances_[name] + self.epsilon for name in feature_names])
        weights /= weights.sum()
        self.feature_weights_ = dict(zip(feature_names, weights.tolist()))
        self._feature_weight_array_ = weights
        self.pair_statistics_ = {}
        self.n_pairs_considered_ = len(feature_names) * (len(feature_names) - 1) // 2
        for left, right in combinations(range(len(feature_names)), 2):
            counts, joint, delta, gamma = self._joint_statistics(
                self.encoded_train_[:, left], self.encoded_train_[:, right],
                encoded_labels, state_sizes[left], state_sizes[right],
            )
            if gamma <= self.gamma_tol:
                continue
            support = counts.sum(axis=0)
            # No state could ever contribute at this support cutoff.
            if not np.any(support >= self.min_pair_support):
                continue
            self.pair_statistics_[(feature_names[left], feature_names[right])] = {
                "columns": (left, right), "gamma": gamma, "counts": counts,
                "joint": joint, "delta": delta, "support": support,
                "stability": np.ones_like(delta),
            }

        self.bootstrap_mode_ = (
            "disabled_mechanism_degradation" if self.n_bootstrap == 0 else
            "single_resample_mechanism_degradation" if self.n_bootstrap == 1 else
            "stratified_bootstrap_sign_and_gamma_agreement"
        )
        if self.n_bootstrap:
            rng = np.random.default_rng(self.random_state)
            class_rows = [np.flatnonzero(encoded_labels == value) for value in (0, 1)]
            for pair in self.pair_statistics_.values():
                pair["stability"] = np.zeros_like(pair["delta"])
            for _ in range(self.n_bootstrap):
                sampled = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in class_rows])
                sample_labels = encoded_labels[sampled]
                for pair in self.pair_statistics_.values():
                    left, right = pair["columns"]
                    boot_counts, _, boot_delta, boot_gamma = self._joint_statistics(
                        self.encoded_train_[sampled, left], self.encoded_train_[sampled, right],
                        sample_labels, state_sizes[left], state_sizes[right],
                    )
                    agrees = (np.sign(boot_delta) == np.sign(pair["delta"])) & (np.abs(pair["delta"]) > self.epsilon)
                    supported = boot_counts.sum(axis=0) >= self.min_pair_support
                    pair["stability"] += agrees & supported & (boot_gamma > self.gamma_tol)
            for pair in self.pair_statistics_.values():
                pair["stability"] /= self.n_bootstrap

        self.metadata_ = {
            "artifact_version": self.ARTIFACT_VERSION,
            "n_train": self.n_train_, "n_features": len(feature_names),
            "n_pairs_considered": self.n_pairs_considered_,
            "n_pairs_retained": len(self.pair_statistics_),
            "bootstrap_mode": self.bootstrap_mode_, "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "mfd": "equal-prior class-conditional Jensen-Shannon divergence, log2 bits",
            "pair_information": "empirical MI(Y;pair)-max(empirical single MI)",
            "pair_support": "pooled query-state count; no both-class occupancy requirement",
            "beta_normalization": "raw/(1+sum(raw)); unit zero-interaction mass preserves shrinkage",
            "reliability_note": "Support thresholds and bootstrap agreement are heuristics, not formal significance tests.",
            "fit_scope": "training features and training labels only; query values never update statistics",
        }
        self._fitted_ = True
        return self

    def _check_fitted(self):
        if not getattr(self, "_fitted_", False):
            raise RuntimeError("Call fit before scoring, retrieving, or saving")

    def _query_state_and_similarity(self, query):
        np, pd = _dependencies()
        self._check_fitted()
        if not hasattr(query, "__getitem__"):
            raise TypeError("query must map feature names to values")
        missing = [name for name in self.feature_names_ if name not in query]
        if missing:
            raise ValueError(f"Query is missing features: {missing}")
        states, known = [], []
        similarities = np.empty((self.n_train_, len(self.feature_names_)), dtype=float)
        for column, name in enumerate(self.feature_names_):
            encoder = self.encoders_[name]
            value = query[name]
            if encoder["kind"] == "categorical":
                token = self._category_token(value)
                is_missing = token == ("missing",)
                states.append(0 if is_missing else encoder["mapping"].get(token, 1))
                known.append(is_missing or token in encoder["seen"])
                similarities[:, column] = [float(token == train_token) for train_token in self._category_train_[name]]
            else:
                try:
                    numeric = np.nan if pd.isna(value) else float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Numeric query feature {name!r} is not numeric") from exc
                train_values = self._numeric_train_[name]
                if not np.isfinite(numeric):
                    states.append(0)
                    similarities[:, column] = np.isnan(train_values).astype(float)
                else:
                    states.append(int(np.searchsorted(encoder["boundaries"], numeric, side="right")) + 1)
                    stats = self.numeric_stats_[name]
                    standardized = (numeric - stats["mean"]) / stats["std"]
                    similarities[:, column] = np.exp(-np.abs(standardized - train_values) / self.tau)
                    similarities[np.isnan(train_values), column] = 0.0
                known.append(True)
        return states, known, similarities

    def _pair_weights(self, states, known):
        if getattr(self, "n_classes_", 2) > 2:
            from ._multiclass import pair_weights_multiclass
            return pair_weights_multiclass(self, states, known)
        raw = {}
        for names, pair in self.pair_statistics_.items():
            left, right = pair["columns"]
            if not known[left] or not known[right]:
                continue
            state = (states[left], states[right])
            support = int(pair["support"][state])
            if support < self.min_pair_support:
                continue
            reliability = support / (support + self.support_shrinkage)
            value = abs(float(pair["delta"][state])) * reliability * float(pair["stability"][state])
            if value > self.epsilon:
                raw[names] = value
        # A unit of no-interaction mass prevents L1 normalization from undoing
        # support/stability shrinkage when there is only one active feature pair.
        normalizer = 1.0 + sum(raw.values())
        return {names: value / normalizer for names, value in raw.items()}

    def query_pair_weights(self, query):
        """Return active normalized beta weights for diagnostics (no query label)."""
        states, known, _ = self._query_state_and_similarity(query)
        return self._pair_weights(states, known)

    def score(self, query):
        """Return one finite similarity score per original training row, in order."""
        states, known, similarities = self._query_state_and_similarity(query)
        weights = self._pair_weights(states, known) if self.interaction_weight != 0 else {}
        result = similarities @ self._feature_weight_array_
        for names, beta in weights.items():
            left, right = self.pair_statistics_[names]["columns"]
            result += self.interaction_weight * beta * similarities[:, left] * similarities[:, right]
        denominator = self._feature_weight_array_.sum() + self.interaction_weight * sum(weights.values()) + self.epsilon
        return result / denominator

    def retrieve(self, query, k=5):
        """Return (positional indices, scores), descending with stable row-order ties."""
        np, _ = _dependencies()
        self._check_fitted()
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        scores = self.score(query)
        indices = np.argsort(-scores, kind="stable")[:min(k, self.n_train_)]
        return indices, scores[indices]

    def save(self, path):
        """Save versioned local pickle, including training rows and frozen statistics.

        Pickle is for trusted local artifacts only: do not load untrusted files.
        """
        self._check_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "mfd_pair.InteractionAwareRetriever", "version": self.ARTIFACT_VERSION,
            "metadata": self.metadata_, "state": self.__dict__,
        }
        with destination.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path):
        """Replay a trusted local artifact; pickle can execute code while loading."""
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format") != "mfd_pair.InteractionAwareRetriever":
            raise ValueError("Not an InteractionAwareRetriever artifact")
        if payload.get("version") != cls.ARTIFACT_VERSION:
            raise ValueError(f"Unsupported InteractionAwareRetriever artifact version: {payload.get('version')}")
        retriever = cls()
        retriever.__dict__.update(payload["state"])
        retriever._check_fitted()
        return retriever
