"""Multiclass extension; binary arithmetic and RNG stay in interaction.py unchanged."""
from itertools import combinations
from .interaction import _dependencies

def fit_multiclass(self, X_train, y_train, categorical_cols=None):
    """Learn all encodings, probabilities, pairs and scales from training only.

    X_train may be a pandas DataFrame or an iterable of feature dictionaries.
    y_train must contain at least three nonmissing classes. Their encounter order
    defines contiguous class IDs; scores use absolute pairwise class contrasts.
    """
    np, pd = _dependencies()
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
        raise ValueError("Class labels must not be missing")
    classes = list(pd.unique(labels))
    if len(classes) < 3:
        raise ValueError("Multiclass fitting requires at least three classes")
    encoded_labels = pd.factorize(labels, sort=False)[0].astype(np.int64)
    n_classes = len(classes)
    self.n_classes_ = n_classes
    self.class_pairs_ = list(combinations(range(n_classes), 2))
    def mutual_information(codes, y, n_states):
        counts = np.bincount(y*n_states+codes, minlength=n_classes*n_states).reshape(n_classes,n_states)
        probability = counts/counts.sum()
        expected = probability.sum(axis=1,keepdims=True)*probability.sum(axis=0,keepdims=True)
        occupied = probability>0
        return float(np.sum(probability[occupied]*np.log2(probability[occupied]/expected[occupied])))

    def joint_statistics(left,right,y,left_size,right_size):
        flat = left*right_size+right
        counts = np.bincount(y*left_size*right_size+flat,
            minlength=n_classes*left_size*right_size).reshape(n_classes,left_size,right_size)
        joint = counts.astype(float)+self.alpha
        joint /= joint.sum(axis=(1,2),keepdims=True)
        independent = joint.sum(axis=2,keepdims=True)*joint.sum(axis=1,keepdims=True)
        lift = np.log(joint/independent)
        delta = np.stack([lift[d]-lift[c] for c,d in self.class_pairs_])
        gamma = mutual_information(flat,y,left_size*right_size)-max(
            mutual_information(left,y,left_size),mutual_information(right,y,right_size))
        return counts,joint,delta,max(0.0,gamma)

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
        counts = np.bincount(encoded_labels * n_states + codes, minlength=n_classes * n_states).reshape(n_classes, n_states)
        conditional = counts.astype(float) + self.alpha
        conditional /= conditional.sum(axis=1, keepdims=True)
        midpoint = conditional.mean(axis=0)
        mfd = (1.0 / n_classes) * np.sum(conditional * np.log2(conditional / midpoint))
        self.feature_importances_[name] = max(0.0, float(mfd))
        self.feature_mi_[name] = mutual_information(codes, encoded_labels, n_states)

    weights = np.asarray([self.feature_importances_[name] + self.epsilon for name in feature_names])
    weights /= weights.sum()
    self.feature_weights_ = dict(zip(feature_names, weights.tolist()))
    self._feature_weight_array_ = weights
    self.pair_statistics_ = {}
    self.n_pairs_considered_ = len(feature_names) * (len(feature_names) - 1) // 2
    for left, right in combinations(range(len(feature_names)), 2):
        counts, joint, delta, gamma = joint_statistics(
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
        class_rows = [np.flatnonzero(encoded_labels == value) for value in range(n_classes)]
        for pair in self.pair_statistics_.values():
            pair["stability"] = np.zeros_like(pair["delta"])
        for _ in range(self.n_bootstrap):
            sampled = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in class_rows])
            sample_labels = encoded_labels[sampled]
            for pair in self.pair_statistics_.values():
                left, right = pair["columns"]
                boot_counts, _, boot_delta, boot_gamma = joint_statistics(
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
        "n_train": self.n_train_, "n_features": len(feature_names), "n_classes": n_classes,
        "class_contrast": "mean absolute pairwise class log-lift contrast weighted by per-contrast bootstrap stability",
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


def pair_weights_multiclass(self, states, known):
    np, _ = _dependencies()
    raw = {}
    for names, pair in self.pair_statistics_.items():
        left,right = pair['columns']
        if not known[left] or not known[right]:
            continue
        a,b = states[left],states[right]
        support = int(pair['support'][a,b])
        if support < self.min_pair_support:
            continue
        contrast = np.abs(pair['delta'][:,a,b])*pair['stability'][:,a,b]
        value = float(contrast.mean())*support/(support+self.support_shrinkage)
        if value>self.epsilon:
            raw[names]=value
    normalizer=1.0+sum(raw.values())
    return {names:value/normalizer for names,value in raw.items()}
