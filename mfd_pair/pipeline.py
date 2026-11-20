"""MFD feature budget, fitted pair statistics, and class-balanced retrieval."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .interaction import InteractionAwareRetriever


@dataclass
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray
    label_counts: dict
    pair_weights: dict


class MFDPairRetriever:
    """Rank all fields by train-only MFD; mine pairs inside the selected subset.

    Returned indices are positions in the original training frame, regardless
    of its pandas index. Query labels are never accepted by the retriever.
    """

    def __init__(self, max_features=15, demonstrations=16, seed=42, **interaction_options):
        if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1:
            raise ValueError('max_features must be a positive integer')
        if isinstance(demonstrations, bool) or not isinstance(demonstrations, int) or demonstrations < 1:
            raise ValueError('demonstrations must be a positive integer')
        if 'random_state' in interaction_options:
            raise ValueError('Use seed to configure the shared random state')
        self._fitted = False
        self.max_features = max_features
        self.demonstrations = demonstrations
        self.seed = seed
        self.interaction_options = dict(interaction_options)

    def fit(self, features, labels, categorical_features=None):
        self._fitted = False
        frame = pd.DataFrame(features).copy(deep=True)
        values = np.asarray(labels)
        if frame.empty or not frame.columns.is_unique or not all(isinstance(name, str) for name in frame):
            raise ValueError('Provide a nonempty feature frame with unique string column names')
        if values.ndim != 1 or len(values) != len(frame) or pd.isna(values).any():
            raise ValueError('Training labels must be one-dimensional, nonmissing, and aligned')
        self.classes_ = list(pd.unique(values))
        if len(self.classes_) < 2:
            raise ValueError('At least two training classes are required')
        categorical = list(categorical_features) if categorical_features is not None else [
            name for name in frame if not pd.api.types.is_numeric_dtype(frame[name])
            or pd.api.types.is_bool_dtype(frame[name])]
        if not set(categorical).issubset(frame):
            raise ValueError('Unknown categorical fields')
        # Match the missing-value convention of the current table adapter.
        for name in categorical:
            frame[name] = frame[name].replace({'?': np.nan, 'missing': np.nan})
        self.candidate_features_ = list(frame.columns)
        self.train_features_ = frame.reset_index(drop=True)
        self.train_labels_ = values.copy()
        self.class_ids_ = pd.factorize(values, sort=False)[0]
        self.mfd_scores_ = {}
        marginal_options = dict(self.interaction_options, n_bootstrap=0)
        for name in frame:
            marginal = InteractionAwareRetriever(random_state=self.seed, **marginal_options).fit(
                frame[[name]], values, [name] if name in categorical else [])
            self.mfd_scores_[name] = float(marginal.feature_importances_[name])
        self.feature_order_ = sorted(frame.columns, key=lambda name: -self.mfd_scores_[name])
        self.selected_features_ = self.feature_order_[:self.max_features]
        self.presentation_features_ = [name for name in frame if name in self.selected_features_]
        self.categorical_features_ = [name for name in self.selected_features_ if name in categorical]
        self.index_ = InteractionAwareRetriever(random_state=self.seed, **self.interaction_options).fit(
            frame[self.selected_features_], values, self.categorical_features_)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError('Call fit before retrieval')

    def _clean_query(self, query):
        self._check_fitted()
        missing = [name for name in self.selected_features_ if name not in query]
        if missing:
            raise ValueError(f'Query is missing selected fields: {missing}')
        return {name: None if isinstance(query[name], str) and query[name] in {'?', 'missing'} else query[name]
                for name in self.selected_features_}

    def score(self, query):
        self._check_fitted()
        return self.index_.score(self._clean_query(query))

    def retrieve(self, query, k=None, *, exclude_position=None):
        self._check_fitted()
        k = self.demonstrations if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError('k must be a nonnegative integer')
        clean = self._clean_query(query)
        scores = self.index_.score(clean)
        candidates = np.arange(len(self.train_labels_))
        if exclude_position is not None:
            if isinstance(exclude_position, bool) or not isinstance(exclude_position, (int, np.integer)) or not 0 <= exclude_position < len(candidates):
                raise ValueError('exclude_position must identify a training row')
            candidates = candidates[candidates != exclude_position]
        selected = []
        if k and len(candidates):
            if len(self.classes_) == 2:
                # Preserve the current binary quota and tie-handling arithmetic:
                # positive class first, odd slot to class zero, then refill.
                for class_id, quota in [(1, k // 2), (0, k // 2 + k % 2)]:
                    pool = candidates[self.class_ids_[candidates] == class_id]
                    ranked = pool[np.argsort(-scores[pool])]
                    selected.extend(int(index) for index in ranked[:quota])
            else:
                active_classes = list(pd.unique(self.class_ids_[candidates]))
                count = len(active_classes)
                pools = {}
                for class_id in active_classes:
                    pool = candidates[self.class_ids_[candidates] == class_id]
                    ranked = pool[np.argsort(-scores[pool])].tolist()
                    take = min(k // count, len(ranked))
                    selected.extend(ranked[:take])
                    pools[class_id] = ranked[take:]
                priority = sorted(active_classes, key=lambda value: scores[pools[value][0]] if pools[value] else float('-inf'), reverse=True)
                remainder = k % count
                for class_id in priority:
                    if not remainder:
                        break
                    if pools[class_id]:
                        selected.append(pools[class_id].pop(0))
                        remainder -= 1
            if len(selected) < min(k, len(candidates)):
                used = set(selected)
                remaining = [int(index) for index in candidates if index not in used]
                remaining.sort(key=lambda index: -float(scores[index]))
                selected.extend(remaining[:k - len(selected)])
            selected.sort(key=lambda index: -float(scores[index]))
        positions = np.asarray(selected, dtype=np.int64)
        counts = {f'class_{i}': int(np.sum(self.class_ids_[positions] == i)) for i in range(len(self.classes_))}
        return RetrievalResult(positions, scores[positions], counts, self.index_.query_pair_weights(clean))

    def metadata(self):
        self._check_fitted()
        return {'max_features': self.max_features, 'demonstrations': self.demonstrations, 'seed': self.seed,
            'interaction_options': {name: getattr(self.index_, name) for name in ('alpha', 'min_category_count', 'min_pair_support', 'support_shrinkage', 'n_bootstrap', 'gamma_tol', 'interaction_weight', 'tau', 'epsilon')}, 'selected_features': list(self.selected_features_),
            'feature_order': list(self.feature_order_), 'mfd_scores': dict(self.mfd_scores_),
            'presentation_features': list(self.presentation_features_), 'classes': list(self.classes_),
            'categorical_features': list(self.categorical_features_),
            'pair_search_scope': 'selected_features_only', 'index': dict(self.index_.metadata_)}

    def build_corpus(self, query_features, *, instruction='Predict the class of the current sample.',
                     feature_descriptions=None, include_feature_explanations=True, query_labels=None):
        self._check_fitted()
        frame = pd.DataFrame(query_features)
        if not set(self.selected_features_).issubset(frame):
            raise ValueError('Query frame is missing selected fields')
        targets = None if query_labels is None else list(query_labels)
        if targets is not None and len(targets) != len(frame):
            raise ValueError('Query labels must align with query rows')
        class_map = {value: i for i, value in enumerate(self.classes_)}
        if targets is not None and any(pd.isna(value) or value not in class_map for value in targets):
            raise ValueError('Query labels must use the frozen training class vocabulary')
        system = instruction.strip() + ' Please explain your reasoning.\n\n'
        system += '\n'.join(f'class_{i} means {value}.' for i, value in enumerate(self.classes_))
        system += '\nAt the very end, output exactly: Prediction is class_i. Replace class_i with a valid class token.'
        if include_feature_explanations:
            descriptions = feature_descriptions or {}
            system += '\n\nFeature explanations:\n' + '\n'.join(
                f'- {name}: {descriptions.get(name, name)}' for name in self.selected_features_)
        system += ' You are also provided with similar historical cases to assist your reasoning.'
        records, traces = [], []
        for position in range(len(frame)):
            query = {name: frame[name].iloc[position] for name in self.selected_features_}
            retrieved = self.retrieve(query)
            lines = ['--- Current Sample Features ---', self._serialize(query), '', '=== Similar Historical Cases ===']
            for rank, (index, score) in enumerate(zip(retrieved.indices, retrieved.scores), 1):
                values = {name: self.train_features_[name].iloc[index] for name in self.selected_features_}
                token = f'class_{self.class_ids_[index]}'
                lines.extend([f'Example {rank} (similarity: {score:.3f}):',
                    f'  Features: {self._serialize(values)}', f'  Ground truth: The label for this case is: {token}'])
            records.append({'instruction': system, 'input': '\n'.join(lines),
                'output': '' if targets is None else f'class_{class_map[targets[position]]}'})
            traces.append({'query_position': position, 'train_positions': retrieved.indices.tolist(),
                'similarities': retrieved.scores.tolist(), 'label_counts': retrieved.label_counts,
                'pair_weights': [{'features': list(pair), 'weight': weight} for pair, weight in retrieved.pair_weights.items()]})
        return {'records': records, 'retrievals': traces, 'metadata': self.metadata()}

    def _serialize(self, values):
        parts = []
        for name in self.presentation_features_:
            value = values[name]
            if pd.isna(value) or (isinstance(value, str) and value in {'?', 'missing'}):
                value = 'missing'
            parts.append(f'{name} is {value}')
        return '. '.join(parts)


def parse_prediction(text, n_classes):
    """Return the last valid explicit class token, or None on parsing failure."""
    matches = re.findall(r'prediction\s+is\s+\**(class_\d+)\b', str(text), re.I)
    allowed = {f'class_{i}' for i in range(n_classes)}
    return next((token.lower() for token in reversed(matches) if token.lower() in allowed), None)


def save_corpus(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [directory / 'corpus.json', directory / 'retrieval.jsonl', directory / 'metadata.json']
    if any(path.exists() for path in files):
        raise FileExistsError('Use a new output directory')
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Not JSON serializable: {type(value).__name__}')
    files[0].write_text(json.dumps(bundle['records'], ensure_ascii=False, indent=2, default=convert) + '\n')
    files[1].write_text(''.join(json.dumps(row, ensure_ascii=False, default=convert) + '\n' for row in bundle['retrievals']))
    files[2].write_text(json.dumps(bundle['metadata'], ensure_ascii=False, indent=2, default=convert) + '\n')
    return files[0]
