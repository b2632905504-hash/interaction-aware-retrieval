import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from mfd_pair import InteractionAwareRetriever, MFDPairRetriever, parse_prediction, save_corpus


class MethodTests(unittest.TestCase):
    @staticmethod
    def binary():
        X = pd.DataFrame([(a, b) for a in range(2) for b in range(2)] * 32, columns=['a', 'b'])
        return X, np.asarray(X.a ^ X.b)

    def test_binary_joint_signal_and_class_quotas(self):
        X, y = self.binary()
        retriever = MFDPairRetriever(n_bootstrap=4).fit(X, y, ['a', 'b'])
        self.assertLess(max(retriever.mfd_scores_.values()), 1e-10)
        result = retriever.retrieve({'a': 0, 'b': 1})
        self.assertEqual(len(result.indices), 16)
        self.assertEqual(result.label_counts, {'class_0': 8, 'class_1': 8})
        self.assertTrue(result.pair_weights)
        np.testing.assert_array_equal(result.scores, np.sort(result.scores)[::-1])

    def test_multiclass_and_label_renaming(self):
        X = pd.DataFrame([(a, b) for a in range(3) for b in range(3)] * 24, columns=['a', 'b'])
        y = np.asarray((X.a + X.b) % 3)
        first = MFDPairRetriever(n_bootstrap=3).fit(X, y, ['a', 'b'])
        second = MFDPairRetriever(n_bootstrap=3).fit(X, np.asarray(['label_' + str(v) for v in y]), ['a', 'b'])
        query = {'a': 1, 'b': 1}
        np.testing.assert_array_equal(first.score(query), second.score(query))
        result = first.retrieve(query)
        self.assertEqual(len(set(result.indices)), 16)
        self.assertEqual(sorted(result.label_counts.values()), [5, 5, 6])
        self.assertTrue(result.pair_weights)

    def test_feature_budget_precedes_pair_mining(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(160, 19)), columns=[f'x{i}' for i in range(19)])
        y = (X.x0 + X.x1 > 0).astype(int)
        r = MFDPairRetriever(max_features=15, n_bootstrap=2).fit(X, y, [])
        self.assertEqual(len(r.selected_features_), 15)
        self.assertTrue(all(set(pair).issubset(r.selected_features_) for pair in r.index_.pair_statistics_))
        self.assertEqual(r.index_.n_pairs_considered_, 105)

    def test_query_labels_change_only_scoring_output(self):
        X, y = self.binary()
        r = MFDPairRetriever(n_bootstrap=2).fit(X, y, ['a', 'b'])
        queries = X.iloc[:4]
        first = r.build_corpus(queries, query_labels=y[:4])
        second = r.build_corpus(queries, query_labels=1-y[:4])
        self.assertEqual(first['retrievals'], second['retrievals'])
        for a, b in zip(first['records'], second['records']):
            self.assertEqual((a['instruction'], a['input']), (b['instruction'], b['input']))
            self.assertNotEqual(a['output'], b['output'])
        self.assertTrue(all(row['output'] == '' for row in r.build_corpus(queries)['records']))

    def test_missing_unseen_and_roundtrip(self):
        X = pd.DataFrame({'a': ['p', 'q', None, 'p']*24, 'b': [0., 1., np.nan, 2.]*24})
        y = [0, 1, 0, 1]*24
        model = InteractionAwareRetriever(n_bootstrap=2).fit(X, y, ['a'])
        query = {'a': 'unseen', 'b': None}
        self.assertTrue(np.isfinite(model.score(query)).all())
        self.assertEqual(model.query_pair_weights(query), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'index.pkl'
            model.save(path)
            restored = InteractionAwareRetriever.load(path)
            np.testing.assert_array_equal(model.score(query), restored.score(query))

    def test_exclusion_refill_and_zero_budget(self):
        X = pd.DataFrame({'x': np.arange(20.)})
        r = MFDPairRetriever(n_bootstrap=2).fit(X, [0]*2+[1]*18, [])
        result = r.retrieve({'x': 1.}, exclude_position=1)
        self.assertEqual(len(set(result.indices)), 16)
        self.assertNotIn(1, result.indices)
        self.assertEqual(len(r.retrieve({'x': 1.}, k=0).indices), 0)
        with self.assertRaises(ValueError):
            r.retrieve({'x': 1.}, k=-1)

    def test_saved_corpus_matches_selected_rows(self):
        X, y = self.binary()
        r = MFDPairRetriever(n_bootstrap=2).fit(X, y, ['a', 'b'])
        bundle = r.build_corpus(X.iloc[:2])
        with tempfile.TemporaryDirectory() as directory:
            path = save_corpus(bundle, directory)
            rows = json.loads(path.read_text())
            self.assertEqual(rows, bundle['records'])
            self.assertIn('=== Similar Historical Cases ===', rows[0]['input'])
            self.assertEqual(rows[0]['input'].count('Ground truth:'), 16)
            with self.assertRaises(FileExistsError):
                save_corpus(bundle, directory)

    def test_final_prediction_parser(self):
        self.assertEqual(parse_prediction('Prediction is class_0.\nPrediction is **class_1**.', 2), 'class_1')
        self.assertIsNone(parse_prediction('Prediction is class_7.', 2))
        self.assertIsNone(parse_prediction('No prediction.', 2))

    def test_training_input_validation(self):
        with self.assertRaises(ValueError):
            MFDPairRetriever(max_features=0)
        with self.assertRaises(ValueError):
            MFDPairRetriever().fit(pd.DataFrame({'x': [0., 1.]}), [0, 0])
        r = MFDPairRetriever(n_bootstrap=2).fit(pd.DataFrame({'x': [0., 1.]*10}), [0, 1]*10)
        with self.assertRaises(ValueError):
            r.retrieve({'other': 1.})


if __name__ == '__main__':
    unittest.main()
