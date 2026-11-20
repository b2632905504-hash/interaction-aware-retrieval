"""Generic caller-provided CSV inputs; no registered tables or downloads."""
import argparse
from pathlib import Path
import pandas as pd
from .pipeline import MFDPairRetriever, save_corpus


def main():
    parser = argparse.ArgumentParser(description='Build a corpus using MFD+PAIR retrieval')
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--query-csv', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--categorical', nargs='*', default=None)
    parser.add_argument('--max-features', type=int, default=15)
    parser.add_argument('--demonstrations', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--instruction-file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.train_csv, na_values=['?', 'missing'])
    queries = pd.read_csv(args.query_csv, na_values=['?', 'missing'])
    if args.target not in train:
        parser.error('The training table must contain the target column')
    y_train = train.pop(args.target)
    y_query = queries.pop(args.target) if args.target in queries else None
    method = MFDPairRetriever(max_features=args.max_features, demonstrations=args.demonstrations, seed=args.seed)
    method.fit(train, y_train, categorical_features=args.categorical)
    options = {'query_labels': y_query}
    if args.instruction_file:
        options['instruction'] = Path(args.instruction_file).read_text()
    path = save_corpus(method.build_corpus(queries, **options), args.output)
    print(path)


if __name__ == '__main__':
    main()
