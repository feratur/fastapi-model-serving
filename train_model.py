#!/usr/bin/env python3
import click
import logging
import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_filepath, output_dir):
    logger = logging.getLogger(__name__)

    logger.info(f'Loading dataset from {input_filepath}...')
    data = pd.read_csv(input_filepath)

    X = data.drop(columns=['target'])
    y = data['target']

    logger.info(f'Total rows: {len(data)} (using 20% as test)')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    clf = RandomForestClassifier(random_state=123)
    logger.info(f'Training RandomForestClassifier...')
    clf.fit(X_train, y_train)

    logger.info(f'Training completed, estimating accuracy...')
    accuracy_percent = round(accuracy_score(y_test, clf.predict(X_test)) * 100, 2)
    logger.info(f'Model accuracy is {accuracy_percent}%')

    model_path = os.path.join(output_dir, 'model.joblib')
    dump(clf, model_path)
    logger.info(f'Model saved to {model_path}')

    columns_path = os.path.join(output_dir, 'columns.txt')
    with open(columns_path, 'w') as f:
        f.writelines([f'{col}\n' for col in X_test.columns])
    logger.info(f'Column names saved to {columns_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
