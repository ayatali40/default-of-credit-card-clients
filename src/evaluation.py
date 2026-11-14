"""Evaluation helpers: metrics, threshold analysis, plots."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
DEFAULT_THRESHOLDS = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
POSITIVE_LABEL = 'Default'
NEGATIVE_LABEL = 'No default'

def positive_class_probabilities(estimator, X) -> np.ndarray:
    if not hasattr(estimator, 'predict_proba'):
        raise TypeError(f'{type(estimator).__name__} does not expose predict_proba; threshold-based evaluation requires probability estimates.')
    classes = getattr(estimator, 'classes_', None)
    if classes is None:
        raise ValueError('Estimator does not expose classes_; fit it before requesting positive-class probabilities.')
    classes = np.asarray(classes)
    positive_positions = np.flatnonzero(classes == 1)
    if positive_positions.size != 1:
        raise ValueError(f'Positive class 1 is unavailable in estimator.classes_={classes.tolist()}.')
    output = np.asarray(estimator.predict_proba(X))
    if output.ndim != 2 or output.shape[1] != classes.size:
        raise ValueError(f'predict_proba returned an unexpected shape {output.shape}; expected (n_samples, {classes.size}).')
    if len(output) != len(X):
        raise ValueError('predict_proba returned a different number of rows than X.')
    probabilities = output[:, positive_positions[0]]
    if not np.isfinite(probabilities).all():
        raise ValueError('predict_proba returned non-finite probabilities.')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('predict_proba returned probabilities outside [0, 1].')
    return probabilities

def _as_index_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f'{name} indices must be one-dimensional.')
    numeric = pd.to_numeric(pd.Series(array), errors='coerce')
    if numeric.isna().any():
        raise ValueError(f'{name} indices must be integers.')
    floats = numeric.to_numpy(dtype=float)
    if not np.equal(floats, np.floor(floats)).all():
        raise ValueError(f'{name} indices must be integers.')
    return floats.astype(int)

def _validate_oof_splits(fold_splits, n_samples: int) -> None:
    fold_splits = list(fold_splits)
    if not fold_splits:
        raise ValueError('splits must contain at least one fold.')
    assigned = np.full(n_samples, -1, dtype=int)
    for fold, pair in enumerate(fold_splits):
        if len(pair) != 2:
            raise ValueError('Each split must be a (train, validation) pair.')
        train_index = _as_index_array(pair[0], 'Training')
        held_out_index = _as_index_array(pair[1], 'Validation')
        for name, index in (('Training', train_index), ('Validation', held_out_index)):
            if index.size == 0:
                raise ValueError(f'{name} indices for fold {fold} are empty.')
            if ((index < 0) | (index >= n_samples)).any():
                raise ValueError(f'{name} indices for fold {fold} are outside [0, {n_samples}).')
        if np.intersect1d(train_index, held_out_index).size:
            raise ValueError(f'Training and validation indices overlap in fold {fold}.')
        if np.unique(held_out_index).size != held_out_index.size:
            raise ValueError(f'Validation indices are duplicated within fold {fold}.')
        if (assigned[held_out_index] != -1).any():
            raise ValueError(f'Validation indices are reused across folds at fold {fold}.')
        assigned[held_out_index] = fold
    if (assigned < 0).any():
        missing = int((assigned < 0).sum())
        raise ValueError(f'Out-of-fold splits leave {missing} record(s) without exactly one held-out prediction.')

def out_of_fold_probabilities(pipeline, X, y, n_splits: int=5, random_state: int=42, splits=None) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(X).reset_index(drop=True)
    target = pd.Series(y).reset_index(drop=True)
    if len(frame) != len(target):
        raise ValueError('X and y must contain the same number of records.')
    probabilities = np.full(len(target), np.nan, dtype=float)
    fold_ids = np.full(len(target), -1, dtype=int)
    if splits is None:
        if n_splits < 2:
            raise ValueError('n_splits must be at least 2 for out-of-fold predictions.')
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_splits = splitter.split(frame, target)
    else:
        fold_splits = list(splits)
        _validate_oof_splits(fold_splits, len(target))
    for fold, (train_index, held_out_index) in enumerate(fold_splits):
        estimator = clone(pipeline)
        estimator.fit(frame.iloc[train_index], target.iloc[train_index])
        probabilities[held_out_index] = positive_class_probabilities(estimator, frame.iloc[held_out_index])
        fold_ids[held_out_index] = fold
    if np.isnan(probabilities).any() or (fold_ids < 0).any():
        raise RuntimeError('Out-of-fold prediction did not cover every record.')
    return (probabilities, fold_ids)

def permutation_importance_table(estimator, X: pd.DataFrame, y, scoring: str='average_precision', n_repeats: int=10, random_state: int=42, n_jobs: int=1) -> pd.DataFrame:
    result = permutation_importance(estimator, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
    frame = pd.DataFrame({'feature': list(X.columns), 'importance': result.importances_mean, 'importance_std': result.importances_std})
    frame['method'] = f'permutation ({scoring})'
    frame['n_repeats'] = int(n_repeats)
    return frame.sort_values('importance', ascending=False).reset_index(drop=True)

def confusion_counts(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def compute_metrics(y_true, probabilities: np.ndarray, threshold: float=0.5) -> dict[str, float | int]:
    raw_labels = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if raw_labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError('y_true and probabilities must be one-dimensional.')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must lie between 0 and 1.')
    if raw_labels.size == 0:
        raise ValueError('y_true and probabilities must contain at least one record.')
    if raw_labels.size != probabilities.size:
        raise ValueError('y_true and probabilities must have the same length.')
    labels = pd.to_numeric(pd.Series(raw_labels), errors='coerce')
    if labels.isna().any():
        raise ValueError('y_true must contain only the labels 0 and 1.')
    label_values = labels.to_numpy(dtype=float)
    if not np.isin(label_values, [0.0, 1.0]).all():
        raise ValueError('y_true must contain only the labels 0 and 1.')
    y_true = label_values.astype(int)
    if not np.isfinite(probabilities).all():
        raise ValueError('probabilities must be finite.')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('probabilities must lie between 0 and 1.')
    y_pred = (probabilities >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    has_both_classes = len(np.unique(y_true)) == 2
    return {'threshold': float(threshold), 'n_samples': len(y_true), 'n_positive': int(y_true.sum()), 'accuracy': float(accuracy_score(y_true, y_pred)), 'precision': float(precision_score(y_true, y_pred, zero_division=0)), 'recall': float(recall_score(y_true, y_pred, zero_division=0)), 'f1': float(f1_score(y_true, y_pred, zero_division=0)), 'roc_auc': float(roc_auc_score(y_true, probabilities)) if has_both_classes else float('nan'), 'pr_auc': float(average_precision_score(y_true, probabilities)) if has_both_classes else float('nan'), **counts, 'predicted_positive_rate': float(y_pred.mean())}

def threshold_analysis_table(y_true, probabilities: np.ndarray, thresholds: list[float] | None=None) -> pd.DataFrame:
    values = thresholds or DEFAULT_THRESHOLDS
    rows = [compute_metrics(y_true, probabilities, threshold) for threshold in values]
    return pd.DataFrame(rows)

def select_threshold(y_true, probabilities: np.ndarray, thresholds: list[float] | None=None, metric: str='f1') -> float:
    table = threshold_analysis_table(y_true, probabilities, thresholds)
    if metric not in table.columns:
        raise ValueError(f'Unknown threshold metric: {metric}')
    ordered = table.sort_values(by=[metric, 'recall', 'threshold'], ascending=[False, False, False])
    return float(ordered.iloc[0]['threshold'])

def classification_report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=[NEGATIVE_LABEL, POSITIVE_LABEL], output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).transpose()
    return frame.reset_index().rename(columns={'index': 'class'})

def get_feature_importances(pipeline, top_n: int=30) -> pd.DataFrame | None:
    if 'preprocessor' not in pipeline.named_steps or 'model' not in pipeline.named_steps:
        raise ValueError("Expected a pipeline with 'preprocessor' and 'model' steps.")
    feature_names = np.asarray(pipeline.named_steps['preprocessor'].get_feature_names_out())
    model = pipeline.named_steps['model']
    values: np.ndarray | None = None
    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, 'coef_'):
        values = np.abs(np.ravel(model.coef_)).astype(float)
    elif hasattr(model, 'calibrated_classifiers_'):
        coefficients = []
        for calibrated in model.calibrated_classifiers_:
            estimator = getattr(calibrated, 'estimator', None)
            if estimator is not None and hasattr(estimator, 'coef_'):
                coefficients.append(np.abs(np.ravel(estimator.coef_)))
        if coefficients:
            values = np.mean(coefficients, axis=0)
    if values is None:
        return None
    size = min(len(feature_names), len(values))
    frame = pd.DataFrame({'feature': feature_names[:size], 'importance': values[:size]})
    frame['feature'] = frame['feature'].str.replace('numeric__', '', regex=False).str.replace('categorical__', '', regex=False)
    return frame.sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)

def _save_figure(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, path: Path, title: str='Confusion matrix') -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=',d', cmap='Blues', cbar=False, xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL], yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL], ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title(title)
    _save_figure(path)
    return Path(path)

def plot_roc_curves(y_true, probability_by_model: dict[str, np.ndarray], path: Path, title: str='ROC curves') -> Path:
    _fig, ax = plt.subplots(figsize=(8, 6))
    for name, probabilities in probability_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_value:.3f})', linewidth=1.8)
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Random')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower right')
    _save_figure(path)
    return Path(path)

def plot_precision_recall_curves(y_true, probability_by_model: dict[str, np.ndarray], path: Path, title: str='Precision-Recall curves') -> Path:
    baseline = float(np.mean(y_true))
    _fig, ax = plt.subplots(figsize=(8, 6))
    for name, probabilities in probability_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap_value = average_precision_score(y_true, probabilities)
        ax.plot(recall, precision, label=f'{name} (AP={ap_value:.3f})', linewidth=1.8)
    ax.axhline(baseline, linestyle='--', color='grey', linewidth=1, label=f'Baseline={baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower left')
    _save_figure(path)
    return Path(path)

def plot_model_comparison(results: pd.DataFrame, path: Path, metrics: tuple[str, ...]=('pr_auc', 'roc_auc', 'f1', 'recall', 'precision'), title: str='Validation performance by model') -> Path:
    available = [metric for metric in metrics if metric in results.columns]
    frame = results.set_index('model')[available].sort_values(available[0])
    _fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(frame) + 1.5)))
    frame.plot(kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('Score')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    _save_figure(path)
    return Path(path)

def plot_threshold_analysis(table: pd.DataFrame, selected_threshold: float, path: Path, title: str='Threshold trade-offs on validation data') -> Path:
    _fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ['precision', 'recall', 'f1']:
        ax.plot(table['threshold'], table[metric], marker='o', markersize=3, label=metric.capitalize())
    ax.axvline(selected_threshold, color='black', linestyle='--', label=f'Selected={selected_threshold:.2f}')
    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save_figure(path)
    return Path(path)

def plot_feature_importance(importance: pd.DataFrame, path: Path, title: str='Top feature importances', xlabel: str='Importance') -> Path:
    frame = importance.sort_values('importance', ascending=True)
    _fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(frame) + 1.5)))
    errors = frame['importance_std'] if 'importance_std' in frame.columns else None
    ax.barh(frame['feature'], frame['importance'], xerr=errors, color='#2a6f97', error_kw={'elinewidth': 0.8, 'capsize': 2, 'ecolor': '#333333'})
    ax.axvline(0, color='#555555', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    _save_figure(path)
    return Path(path)