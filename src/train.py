"""Training pipeline for credit card default prediction."""

from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.data_cleaning import clean_dataset, count_unexpected_codes, data_quality_report
from src.data_loader import PROJECT_ROOT, RAW_DATA_DIR, TARGET_COLUMN, find_dataset_file, load_raw_dataset
from src.evaluation import classification_report_frame, compute_metrics, get_feature_importances, out_of_fold_probabilities, permutation_importance_table, plot_confusion_matrix, plot_feature_importance, plot_model_comparison, plot_precision_recall_curves, plot_roc_curves, plot_threshold_analysis, positive_class_probabilities, select_threshold, threshold_analysis_table
from src.fairness import fairness_report, plot_subgroup_performance, subgroup_performance
from src.feature_engineering import ENGINEERED_FEATURES, engineer_features
from src.preprocessing import RANDOM_STATE, build_model_pipeline, build_preprocessor, get_feature_lists, split_dataset
from src.selection import REFERENCE_THRESHOLD, apply_selection_policy, expected_cost_table, model_selection_policy
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
IMAGES_DIR = PROJECT_ROOT / 'images'
MODEL_FACTORIES = {'Logistic Regression': lambda: LogisticRegression(max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE), 'Decision Tree': lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE), 'Random Forest': lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE), 'K-Nearest Neighbors': lambda: KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=-1), 'Support Vector Machine': lambda: CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000, random_state=RANDOM_STATE), cv=3, method='sigmoid'), 'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE)}
TUNING_GRIDS = {'Logistic Regression': {'model__C': [0.01, 0.1, 1.0, 10.0], 'model__class_weight': [None, 'balanced']}, 'Decision Tree': {'model__max_depth': [4, 6, 8, None], 'model__min_samples_leaf': [20, 50, 100], 'model__class_weight': [None, 'balanced']}, 'Random Forest': {'model__max_depth': [8, 12, 20, None], 'model__min_samples_leaf': [1, 5, 10], 'model__n_estimators': [200, 300], 'model__max_features': ['sqrt', 0.5]}, 'Gradient Boosting': {'model__n_estimators': [150, 250], 'model__learning_rate': [0.03, 0.05, 0.1], 'model__max_depth': [2, 3], 'model__subsample': [0.8, 1.0]}}
METRIC_COLUMNS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tn', 'fp', 'fn', 'tp', 'predicted_positive_rate']

def _relative_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value

def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(_clean_json(payload), handle, indent=2, ensure_ascii=False)
    return path

def _validation_row(name: str, role: str, handling: str, metrics: dict, fit_time: float, predict_time: float, cv_stats: dict | None=None, cv_time: float | None=None) -> dict:
    row = {'model': name, 'role': role, 'imbalance_handling': handling}
    row.update({f'validation_{key}': metrics[key] for key in METRIC_COLUMNS})
    row['validation_threshold'] = metrics['threshold']
    row['fit_time_seconds'] = round(fit_time, 3)
    row['predict_time_seconds'] = round(predict_time, 3)
    row['cv_folds'] = (cv_stats or {}).get('folds')
    row['cv_pr_auc_mean'] = (cv_stats or {}).get('pr_auc_mean', np.nan)
    row['cv_pr_auc_std'] = (cv_stats or {}).get('pr_auc_std', np.nan)
    row['cv_roc_auc_mean'] = (cv_stats or {}).get('roc_auc_mean', np.nan)
    row['cv_roc_auc_std'] = (cv_stats or {}).get('roc_auc_std', np.nan)
    row['cv_time_seconds'] = round(cv_time, 3) if cv_time is not None else np.nan
    return row

def _cross_validate_pipeline(pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, random_state: int) -> tuple[dict | None, float]:
    if cv_folds < 2:
        return (None, 0.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    start = time.perf_counter()
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring={'pr_auc': 'average_precision', 'roc_auc': 'roc_auc'}, n_jobs=1, return_train_score=False)
    elapsed = time.perf_counter() - start
    stats = {'folds': cv_folds, 'pr_auc_mean': float(np.mean(scores['test_pr_auc'])), 'pr_auc_std': float(np.std(scores['test_pr_auc'])), 'roc_auc_mean': float(np.mean(scores['test_roc_auc'])), 'roc_auc_std': float(np.std(scores['test_roc_auc']))}
    return (stats, elapsed)

def tune_models(model_names: list[str], X_train: pd.DataFrame, y_train: pd.Series, numeric_features: list[str], categorical_features: list[str], random_state: int=RANDOM_STATE, n_iter: int=6, cv_folds: int=3) -> tuple[dict[str, object], list[dict]]:
    fitted: dict[str, object] = {}
    records: list[dict] = []
    for name in model_names:
        pipeline = build_model_pipeline(MODEL_FACTORIES[name](), numeric_features, categorical_features)
        grid = TUNING_GRIDS[name]
        n_candidates = min(n_iter, len(ParameterGrid(grid)))
        search = RandomizedSearchCV(estimator=pipeline, param_distributions=grid, n_iter=n_candidates, scoring='average_precision', cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state), random_state=random_state, n_jobs=1, refit=True)
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        tuned_name = f'{name} (tuned)'
        fitted[tuned_name] = search.best_estimator_
        records.append({'model': tuned_name, 'candidates_tested': n_candidates, 'cv_folds': cv_folds, 'scoring': 'average_precision', 'selection_basis': 'training CV PR-AUC', 'best_cv_pr_auc': float(search.best_score_), 'best_params': json.dumps(search.best_params_), 'search_time_seconds': round(elapsed, 3)})
        print(f'  Tuned {name}: AP={search.best_score_:.4f} in {elapsed:.1f}s')
    return (fitted, records)

def _evaluate_validation(pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[dict, np.ndarray, float]:
    start = time.perf_counter()
    probabilities = positive_class_probabilities(pipeline, X_valid)
    predict_time = time.perf_counter() - start
    metrics = compute_metrics(y_valid, probabilities)
    return (metrics, probabilities, predict_time)

def run_training_pipeline(dataset_path: Path | str | None=None, raw_dir: Path | str=RAW_DATA_DIR, reports_dir: Path | str=REPORTS_DIR, images_dir: Path | str=IMAGES_DIR, models_dir: Path | str=MODELS_DIR, model_names: list[str] | None=None, cv_folds: int=5, tune: bool=True, run_imbalance_experiments: bool=True, tuning_cv_folds: int=3, tuning_iterations: int=6, threshold_cv_folds: int=5, importance_repeats: int=10, random_state: int=RANDOM_STATE) -> dict:
    pipeline_start = time.perf_counter()
    reports_path = Path(reports_dir)
    images_path = Path(images_dir)
    models_path = Path(models_dir)
    for directory in (reports_path, images_path, models_path):
        directory.mkdir(parents=True, exist_ok=True)
    selected_models = model_names or list(MODEL_FACTORIES)
    unknown_models = sorted(set(selected_models) - set(MODEL_FACTORIES))
    if unknown_models:
        raise ValueError(f'Unknown models requested: {unknown_models}')
    if dataset_path is None:
        dataset_file = find_dataset_file(raw_dir)
    else:
        dataset_file = Path(dataset_path)
    print(f'Loading dataset: {dataset_file.name}')
    raw_df = load_raw_dataset(dataset_path=dataset_file)
    raw_quality = data_quality_report(raw_df)
    unexpected_before = count_unexpected_codes(raw_df)
    cleaned_df = clean_dataset(raw_df)
    cleaned_quality = data_quality_report(cleaned_df)
    model_df = engineer_features(cleaned_df)
    print(f'Rows: {len(raw_df):,} | Raw columns: {raw_df.shape[1]} | Modelled columns: {model_df.shape[1]}')
    if model_df.isna().any().any():
        missing = model_df.isna().sum()
        print('Warning: missing values found; the pipeline will impute them.')
        print(missing[missing > 0].to_string())
    train_df, valid_df, test_df = split_dataset(model_df, random_state=random_state)
    numeric_features, categorical_features = get_feature_lists(train_df)
    feature_columns = [column for column in train_df.columns if column != TARGET_COLUMN]
    feature_columns = [column for column in feature_columns if column != 'ID']
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]
    print(f'Split sizes -> train: {len(train_df):,} | validation: {len(valid_df):,} | test: {len(test_df):,}')
    comparison_rows: list[dict] = []
    fitted_pipelines: dict[str, object] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    curve_probabilities: dict[str, np.ndarray] = {}
    dummy_pipeline = build_model_pipeline(DummyClassifier(strategy='prior', random_state=random_state), numeric_features, categorical_features)
    start = time.perf_counter()
    dummy_pipeline.fit(X_train, y_train)
    dummy_fit = time.perf_counter() - start
    dummy_metrics, _dummy_probabilities, dummy_predict = _evaluate_validation(dummy_pipeline, X_valid, y_valid)
    comparison_rows.append(_validation_row('Dummy (majority class)', 'baseline', '', dummy_metrics, dummy_fit, dummy_predict))
    print('Training base models...')
    for name in selected_models:
        pipeline = build_model_pipeline(MODEL_FACTORIES[name](), numeric_features, categorical_features)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        metrics, probabilities, predict_time = _evaluate_validation(pipeline, X_valid, y_valid)
        cv_stats, cv_time = _cross_validate_pipeline(pipeline, X_train, y_train, cv_folds, random_state)
        comparison_rows.append(_validation_row(name, 'base', '', metrics, fit_time, predict_time, cv_stats, cv_time))
        fitted_pipelines[name] = pipeline
        validation_probabilities[name] = probabilities
        curve_probabilities[name] = probabilities
        print(f'  {name}: PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f}')
    imbalance_records: list[dict] = []
    if run_imbalance_experiments:
        print('Running imbalance-handling experiments...')
        balanced_specs = [('Logistic Regression (class weights)', lambda: LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state), 'class_weight=balanced'), ('Random Forest (class weights)', lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=random_state), 'class_weight=balanced_subsample')]
        for name, factory, handling in balanced_specs:
            pipeline = build_model_pipeline(factory(), numeric_features, categorical_features)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_time = time.perf_counter() - start
            metrics, probabilities, predict_time = _evaluate_validation(pipeline, X_valid, y_valid)
            comparison_rows.append(_validation_row(name, 'imbalance', handling, metrics, fit_time, predict_time))
            fitted_pipelines[name] = pipeline
            validation_probabilities[name] = probabilities
            imbalance_records.append({'model': name, 'handling': handling, 'validation_pr_auc': metrics['pr_auc']})
            print(f'  {name}: PR-AUC={metrics['pr_auc']:.4f}')
        if IMBLEARN_AVAILABLE:
            smote_pipeline = ImbalancedPipeline(steps=[('preprocessor', build_preprocessor(numeric_features, categorical_features)), ('sampler', SMOTE(random_state=random_state)), ('model', LogisticRegression(max_iter=2000, random_state=random_state))])
            start = time.perf_counter()
            smote_pipeline.fit(X_train, y_train)
            fit_time = time.perf_counter() - start
            metrics, probabilities, predict_time = _evaluate_validation(smote_pipeline, X_valid, y_valid)
            comparison_rows.append(_validation_row('Logistic Regression + SMOTE', 'imbalance', 'SMOTE on training folds only', metrics, fit_time, predict_time))
            fitted_pipelines['Logistic Regression + SMOTE'] = smote_pipeline
            validation_probabilities['Logistic Regression + SMOTE'] = probabilities
            imbalance_records.append({'model': 'Logistic Regression + SMOTE', 'handling': 'SMOTE on training folds only', 'validation_pr_auc': metrics['pr_auc']})
            print(f'  Logistic Regression + SMOTE: PR-AUC={metrics['pr_auc']:.4f}')
        else:
            print('  imbalanced-learn not installed; SMOTE experiment skipped.')
    tuning_records: list[dict] = []
    if tune and selected_models:
        base_frame = pd.DataFrame([row for row in comparison_rows if row['role'] == 'base'])
        tunable = [name for name in base_frame['model'].tolist() if name in TUNING_GRIDS]
        tunable_frame = base_frame[base_frame['model'].isin(tunable)]
        candidate_names = tunable_frame.sort_values('cv_pr_auc_mean', ascending=False, na_position='last')['model'].head(2).tolist()
        if candidate_names:
            print(f'Tuning top training-CV models: {candidate_names}')
            tuned_pipelines, tuning_records = tune_models(candidate_names, X_train, y_train, numeric_features, categorical_features, random_state=random_state, n_iter=tuning_iterations, cv_folds=tuning_cv_folds)
            for tuned_name, tuned_pipeline in tuned_pipelines.items():
                metrics, probabilities, predict_time = _evaluate_validation(tuned_pipeline, X_valid, y_valid)
                record = next((record for record in tuning_records if record['model'] == tuned_name))
                cv_stats = {'folds': tuning_cv_folds, 'pr_auc_mean': record['best_cv_pr_auc'], 'pr_auc_std': np.nan, 'roc_auc_mean': np.nan, 'roc_auc_std': np.nan}
                comparison_rows.append(_validation_row(tuned_name, 'tuned', 'randomized search', metrics, record['search_time_seconds'], predict_time, cv_stats))
                fitted_pipelines[tuned_name] = tuned_pipeline
                validation_probabilities[tuned_name] = probabilities
    comparison = pd.DataFrame(comparison_rows)
    comparison = apply_selection_policy(comparison)
    candidates = comparison[comparison['role'] != 'baseline']
    if candidates.empty:
        raise RuntimeError('No non-baseline candidate models were trained; a production model cannot be selected from the baseline alone.')
    selected_row = candidates.sort_values('selection_rank').iloc[0]
    selected_model = str(selected_row['model'])
    selected_pipeline = fitted_pipelines[selected_model]
    policy_payload = {**model_selection_policy(), 'eligible_models': comparison.loc[comparison['eligible'], 'model'].tolist(), 'policy_satisfied': bool(selected_row['eligible']), 'selection_data': 'validation split', 'threshold_calibration_data': 'out-of-fold predictions on the training split only', 'baseline_excluded_from_ranking': True, 'selected_model': selected_model, 'selected_validation_metrics': {key.replace('validation_', ''): float(selected_row[key]) for key in ['validation_precision', 'validation_recall', 'validation_f1', 'validation_roc_auc', 'validation_pr_auc']}, 'ranked_models': comparison[['selection_rank', 'model', 'role', 'eligible', 'validation_precision', 'validation_recall', 'validation_f1', 'validation_roc_auc', 'validation_pr_auc']].to_dict(orient='records')}
    cost_table = expected_cost_table(comparison, threshold=REFERENCE_THRESHOLD)
    print(f'Selected model: {selected_model} (recall={selected_row['validation_recall']:.4f}, precision={selected_row['validation_precision']:.4f}, PR-AUC={selected_row['validation_pr_auc']:.4f}, policy_satisfied={bool(selected_row['eligible'])})')
    selected_validation_probabilities = positive_class_probabilities(selected_pipeline, X_valid)
    diagnostic_validation_threshold = select_threshold(y_valid, selected_validation_probabilities)
    X_train_validation = pd.concat([X_train, X_valid], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_valid], ignore_index=True)
    print(f'Calibrating threshold from out-of-fold probabilities ({threshold_cv_folds}-fold, training split only)...')
    oof_probabilities, oof_folds = out_of_fold_probabilities(selected_pipeline, X_train, y_train, n_splits=threshold_cv_folds, random_state=random_state)
    threshold_table = threshold_analysis_table(y_train, oof_probabilities)
    selected_threshold = select_threshold(y_train, oof_probabilities)
    threshold_table['selected'] = threshold_table['threshold'].eq(selected_threshold)
    oof_metrics = compute_metrics(y_train, oof_probabilities, selected_threshold)
    threshold_selection_payload = {'method': 'out-of-fold predictions', 'data': 'training split only', 'n_samples': len(y_train), 'folds': int(threshold_cv_folds), 'cv': 'StratifiedKFold(shuffle=True)', 'metric': 'f1', 'selected_threshold': float(selected_threshold), 'model_selection_data': 'validation split', 'diagnostic_validation_f1_threshold': float(diagnostic_validation_threshold), 'diagnostic_note': 'The validation-only F1 threshold is reported for transparency and is not used to make any model or threshold decision.', 'oof_metrics_at_selected_threshold': oof_metrics, 'model_threshold_dependence': "The model is selected on validation, while the threshold is calibrated from training-only out-of-fold predictions.  The two decisions therefore use different data.  The calibrated threshold assumes the selected model's probability ranking transfers to the train+validation refit.", 'note': 'Each out-of-fold probability comes from a model fitted without the corresponding training record.  The test set was not accessed.  The threshold is fixed before test evaluation; the final pipeline is then refit on train+validation.'}
    print(f'Selected threshold on training out-of-fold data: {selected_threshold:.2f} (diagnostic validation-only F1 threshold: {diagnostic_validation_threshold:.2f})')
    print('Refitting selected pipeline on train + validation...')
    final_pipeline = clone(selected_pipeline)
    start = time.perf_counter()
    final_pipeline.fit(X_train_validation, y_train_validation)
    final_fit_time = time.perf_counter() - start
    print('Evaluating once on the untouched test set...')
    test_probabilities = positive_class_probabilities(final_pipeline, X_test)
    test_metrics = compute_metrics(y_test, test_probabilities, selected_threshold)
    test_metrics_default = compute_metrics(y_test, test_probabilities, 0.5)
    y_test_pred = (test_probabilities >= selected_threshold).astype(int)
    report_frame = classification_report_frame(y_test, y_test_pred)
    print('Computing permutation importance on the validation split...')
    permutation_importance_frame = permutation_importance_table(selected_pipeline, X_valid, y_valid, scoring='average_precision', n_repeats=importance_repeats, random_state=random_state)
    model_based_importance = get_feature_importances(final_pipeline, top_n=30)
    print('Running exploratory subgroup analysis on the test set...')
    subgroup_table = subgroup_performance(test_df, y_test, test_probabilities, selected_threshold)
    subgroup_report = fairness_report(subgroup_table, model_name=selected_model, threshold=selected_threshold, test_samples=len(y_test))
    model_comparison_path = reports_path / 'model_comparison.csv'
    comparison.to_csv(model_comparison_path, index=False)
    classification_report_path = reports_path / 'classification_report.csv'
    report_frame.to_csv(classification_report_path, index=False)
    threshold_path = reports_path / 'threshold_analysis.csv'
    threshold_table.to_csv(threshold_path, index=False)
    subgroup_path = reports_path / 'subgroup_performance.csv'
    subgroup_table.to_csv(subgroup_path, index=False)
    fairness_path = _write_json(reports_path / 'fairness_analysis.json', subgroup_report)
    selection_path = _write_json(reports_path / 'selection_policy.json', {**policy_payload, 'cost_sensitivity': cost_table.to_dict(orient='records')})
    importance_path = reports_path / 'feature_importance.csv'
    permutation_importance_frame.to_csv(importance_path, index=False)
    model_importance_path = None
    if model_based_importance is not None:
        model_importance_path = reports_path / 'feature_importance_model_based.csv'
        model_based_importance.to_csv(model_importance_path, index=False)
    final_test_payload = {'selected_model': selected_model, 'threshold': selected_threshold, 'threshold_selection': threshold_selection_payload, 'test_samples': len(y_test), 'test_defaults': int(y_test.sum()), 'at_selected_threshold': test_metrics, 'at_0_5': test_metrics_default, 'model_selection': {'rule': policy_payload['rule'], 'policy_satisfied': policy_payload['policy_satisfied'], 'reference_threshold': policy_payload['reference_threshold'], 'recall_floor': policy_payload['recall_floor'], 'precision_floor': policy_payload['precision_floor']}}
    final_test_path = _write_json(reports_path / 'final_test_metrics.json', final_test_payload)
    data_quality_payload = {'raw': raw_quality, 'cleaned': cleaned_quality, 'unexpected_codes_before_cleaning': unexpected_before, 'rows_removed_as_duplicates': int(len(raw_df) - len(cleaned_df)), 'cleaning_policy': {'duplicate_rows': 'removed', 'extreme_financial_values': 'retained (legitimate high-value customers)', 'education_codes_0_5_6': 'mapped to 4 (others/unknown)', 'marriage_code_0': 'mapped to 3 (others/unknown)', 'missing_values': 'imputed inside training pipeline only'}, 'feature_engineering': {'engineered_features': ENGINEERED_FEATURES, 'leakage_control': 'row-wise transformations only; target not used', 'missing_month_policy': 'six-month aggregates require all six months; partial histories stay missing and are imputed in the pipeline'}}
    data_quality_path = _write_json(reports_path / 'data_quality_summary.json', data_quality_payload)
    tuning_path = None
    if tuning_records:
        tuning_path = reports_path / 'hyperparameter_tuning.csv'
        pd.DataFrame(tuning_records).to_csv(tuning_path, index=False)
    base_plot = comparison[comparison['role'] != 'baseline'].copy()
    comparison_plot = base_plot.rename(columns={'validation_pr_auc': 'pr_auc', 'validation_roc_auc': 'roc_auc', 'validation_f1': 'f1', 'validation_recall': 'recall', 'validation_precision': 'precision'})
    model_comparison_plot = plot_model_comparison(comparison_plot, images_path / '07_model_comparison.png')
    roc_plot = pr_plot = None
    if curve_probabilities:
        roc_plot = plot_roc_curves(y_valid, curve_probabilities, images_path / '08_roc_curves.png')
        pr_plot = plot_precision_recall_curves(y_valid, curve_probabilities, images_path / '09_precision_recall_curves.png')
    confusion_plot = plot_confusion_matrix(y_test, y_test_pred, images_path / '10_confusion_matrix.png', title=f'Final test confusion matrix ({selected_model}, threshold={selected_threshold:.2f})')
    threshold_plot = plot_threshold_analysis(threshold_table, selected_threshold, images_path / '11_threshold_analysis.png')
    importance_plot = plot_feature_importance(permutation_importance_frame.head(15), images_path / '12_feature_importance.png', title=f'Permutation importance - {selected_model} (validation, average precision)', xlabel='Mean drop in average precision when shuffled')
    fairness_plot = plot_subgroup_performance(subgroup_table, images_path / '13_subgroup_performance.png', threshold=selected_threshold)
    artifact = {'pipeline': final_pipeline, 'threshold': selected_threshold, 'threshold_method': 'out-of-fold (training split only)', 'threshold_calibration': {'method': 'out-of-fold predictions', 'data': 'training split only', 'folds': int(threshold_cv_folds), 'n_samples': len(y_train)}, 'diagnostic_validation_f1_threshold': float(diagnostic_validation_threshold), 'selection_policy': policy_payload, 'model_name': selected_model, 'target_column': TARGET_COLUMN, 'feature_columns': feature_columns, 'numeric_features': numeric_features, 'categorical_features': categorical_features, 'trained_on': 'train + validation', 'random_state': random_state, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    model_path = models_path / 'final_model.joblib'
    joblib.dump(artifact, model_path)
    metadata_path = _write_json(models_path / 'model_metadata.json', {'model_name': selected_model, 'threshold': selected_threshold, 'threshold_method': 'out-of-fold predictions on the training split only', 'threshold_cv_folds': int(threshold_cv_folds), 'threshold_calibration_samples': len(y_train), 'diagnostic_validation_f1_threshold': float(diagnostic_validation_threshold), 'selection_policy': policy_payload, 'trained_on': 'train + validation', 'target_column': TARGET_COLUMN, 'feature_columns': feature_columns, 'numeric_features': numeric_features, 'categorical_features': categorical_features, 'engineered_features': ENGINEERED_FEATURES, 'random_state': random_state, 'created_at': artifact['created_at']})
    elapsed_total = time.perf_counter() - pipeline_start
    best_validation_row = comparison.loc[comparison['model'].eq(selected_model)].iloc[0]
    summary_payload = {'project': 'Credit Card Default Prediction', 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dataset': {'file': _relative_path(dataset_file), 'raw_rows': len(raw_df), 'raw_columns': int(raw_df.shape[1]), 'cleaned_rows': len(cleaned_df), 'modelled_columns': int(model_df.shape[1]), 'target': TARGET_COLUMN, 'default_rate': raw_quality.get('default_rate'), 'duplicate_rows_removed': int(len(raw_df) - len(cleaned_df)), 'unexpected_codes_before_cleaning': unexpected_before}, 'split': {'strategy': 'stratified 70/15/15', 'random_state': random_state, 'train_size': len(train_df), 'validation_size': len(valid_df), 'test_size': len(test_df), 'train_default_rate': float(y_train.mean()), 'validation_default_rate': float(y_valid.mean()), 'test_default_rate': float(y_test.mean())}, 'features': {'numeric': numeric_features, 'categorical': categorical_features, 'engineered': ENGINEERED_FEATURES, 'n_features_used': len(feature_columns)}, 'preprocessing': {'numeric': 'median imputation + StandardScaler, fitted on training data', 'categorical': "most-frequent imputation + OneHotEncoder(handle_unknown='ignore')", 'pipeline': 'ColumnTransformer inside sklearn Pipeline'}, 'models': {'base_models': selected_models, 'cross_validation': {'folds': cv_folds, 'scoring': ['average_precision (PR-AUC)', 'ROC-AUC'], 'data': 'training split only'}, 'tuning': {'method': 'RandomizedSearchCV', 'scoring': 'average_precision', 'cv_folds': tuning_cv_folds, 'candidate_selection': 'top 2 base models by training CV PR-AUC', 'records': tuning_records}}, 'imbalance_experiments': imbalance_records, 'selection': {'policy': policy_payload, 'selected_model': selected_model, 'validation_metrics': {key.replace('validation_', ''): best_validation_row[key] for key in best_validation_row.index if key.startswith('validation_')}, 'threshold': selected_threshold, 'threshold_selection': threshold_selection_payload, 'fit_on': 'train + validation', 'final_fit_time_seconds': round(final_fit_time, 3)}, 'feature_importance': {'primary_method': 'permutation importance on the validation split, scoring by average precision, using the model fitted on the training split', 'n_repeats': int(importance_repeats), 'model_based_method': 'impurity-based importances or absolute coefficients from the final refitted pipeline; reported for comparison only', 'caveat': 'Correlated and derived features can share or duplicate importance. Importance describes predictive association, not causation.'}, 'fairness_analysis': {'report': _relative_path(fairness_path), 'table': _relative_path(subgroup_path), 'n_groups': subgroup_report['n_groups'], 'n_small_groups': subgroup_report['n_small_groups'], 'caveat': 'Exploratory subgroup performance assessment, not a fairness audit.'}, 'final_test': final_test_payload, 'runtimes_seconds': {'total': round(elapsed_total, 3)}, 'artefacts': {'model': _relative_path(model_path), 'metadata': _relative_path(metadata_path), 'model_comparison': _relative_path(model_comparison_path), 'classification_report': _relative_path(classification_report_path), 'final_test_metrics': _relative_path(final_test_path), 'threshold_analysis': _relative_path(threshold_path), 'data_quality_summary': _relative_path(data_quality_path), 'hyperparameter_tuning': _relative_path(tuning_path) if tuning_path else None, 'selection_policy': _relative_path(selection_path), 'feature_importance': _relative_path(importance_path), 'feature_importance_model_based': _relative_path(model_importance_path) if model_importance_path else None, 'subgroup_performance': _relative_path(subgroup_path), 'fairness_analysis': _relative_path(fairness_path), 'figures': [_relative_path(path) for path in [model_comparison_plot, roc_plot, pr_plot, confusion_plot, threshold_plot, importance_plot, fairness_plot] if path is not None]}}
    summary_path = _write_json(reports_path / 'training_summary.json', summary_payload)
    print('Final test metrics (selected threshold):')
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        print(f'  {key}: {test_metrics[key]:.4f}')
    print(f'Pipeline completed in {elapsed_total:.1f}s')
    return {'comparison': comparison, 'selection_policy': policy_payload, 'cost_sensitivity': cost_table, 'threshold_table': threshold_table, 'threshold': selected_threshold, 'threshold_selection': threshold_selection_payload, 'diagnostic_validation_threshold': diagnostic_validation_threshold, 'oof_probabilities': oof_probabilities, 'oof_folds': oof_folds, 'selected_model': selected_model, 'validation_metrics': {key.replace('validation_', ''): selected_row[key] for key in selected_row.index if key.startswith('validation_')}, 'final_test_metrics': test_metrics, 'final_test_metrics_at_0_5': test_metrics_default, 'classification_report': report_frame, 'feature_importance': permutation_importance_frame, 'model_based_importance': model_based_importance, 'subgroup_table': subgroup_table, 'fairness_report': subgroup_report, 'fairness_plot': str(fairness_plot), 'tuning_records': tuning_records, 'imbalance_records': imbalance_records, 'data_quality': data_quality_payload, 'summary': summary_payload, 'artefacts': summary_payload['artefacts'], 'model_path': str(model_path), 'validation_probabilities': validation_probabilities, 'summary_path': str(summary_path), 'splits': {'train': train_df, 'validation': valid_df, 'test': test_df}, 'feature_lists': {'numeric': numeric_features, 'categorical': categorical_features, 'all': feature_columns}}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Train and evaluate credit card default classifiers.')
    parser.add_argument('--dataset', type=Path, default=None, help='Path to the raw dataset file.')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DATA_DIR)
    parser.add_argument('--reports-dir', type=Path, default=REPORTS_DIR)
    parser.add_argument('--images-dir', type=Path, default=IMAGES_DIR)
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--tuning-cv-folds', type=int, default=3)
    parser.add_argument('--tuning-iterations', type=int, default=6)
    parser.add_argument('--threshold-cv-folds', type=int, default=5)
    parser.add_argument('--importance-repeats', type=int, default=10)
    parser.add_argument('--no-tune', action='store_true')
    parser.add_argument('--no-imbalance', action='store_true')
    parser.add_argument('--models', nargs='*', choices=list(MODEL_FACTORIES), default=None, help='Subset of base models to train.')
    parser.add_argument('--quick', action='store_true', help='Fast smoke run: two models, 2-fold CV, no tuning or imbalance experiments.')
    args = parser.parse_args(argv)
    model_names = args.models
    cv_folds = args.cv_folds
    threshold_cv_folds = args.threshold_cv_folds
    importance_repeats = args.importance_repeats
    tune = not args.no_tune
    imbalance = not args.no_imbalance
    if args.quick:
        model_names = ['Logistic Regression', 'Decision Tree']
        cv_folds = 2
        threshold_cv_folds = 3
        importance_repeats = 3
        tune = False
        imbalance = False
    run_training_pipeline(dataset_path=args.dataset, raw_dir=args.raw_dir, reports_dir=args.reports_dir, images_dir=args.images_dir, models_dir=args.models_dir, model_names=model_names, cv_folds=cv_folds, tune=tune, run_imbalance_experiments=imbalance, tuning_cv_folds=args.tuning_cv_folds, tuning_iterations=args.tuning_iterations, threshold_cv_folds=threshold_cv_folds, importance_repeats=importance_repeats)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())