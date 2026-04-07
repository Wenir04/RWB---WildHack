import warnings
warnings.filterwarnings('ignore')

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor

TRAIN_PATH = 'train_solo_track.parquet'
TEST_PATH = 'test_solo_track.parquet'
SUBMISSION_PATH = 'submission_catboost_direct_memory_safe.csv'
HORIZONS = list(range(1, 9))

# уменьшение потребления памяти за счёт downcast типов
def reduce_memory(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


# генерация календарных признаков
def add_time_features(df):
    df = df.copy()
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour.astype('int8')
    df['minute'] = ts.dt.minute.astype('int8')
    df['halfhour_idx'] = (ts.dt.hour * 2 + (ts.dt.minute // 30)).astype('int8')
    df['dow'] = ts.dt.dayofweek.astype('int8')
    df['dom'] = ts.dt.day.astype('int8')
    df['month'] = ts.dt.month.astype('int8')
    df['is_weekend'] = (df['dow'] >= 5).astype('int8')

    for col, period in [('halfhour_idx', 48), ('dow', 7), ('dom', 31)]:
        angle = 2 * np.pi * df[col].astype('float32') / np.float32(period)
        df[f'{col}_sin'] = np.sin(angle).astype('float32')
        df[f'{col}_cos'] = np.cos(angle).astype('float32')
    return df

# построение исторических признаков
def build_history_features(df):
    df = df.sort_values(['route_id', 'timestamp']).copy()
    df = add_time_features(df)
    df['route_id'] = df['route_id'].astype('category')

    g = df.groupby('route_id', sort=False, observed=True)

    for lag in [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]:
        df[f'target_lag_{lag}'] = g['target_1h'].shift(lag).astype('float32')

    shifted_target = g['target_1h'].shift(1)
    for window in [2, 4, 8, 16, 48]:
        df[f'target_roll_mean_{window}'] = (
            shifted_target.groupby(df['route_id'], observed=True)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )
        df[f'target_roll_std_{window}'] = (
            shifted_target.groupby(df['route_id'], observed=True)
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )

    status_cols = [f'status_{i}' for i in range(1, 7)]
    for col in status_cols:
        for lag in [1, 2, 4, 8, 16, 48]:
            df[f'{col}_lag_{lag}'] = g[col].shift(lag).astype('float32')

        shifted = g[col].shift(1)
        for window in [2, 4, 8, 16]:
            df[f'{col}_roll_mean_{window}'] = (
                shifted.groupby(df['route_id'], observed=True)
                .rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
                .astype('float32')
            )

    for lag in [1, 2, 4, 8]:
        local_sum = (
            df[f'status_1_lag_{lag}'] +
            df[f'status_2_lag_{lag}'] +
            df[f'status_3_lag_{lag}']
        ).astype('float32')
        prev_sum = (
            df[f'status_4_lag_{lag}'] +
            df[f'status_5_lag_{lag}'] +
            df[f'status_6_lag_{lag}']
        ).astype('float32')
        df[f'sum_local_status_lag_{lag}'] = local_sum
        df[f'sum_prev_status_lag_{lag}'] = prev_sum
        df[f'ratio_local_prev_lag_{lag}'] = (local_sum / (prev_sum + 1.0)).astype('float32')

    route_stats = (
        df.groupby('route_id', observed=True)['target_1h']
        .agg(['mean', 'median', 'std'])
        .rename(columns=lambda x: f'route_target_{x}')
        .reset_index()
    )
    for c in ['route_target_mean', 'route_target_median', 'route_target_std']:
        route_stats[c] = route_stats[c].astype('float32')

    df = df.merge(route_stats, on='route_id', how='left')
    return reduce_memory(df)

# формирование train выборки для конкретного горизонта
def make_horizon_train(df_feat, horizon, feature_cols):
    y_h = df_feat.groupby('route_id', sort=False, observed=True)['target_1h'].shift(-horizon)
    mask = y_h.notna().to_numpy()
    X = df_feat.loc[mask, feature_cols]
    y = y_h.loc[mask].astype('float32')
    return X, y

# формирование признаков для теста
def make_future_frame(train_feat, test_df, feature_cols):
    last_hist = (
        train_feat.sort_values(['route_id', 'timestamp'])
        .groupby('route_id', as_index=False, observed=True)
        .tail(1)
        .drop(columns=['timestamp'], errors='ignore')
    )

    future = test_df.merge(last_hist, on='route_id', how='left', suffixes=('', '_hist'))
    future = add_time_features(future)
    future['route_id'] = future['route_id'].astype(train_feat['route_id'].dtype)
    future = future.loc[:, ~future.columns.duplicated()].copy()
    future = reduce_memory(future)
    return future[['id', 'route_id', 'timestamp'] + feature_cols]


def main():
    train = pq.read_table(TRAIN_PATH).to_pandas()
    test = pq.read_table(TEST_PATH).to_pandas()

    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    train = train.sort_values(['route_id', 'timestamp']).reset_index(drop=True)
    test = test.sort_values(['route_id', 'timestamp']).reset_index(drop=True)

    for col in [f'status_{i}' for i in range(1, 7)] + ['target_1h']:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], downcast='float')

    train_feat = build_history_features(train)

    feature_cols = [c for c in train_feat.columns if c not in {'target_1h', 'timestamp'}]
    cat_features = ['route_id', 'hour', 'minute', 'halfhour_idx', 'dow', 'dom', 'month', 'is_weekend']

    models = {}
    for h in HORIZONS:
        print(f'Training horizon {h}...')
        X, y = make_horizon_train(train_feat, horizon=h, feature_cols=feature_cols)

        model = CatBoostRegressor(
            loss_function='MAE',
            eval_metric='MAE',
            iterations=900,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=7.0,
            min_data_in_leaf=40,
            bootstrap_type='Bayesian',
            random_seed=42,
            verbose=200,
        )
        model.fit(X, y, cat_features=cat_features, use_best_model=False)
        models[h] = model

        del X, y
        gc.collect()

    future = make_future_frame(train_feat, test, feature_cols)
    future['horizon'] = future.groupby('route_id', observed=True)['timestamp'].rank(method='first').astype('int8')

    preds = np.zeros(len(future), dtype=np.float32)
    for h in HORIZONS:
        mask = future['horizon'] == h
        preds[mask] = models[h].predict(future.loc[mask, feature_cols]).astype('float32')

    preds = np.clip(preds, 0, None)

    submission = future[['id']].copy()
    submission['target_1h'] = preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    print('Saved:', SUBMISSION_PATH)
    print(submission.head())


if __name__ == '__main__':
    main()
