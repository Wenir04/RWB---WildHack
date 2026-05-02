import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


TRAIN_PATH = Path("train_solo_track.parquet")
TEST_PATH = Path("test_solo_track.parquet")
SUBMISSION_PATH = Path("submission_catboost.csv")

RANDOM_STATE = 42
HORIZONS = list(range(1, 9))

STATUS_COLUMNS = [f"status_{i}" for i in range(1, 7)]

TARGET_LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
TARGET_WINDOWS = [2, 4, 8, 16, 48]

STATUS_LAGS = [1, 2, 4, 8, 16, 48]
STATUS_WINDOWS = [2, 4, 8, 16]

CAT_FEATURES = [
    "route_id",
    "hour",
    "minute",
    "halfhour_idx",
    "dow",
    "dom",
    "month",
    "is_weekend",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Уменьшает размер датафрейма за счёт более компактных числовых типов."""
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def read_parquet(path: Path) -> pd.DataFrame:
    logger.info("Reading %s", path)
    df = pq.read_table(path).to_pandas()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return reduce_memory(df)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные признаки из timestamp."""
    df = df.copy()

    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["halfhour_idx"] = (ts.dt.hour * 2 + ts.dt.minute // 30).astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["dom"] = ts.dt.day.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    cyclic_features = {
        "halfhour_idx": 48,
        "dow": 7,
        "dom": 31,
    }

    for col, period in cyclic_features.items():
        angle = 2 * np.pi * df[col].astype("float32") / np.float32(period)
        df[f"{col}_sin"] = np.sin(angle).astype("float32")
        df[f"{col}_cos"] = np.cos(angle).astype("float32")

    return df


def prepare_base_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    numeric_cols = STATUS_COLUMNS + ["target_1h"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def add_target_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт лаги и rolling-признаки по target_1h.

    Все rolling-признаки считаются через shift(1), чтобы не использовать
    текущее значение target_1h как признак.
    """
    group = df.groupby("route_id", sort=False, observed=True)

    for lag in TARGET_LAGS:
        df[f"target_lag_{lag}"] = (
            group["target_1h"]
            .shift(lag)
            .astype("float32")
        )

    shifted_target = group["target_1h"].shift(1)

    for window in TARGET_WINDOWS:
        rolled = shifted_target.groupby(df["route_id"], observed=True).rolling(window)

        df[f"target_roll_mean_{window}"] = (
            rolled.mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

        df[f"target_roll_std_{window}"] = (
            rolled.std()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    df["route_target_expanding_mean"] = (
        shifted_target
        .groupby(df["route_id"], observed=True)
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    df["route_target_expanding_std"] = (
        shifted_target
        .groupby(df["route_id"], observed=True)
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    return df


def add_status_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт исторические признаки по статусам маршрута."""
    group = df.groupby("route_id", sort=False, observed=True)

    for col in STATUS_COLUMNS:
        if col not in df.columns:
            continue

        for lag in STATUS_LAGS:
            df[f"{col}_lag_{lag}"] = (
                group[col]
                .shift(lag)
                .astype("float32")
            )

        shifted_status = group[col].shift(1)

        for window in STATUS_WINDOWS:
            df[f"{col}_roll_mean_{window}"] = (
                shifted_status
                .groupby(df["route_id"], observed=True)
                .rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
                .astype("float32")
            )

    return df


def add_status_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает локальные и предыдущие статусы на разных лагах."""
    required_cols = [f"status_{i}_lag_1" for i in range(1, 7)]
    if not all(col in df.columns for col in required_cols):
        return df

    for lag in [1, 2, 4, 8]:
        local_sum = (
            df[f"status_1_lag_{lag}"]
            + df[f"status_2_lag_{lag}"]
            + df[f"status_3_lag_{lag}"]
        ).astype("float32")

        previous_sum = (
            df[f"status_4_lag_{lag}"]
            + df[f"status_5_lag_{lag}"]
            + df[f"status_6_lag_{lag}"]
        ).astype("float32")

        df[f"sum_local_status_lag_{lag}"] = local_sum
        df[f"sum_prev_status_lag_{lag}"] = previous_sum
        df[f"ratio_local_prev_lag_{lag}"] = (
            local_sum / (previous_sum + 1.0)
        ).astype("float32")

    return df


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["route_id", "timestamp"]).copy()
    df = add_time_features(df)

    df["route_id"] = df["route_id"].astype("category")

    df = add_target_history_features(df)
    df = add_status_history_features(df)
    df = add_status_ratio_features(df)

    return reduce_memory(df)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    ignored_columns = {
        "target_1h",
        "timestamp",
    }

    return [col for col in df.columns if col not in ignored_columns]


def make_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    target = (
        df.groupby("route_id", sort=False, observed=True)["target_1h"]
        .shift(-horizon)
    )

    mask = target.notna()

    X = df.loc[mask, feature_cols]
    y = target.loc[mask].astype("float32")

    return X, y


def make_train_valid_split(
    X: pd.DataFrame,
    y: pd.Series,
    valid_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * (1 - valid_size))

    X_train = X.iloc[:split_idx]
    X_valid = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_valid = y.iloc[split_idx:]

    return X_train, X_valid, y_train, y_valid


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=900,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=7.0,
        min_data_in_leaf=40,
        bootstrap_type="Bayesian",
        random_seed=RANDOM_STATE,
        verbose=200,
    )

    fit_params = {
        "X": X_train,
        "y": y_train,
        "cat_features": CAT_FEATURES,
        "use_best_model": False,
    }

    if X_valid is not None and y_valid is not None:
        fit_params["eval_set"] = (X_valid, y_valid)

    model.fit(**fit_params)

    return model


def train_models(
    train_feat: pd.DataFrame,
    feature_cols: list[str],
) -> dict[int, CatBoostRegressor]:
    models = {}
    scores = []

    for horizon in HORIZONS:
        logger.info("Training horizon %s", horizon)

        X, y = make_horizon_dataset(train_feat, horizon, feature_cols)
        X_train, X_valid, y_train, y_valid = make_train_valid_split(X, y)

        model = train_model(X_train, y_train, X_valid, y_valid)

        valid_pred = model.predict(X_valid)
        valid_mae = mean_absolute_error(y_valid, valid_pred)

        logger.info("Horizon %s | validation MAE: %.5f", horizon, valid_mae)

        scores.append(
            {
                "horizon": horizon,
                "valid_mae": valid_mae,
                "train_rows": len(X_train),
                "valid_rows": len(X_valid),
            }
        )

        models[horizon] = model

        del X, y, X_train, X_valid, y_train, y_valid
        gc.collect()

    score_df = pd.DataFrame(scores)
    logger.info("\n%s", score_df)

    return models


def build_future_features(
    train_feat: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Готовит признаки для test.

    Для каждого route_id берём последнее известное историческое состояние.
    Затем соединяем его с будущими timestamp из test.
    """
    last_history = (
        train_feat
        .sort_values(["route_id", "timestamp"])
        .groupby("route_id", as_index=False, observed=True)
        .tail(1)
        .drop(columns=["timestamp"], errors="ignore")
    )

    future = test_df.merge(
        last_history,
        on="route_id",
        how="left",
        suffixes=("", "_history"),
    )

    future = add_time_features(future)

    future["route_id"] = future["route_id"].astype(train_feat["route_id"].dtype)

    future = future.loc[:, ~future.columns.duplicated()].copy()
    future = reduce_memory(future)

    return future[["id", "route_id", "timestamp"] + feature_cols]


def predict_test(
    models: dict[int, CatBoostRegressor],
    future: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    future = future.copy()

    future["horizon"] = (
        future
        .groupby("route_id", observed=True)["timestamp"]
        .rank(method="first")
        .astype("int8")
    )

    preds = np.zeros(len(future), dtype=np.float32)

    for horizon, model in models.items():
        mask = future["horizon"] == horizon

        if not mask.any():
            continue

        preds[mask] = (
            model
            .predict(future.loc[mask, feature_cols])
            .astype("float32")
        )

    return np.clip(preds, 0, None)


def save_submission(future: pd.DataFrame, preds: np.ndarray) -> None:
    submission = future[["id"]].copy()
    submission["target_1h"] = preds

    submission.to_csv(SUBMISSION_PATH, index=False)

    logger.info("Saved submission to %s", SUBMISSION_PATH)
    logger.info("\n%s", submission.head())


def main() -> None:
    train = read_parquet(TRAIN_PATH)
    test = read_parquet(TEST_PATH)

    train = prepare_base_train(train)
    test = test.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    logger.info("Building train features")
    train_feat = build_history_features(train)

    feature_cols = get_feature_columns(train_feat)

    logger.info("Number of features: %s", len(feature_cols))

    models = train_models(train_feat, feature_cols)

    logger.info("Building test features")
    future = build_future_features(train_feat, test, feature_cols)

    logger.info("Predicting test")
    preds = predict_test(models, future, feature_cols)

    save_submission(future, preds)


if __name__ == "__main__":
    main()
