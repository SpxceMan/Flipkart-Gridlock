#!/usr/bin/env python3
"""
Complete production-ready solution for demand regression hackathon.
Optimized for max(0, 100 * R²) leaderboard score.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import subprocess

# ─── Auto-install missing packages ───────────────────────────────────────────
REQUIRED = [
    'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost',
    'optuna', 'scipy'
]
for pkg in REQUIRED:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

# ─── Geohash decode (pure python, no dependency) ────────────────────────────
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_DECODEMAP = {c: i for i, c in enumerate(_BASE32)}

def _geohash_decode(gh):
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_lon = True
    for c in gh:
        cd = _DECODEMAP.get(c, 0)
        for mask in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if cd & mask:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if cd & mask:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_lon = not is_lon
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")

target_col = 'demand'
y_train = train[target_col].values.copy()
test_index = test['Index'].values.copy()

# ── Temporal holdout split (mirrors john_soln.py) ──────────────────────────
# Test set = day 49, timestamps 2:15–13:45.
# Best validation proxy: day 48 rows with those same timestamps.
# Train fold: day 48 rows outside that window + all day 49 rows.
def _ts_to_min(t):
    h, m = str(t).split(':')
    return int(h) * 60 + int(m)

TEST_TS_SET = set(test['timestamp'].unique())

_train48 = train[train['day'] == 48]
_train49 = train[train['day'] == 49]

# Boolean masks over the full train DataFrame
val_mask_full = (train['day'] == 48) & (train['timestamp'].isin(TEST_TS_SET))
train_mask_full = ~val_mask_full  # everything else is training

print(f"Temporal holdout — Val rows: {val_mask_full.sum():,}  Train rows: {train_mask_full.sum():,}")
print(f"Val timestamps mirror test window: {sorted(list(TEST_TS_SET), key=_ts_to_min)[:3]} … {sorted(list(TEST_TS_SET), key=_ts_to_min)[-3:]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEEP EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EDA")
print("=" * 70)
print("\n--- Missing Values (train) ---")
print(train.isnull().sum())
print(f"\n--- Duplicates: {train.duplicated().sum()} ---")
print("\n--- Data Types ---")
print(train.dtypes)
print("\n--- Target Distribution ---")
print(train[target_col].describe())
print(f"Skewness: {train[target_col].skew():.4f}")
print(f"Kurtosis: {train[target_col].kurtosis():.4f}")

for c in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']:
    print(f"\n--- {c} value counts ---")
    print(train[c].value_counts(dropna=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Combine for consistent processing
train['_is_train'] = 1
test['_is_train'] = 0
if target_col not in test.columns:
    test[target_col] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# ── 3a. Timestamp parsing ──
def parse_timestamp(ts):
    parts = str(ts).split(':')
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m

hours = []
minutes = []
for ts in df['timestamp']:
    h, m = parse_timestamp(ts)
    hours.append(h)
    minutes.append(m)

df['hour'] = hours
df['minute'] = minutes
df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95 unique slot index

# Day-based features (day is integer, not a real date, so we derive what we can)
df['dayofweek'] = df['day'] % 7
df['dayofmonth'] = df['day'] % 31
df['weekofyear'] = df['day'] // 7
df['month'] = (df['day'] // 30) % 12 + 1
df['quarter'] = ((df['month'] - 1) // 3) + 1
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['is_month_start'] = (df['dayofmonth'] <= 1).astype(int)
df['is_month_end'] = (df['dayofmonth'] >= 29).astype(int)

# ── 3b. Cyclical encodings ──
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
df['sin_time_slot'] = np.sin(2 * np.pi * df['time_slot'] / 96)
df['cos_time_slot'] = np.cos(2 * np.pi * df['time_slot'] / 96)

# ── 3c. Geohash processing ──
print("Decoding geohash...")
geo_cache = {}
lats, lons = [], []
for gh in df['geohash']:
    if gh not in geo_cache:
        geo_cache[gh] = _geohash_decode(gh)
    lat, lon = geo_cache[gh]
    lats.append(lat)
    lons.append(lon)

df['latitude'] = lats
df['longitude'] = lons
df['lat_lon_interaction'] = df['latitude'] * df['longitude']
df['lat_squared'] = df['latitude'] ** 2
df['lon_squared'] = df['longitude'] ** 2
df['lat_plus_lon'] = df['latitude'] + df['longitude']
df['lat_minus_lon'] = df['latitude'] - df['longitude']

# Geohash precision features
df['geohash_prefix3'] = df['geohash'].str[:3]
df['geohash_prefix4'] = df['geohash'].str[:4]
df['geohash_prefix5'] = df['geohash'].str[:5]

# KMeans clusters
print("Creating geo clusters...")
geo_coords = df[['latitude', 'longitude']].values
for k in [5, 10, 20, 50]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    df[f'geo_cluster_{k}'] = km.fit_predict(geo_coords)

# ── 3d. Categorical encoding ──
# Binary encoding
df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)

# Label encode categoricals
cat_cols = ['RoadType', 'Weather', 'geohash', 'geohash_prefix3', 'geohash_prefix4', 'geohash_prefix5']
label_encoders = {}
for c in cat_cols:
    df[c] = df[c].fillna('MISSING')
    le = LabelEncoder()
    df[c + '_le'] = le.fit_transform(df[c].astype(str))
    label_encoders[c] = le

# Frequency encoding
for c in cat_cols:
    freq = df[c].value_counts(normalize=True)
    df[c + '_freq'] = df[c].map(freq).astype(float)

# ── 3e. Interaction features ──
df['RoadType_Weather'] = df['RoadType_le'] * 100 + df['Weather_le']
df['RoadType_Lanes'] = df['RoadType_le'] * 10 + df['NumberofLanes']
df['Weather_Temp'] = df['Weather_le'].astype(float) * df['Temperature'].fillna(df['Temperature'].median())
df['LargeVeh_Lanes'] = df['LargeVehicles_enc'] * df['NumberofLanes']
df['Landmarks_RoadType'] = df['Landmarks_enc'] * df['RoadType_le']
df['Hour_RoadType'] = df['hour'] * 10 + df['RoadType_le']
df['Hour_Weather'] = df['hour'] * 10 + df['Weather_le']
df['Lanes_Landmarks'] = df['NumberofLanes'] * df['Landmarks_enc']
df['Hour_Lanes'] = df['hour'] * df['NumberofLanes']
df['TimeSlot_RoadType'] = df['time_slot'] * 10 + df['RoadType_le']

# ── 3f. Missing value handling ──
# Temperature: median fill
temp_median = df['Temperature'].median()
df['Temperature_missing'] = df['Temperature'].isnull().astype(int)
df['Temperature'] = df['Temperature'].fillna(temp_median)

# Weather: mode fill (already label encoded, handle in le column)
weather_mode = df.loc[df['Weather'] != 'MISSING', 'Weather'].mode()[0]
df['Weather_missing'] = (df['Weather'] == 'MISSING').astype(int)
df.loc[df['Weather'] == 'MISSING', 'Weather'] = weather_mode

roadtype_mode = df.loc[df['RoadType'] != 'MISSING', 'RoadType'].mode()[0]
df['RoadType_missing'] = (df['RoadType'] == 'MISSING').astype(int)
df.loc[df['RoadType'] == 'MISSING', 'RoadType'] = roadtype_mode

# Re-encode after filling
for c in ['RoadType', 'Weather']:
    le = label_encoders[c]
    df[c + '_le'] = le.transform(df[c].astype(str))

# ── 3g. Aggregation features ──
print("Creating aggregation features...")
train_mask = df['_is_train'] == 1

agg_groups = ['geohash', 'RoadType', 'Weather', 'geohash_prefix4']
for grp in agg_groups:
    # Only compute aggregations from training data to avoid leakage
    train_agg = df.loc[train_mask].groupby(grp)['Temperature'].agg(['count', 'mean', 'median', 'std'])
    train_agg.columns = [f'{grp}_temp_{s}' for s in ['count', 'mean', 'median', 'std']]
    train_agg = train_agg.reset_index()
    
    # Merge
    for col in train_agg.columns:
        if col != grp and col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = df.merge(train_agg, on=grp, how='left')

# Geohash demand aggregations (only from train, for train-known geohashes)
for grp in ['geohash', 'geohash_prefix4', 'geohash_prefix3']:
    grp_demand = df.loc[train_mask].groupby(grp)[target_col].agg(['mean', 'median', 'std', 'min', 'max'])
    grp_demand.columns = [f'{grp}_demand_{s}' for s in ['mean', 'median', 'std', 'min', 'max']]
    grp_demand = grp_demand.reset_index()
    for col in grp_demand.columns:
        if col != grp and col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = df.merge(grp_demand, on=grp, how='left')

# Hour-level demand aggregations
for grp in ['hour', 'time_slot']:
    grp_demand = df.loc[train_mask].groupby(grp)[target_col].agg(['mean', 'median', 'std'])
    grp_demand.columns = [f'{grp}_demand_{s}' for s in ['mean', 'median', 'std']]
    grp_demand = grp_demand.reset_index()
    for col in grp_demand.columns:
        if col != grp and col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = df.merge(grp_demand, on=grp, how='left')

# Geohash × hour demand
gh_hour = df.loc[train_mask].groupby(['geohash', 'hour'])[target_col].agg(['mean', 'count'])
gh_hour.columns = ['geohash_hour_demand_mean', 'geohash_hour_demand_count']
gh_hour = gh_hour.reset_index()
for col in gh_hour.columns:
    if col not in ['geohash', 'hour'] and col in df.columns:
        df.drop(col, axis=1, inplace=True)
df = df.merge(gh_hour, on=['geohash', 'hour'], how='left')

# Geohash × RoadType demand
gh_road = df.loc[train_mask].groupby(['geohash', 'RoadType_le'])[target_col].agg(['mean'])
gh_road.columns = ['geohash_road_demand_mean']
gh_road = gh_road.reset_index()
for col in gh_road.columns:
    if col not in ['geohash', 'RoadType_le'] and col in df.columns:
        df.drop(col, axis=1, inplace=True)
df = df.merge(gh_road, on=['geohash', 'RoadType_le'], how='left')

# Cluster demand aggregations
for k in [5, 10, 20, 50]:
    cl_col = f'geo_cluster_{k}'
    grp_demand = df.loc[train_mask].groupby(cl_col)[target_col].agg(['mean', 'std'])
    grp_demand.columns = [f'{cl_col}_demand_mean', f'{cl_col}_demand_std']
    grp_demand = grp_demand.reset_index()
    for col in grp_demand.columns:
        if col != cl_col and col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = df.merge(grp_demand, on=cl_col, how='left')

# ── 3h. Additional engineered features ──
# Temperature bins
df['temp_bin'] = pd.cut(df['Temperature'], bins=10, labels=False)

# ── 3h-i. Day-level lag features (temporal-isolation aware) ──────────────────
# Core insight: (geohash, timestamp) is the dominant predictive unit.
# We need "what happened here, at this exact time, yesterday" as a direct signal.
#
# Temporal isolation:
#   - Val rows  (day 48, test-window timestamps) → lag from day 47
#   - Test rows (day 49)                         → lag from day 48 (full)
#   - Other train rows (day d)                   → lag from day d-1
#
# The old geohash_ts_day48 feature was broken: it only used non-test-window
# hours from day 48 (night/evening), giving wrong signal for morning test window.
print("Building day-level lag features...")

# -- Build per-day lookup dictionaries for (geohash, timestamp) → demand --
# We use the ORIGINAL train DataFrame (before concat with test) to avoid any
# accidental leakage from test rows.
_train_for_lags = df.loc[df['_is_train'] == 1].copy()
_lag_dicts = {}  # day → {(geohash, timestamp): demand_mean}
for d in _train_for_lags['day'].unique():
    day_data = _train_for_lags[_train_for_lags['day'] == d]
    _lag_dicts[int(d)] = day_data.groupby(['geohash', 'timestamp'])[target_col].mean().to_dict()

# Also build per-day (geohash, hour) → demand for coarser fallback
_lag_hour_dicts = {}
for d in _train_for_lags['day'].unique():
    day_data = _train_for_lags[_train_for_lags['day'] == d]
    _lag_hour_dicts[int(d)] = day_data.groupby(['geohash', 'hour'])[target_col].mean().to_dict()

# Per-day geohash-level aggregates (fallback when exact timestamp unseen)
_lag_geo_dicts = {}
for d in _train_for_lags['day'].unique():
    day_data = _train_for_lags[_train_for_lags['day'] == d]
    _lag_geo_dicts[int(d)] = day_data.groupby('geohash')[target_col].mean().to_dict()

# Global mean for ultimate fallback
_global_demand_mean = _train_for_lags[target_col].mean()

# -- Compute lag day for each row --
# Val rows: day 48 with test-window timestamps → lag day 47
# Test rows: day 49 → lag day 48
# Other train rows: day d → lag day d-1
is_val_row = (df['_is_train'] == 1) & (df['day'] == 48) & (df['timestamp'].isin(TEST_TS_SET))
is_test_row = (df['_is_train'] == 0)

lag_day = (df['day'] - 1).astype(int).values.copy()
# Val rows explicitly get day 47
lag_day[is_val_row.values] = 47
# Test rows explicitly get day 48
lag_day[is_test_row.values] = 48

# -- Vectorized lag lookup --
geohashes = df['geohash'].values
timestamps = df['timestamp'].values
hours = df['hour'].values

# demand_lag1: exact (geohash, timestamp) match from previous day
demand_lag1 = np.full(len(df), np.nan)
# demand_lag1_hour: (geohash, hour) match from previous day (coarser)
demand_lag1_hour = np.full(len(df), np.nan)
# demand_lag1_geo: geohash-level mean from previous day
demand_lag1_geo = np.full(len(df), np.nan)

for i in range(len(df)):
    ld = lag_day[i]
    gh = geohashes[i]
    ts = timestamps[i]
    hr = hours[i]
    
    # Exact (geohash, timestamp) lag
    if ld in _lag_dicts:
        demand_lag1[i] = _lag_dicts[ld].get((gh, ts), np.nan)
    
    # (geohash, hour) lag
    if ld in _lag_hour_dicts:
        demand_lag1_hour[i] = _lag_hour_dicts[ld].get((gh, hr), np.nan)
    
    # geohash-level lag
    if ld in _lag_geo_dicts:
        demand_lag1_geo[i] = _lag_geo_dicts[ld].get(gh, np.nan)

df['demand_lag1'] = demand_lag1
df['demand_lag1_hour'] = demand_lag1_hour
df['demand_lag1_geo'] = demand_lag1_geo

# -- Bayesian smoothed lag (handles sparse coverage) --
# For (geohash, timestamp) lags: smooth toward geohash mean with prior strength M
# smoothed = (sum + M * prior) / (count + M)
# We compute count and sum per (geohash, timestamp) per lag-day
SMOOTH_M = 10  # prior strength

# Build sum and count dicts per day
_lag_sum_dicts = {}
_lag_count_dicts = {}
for d in _train_for_lags['day'].unique():
    day_data = _train_for_lags[_train_for_lags['day'] == d]
    _lag_sum_dicts[int(d)] = day_data.groupby(['geohash', 'timestamp'])[target_col].sum().to_dict()
    _lag_count_dicts[int(d)] = day_data.groupby(['geohash', 'timestamp'])[target_col].count().to_dict()

lag_sum = np.zeros(len(df))
lag_count = np.zeros(len(df))
lag_prior = np.full(len(df), _global_demand_mean)

for i in range(len(df)):
    ld = lag_day[i]
    gh = geohashes[i]
    ts = timestamps[i]
    
    if ld in _lag_sum_dicts:
        lag_sum[i] = _lag_sum_dicts[ld].get((gh, ts), 0.0)
        lag_count[i] = _lag_count_dicts[ld].get((gh, ts), 0.0)
    
    # Use geohash-level lag as the prior (better than global mean)
    if not np.isnan(demand_lag1_geo[i]):
        lag_prior[i] = demand_lag1_geo[i]

df['demand_lag1_smooth'] = (lag_sum + SMOOTH_M * lag_prior) / (lag_count + SMOOTH_M)

# -- Coverage indicator: does this row have an exact lag? --
df['demand_lag1_available'] = (~np.isnan(demand_lag1)).astype(int)

# -- Cascading fallback for demand_lag1 --
# Fill NaN with: hour-level lag → geo-level lag → overall geohash mean
df['demand_lag1'] = df['demand_lag1'].fillna(df['demand_lag1_hour'])
df['demand_lag1'] = df['demand_lag1'].fillna(df['demand_lag1_geo'])
df['demand_lag1'] = df['demand_lag1'].fillna(df['geohash_demand_mean'])
df['demand_lag1'] = df['demand_lag1'].fillna(_global_demand_mean)

df['demand_lag1_hour'] = df['demand_lag1_hour'].fillna(df['demand_lag1_geo'])
df['demand_lag1_hour'] = df['demand_lag1_hour'].fillna(_global_demand_mean)
df['demand_lag1_geo'] = df['demand_lag1_geo'].fillna(_global_demand_mean)

# -- Delta: how much does the lag deviate from the geohash average? --
df['demand_lag1_delta'] = df['demand_lag1'] - df['geohash_demand_mean'].fillna(_global_demand_mean)

lag1_coverage = df.loc[df['_is_train'] == 0, 'demand_lag1_available'].mean()
lag1_val_coverage = df.loc[is_val_row, 'demand_lag1_available'].mean()
print(f"  Lag1 coverage — test: {lag1_coverage:.1%}, val: {lag1_val_coverage:.1%}")
print(f"  New lag features: demand_lag1, demand_lag1_hour, demand_lag1_geo, "
      f"demand_lag1_smooth, demand_lag1_available, demand_lag1_delta")

# Ratio features
df['demand_ratio_gh_hour'] = df['geohash_hour_demand_mean'] / (df['geohash_demand_mean'] + 1e-8)
df['demand_ratio_ts_hour'] = df['time_slot_demand_mean'] / (df['hour_demand_mean'] + 1e-8)

print(f"Total features: {df.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TARGET ENCODING (TEMPORAL HOLDOUT)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TARGET ENCODING (TEMPORAL HOLDOUT)")
print("=" * 70)

te_cols = ['RoadType_le', 'Weather_le', 'geohash_le', 'geohash_prefix4_le',
           'geo_cluster_10', 'geo_cluster_20', 'RoadType_Weather',
           'Hour_RoadType', 'Hour_Weather', 'TimeSlot_RoadType']

train_df = df[df['_is_train'] == 1].reset_index(drop=True)
test_df = df[df['_is_train'] == 0].reset_index(drop=True)

# Recompute temporal holdout masks on the reset-index train_df
_te_val_mask = (train_df['day'] == 48) & (train_df['timestamp'].isin(TEST_TS_SET))
_te_tr_mask = ~_te_val_mask

for c in te_cols:
    col_name = f'{c}_te'
    global_mean = train_df[target_col].mean()
    
    # Compute target encoding from training fold only
    tr_data = train_df.loc[_te_tr_mask]
    grp = tr_data.groupby(c)[target_col].mean()
    
    # Apply to ALL train rows (both train-fold and val-fold get the same encoding)
    train_df[col_name] = train_df[c].map(grp).fillna(global_mean)
    
    # For test: use full training data
    full_grp = train_df.groupby(c)[target_col].mean()
    test_df[col_name] = test_df[c].map(full_grp).fillna(global_mean)

# Recombine
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREPARE FINAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

drop_cols = ['Index', 'geohash', 'timestamp', 'demand', '_is_train',
             'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
             'geohash_prefix3', 'geohash_prefix4', 'geohash_prefix5']

feature_cols = [c for c in df.columns if c not in drop_cols]

# Fill any remaining NaN
for c in feature_cols:
    if df[c].isnull().any():
        if df[c].dtype in ['float64', 'float32', 'int64', 'int32']:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode()[0])

train_df = df[df['_is_train'] == 1].reset_index(drop=True)
test_df = df[df['_is_train'] == 0].reset_index(drop=True)

# Re-extract since we added TE cols
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train = train_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)
y = train_df[target_col].values.astype(np.float64)

print(f"X_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"Features: {len(feature_cols)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VALIDATION FRAMEWORK (TEMPORAL HOLDOUT)
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-compute temporal train/val index arrays for the numpy feature matrices.
# train_df was rebuilt at line ~397 with reset_index, matching X_train row order.
_holdout_val_idx = np.where(
    (train_df['day'] == 48) & (train_df['timestamp'].isin(TEST_TS_SET))
)[0]
_holdout_tr_idx = np.where(
    ~((train_df['day'] == 48) & (train_df['timestamp'].isin(TEST_TS_SET)))
)[0]

def evaluate_cv(model_fn, X, y, return_oof=True, **_ignored):
    """Train on the temporal training fold, evaluate on the temporal val fold."""
    tr_idx, val_idx = _holdout_tr_idx, _holdout_val_idx
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = model_fn(X_tr, y_tr, X_val, y_val, 0)
    preds = model.predict(X_val)
    preds = np.clip(preds, 0, 1)
    
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f"  Temporal holdout: R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
    print(f"  Score (100*R²): {max(0, 100*r2):.4f}")
    
    # Return a list with one model for API compatibility
    return [model], preds, r2, [{'fold': 0, 'r2': r2, 'rmse': rmse, 'mae': mae}]

# ═══════════════════════════════════════════════════════════════════════════════
# 7. BASELINE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BASELINE MODELS")
print("=" * 70)

# ── 7a. LightGBM baseline ──
print("\n--- LightGBM Baseline ---")
def lgb_baseline(X_tr, y_tr, X_val, y_val, fold):
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'random_state': SEED,
        'verbosity': -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=5000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return model

lgb_models, lgb_oof, lgb_r2, _ = evaluate_cv(lgb_baseline, X_train, y)

# ── Feature importance from LightGBM ──
imp = np.zeros(len(feature_cols))
for m in lgb_models:
    imp += m.feature_importance(importance_type='gain')
imp /= len(lgb_models)
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': imp})
feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
print("\n--- Top 30 Features ---")
print(feat_imp.head(30).to_string())

# ── 7b. XGBoost baseline ──
print("\n--- XGBoost Baseline ---")
def xgb_baseline(X_tr, y_tr, X_val, y_val, fold):
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dval = xgb.DMatrix(X_val, y_val)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'random_state': SEED,
        'tree_method': 'hist',
        'verbosity': 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=5000,
                      evals=[(dval, 'val')],
                      early_stopping_rounds=100, verbose_eval=False)
    return model

class XGBWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))

def xgb_baseline_wrapped(X_tr, y_tr, X_val, y_val, fold):
    model = xgb_baseline(X_tr, y_tr, X_val, y_val, fold)
    return XGBWrapper(model)

xgb_models, xgb_oof, xgb_r2, _ = evaluate_cv(xgb_baseline_wrapped, X_train, y)

# ── 7c. CatBoost baseline ──
print("\n--- CatBoost Baseline ---")
def cb_baseline(X_tr, y_tr, X_val, y_val, fold):
    model = cb.CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_seed=SEED,
        eval_metric='RMSE',
        early_stopping_rounds=100,
        verbose=0,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model

cb_models, cb_oof, cb_r2, _ = evaluate_cv(cb_baseline, X_train, y)

# ── 7d. ExtraTrees baseline ──
print("\n--- ExtraTrees Baseline ---")
def et_baseline(X_tr, y_tr, X_val, y_val, fold):
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.7,
        n_jobs=2,
        random_state=SEED
    )
    model.fit(X_tr, y_tr)
    return model

et_models, et_oof, et_r2, _ = evaluate_cv(et_baseline, X_train, y)

# ── 7e. RandomForest baseline ──
print("\n--- RandomForest Baseline ---")
def rf_baseline(X_tr, y_tr, X_val, y_val, fold):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.7,
        n_jobs=2,
        random_state=SEED
    )
    model.fit(X_tr, y_tr)
    return model

rf_models, rf_oof, rf_r2, _ = evaluate_cv(rf_baseline, X_train, y)

print("\n--- Baseline Summary ---")
print(f"LightGBM  R²: {lgb_r2:.6f}  (100*R²={max(0,100*lgb_r2):.4f})")
print(f"XGBoost   R²: {xgb_r2:.6f}  (100*R²={max(0,100*xgb_r2):.4f})")
print(f"CatBoost  R²: {cb_r2:.6f}  (100*R²={max(0,100*cb_r2):.4f})")
print(f"ExtraTrees R²: {et_r2:.6f}  (100*R²={max(0,100*et_r2):.4f})")
print(f"RandomForest R²: {rf_r2:.6f}  (100*R²={max(0,100*rf_r2):.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

# Keep features with importance > threshold
importance_threshold = feat_imp['importance'].quantile(0.05)
selected_features = feat_imp[feat_imp['importance'] > importance_threshold]['feature'].tolist()
selected_idx = [feature_cols.index(f) for f in selected_features]

print(f"Selected {len(selected_features)} / {len(feature_cols)} features")

X_train_sel = X_train[:, selected_idx]
X_test_sel = X_test[:, selected_idx]

# ═══════════════════════════════════════════════════════════════════════════════
# 9. OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OPTUNA HYPERPARAMETER TUNING")
print("=" * 70)

# ── 9a. LightGBM Optuna ──
print("\n--- LightGBM Optuna (100 trials) ---")

def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 512),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.4, 1.0),
        'n_jobs': -1,
        'random_state': SEED,
        'verbosity': -1,
    }
    
    tr_idx, val_idx = _holdout_tr_idx, _holdout_val_idx
    dtrain = lgb.Dataset(X_train_sel[tr_idx], y[tr_idx])
    dval = lgb.Dataset(X_train_sel[val_idx], y[val_idx], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    val_preds = np.clip(model.predict(X_train_sel[val_idx]), 0, 1)
    
    return r2_score(y[val_idx], val_preds)

lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print(f"Best LightGBM R²: {lgb_study.best_value:.6f}")
print(f"Best params: {lgb_study.best_params}")

# ── 9b. XGBoost Optuna ──
print("\n--- XGBoost Optuna (50 trials) ---")

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'tree_method': 'hist',
        'random_state': SEED,
        'verbosity': 0,
    }
    
    tr_idx, val_idx = _holdout_tr_idx, _holdout_val_idx
    dtrain = xgb.DMatrix(X_train_sel[tr_idx], y[tr_idx])
    dval = xgb.DMatrix(X_train_sel[val_idx], y[val_idx])
    model = xgb.train(params, dtrain, num_boost_round=3000,
                      evals=[(dval, 'val')],
                      early_stopping_rounds=50, verbose_eval=False)
    val_preds = np.clip(model.predict(dval), 0, 1)
    
    return r2_score(y[val_idx], val_preds)

xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
print(f"Best XGBoost R²: {xgb_study.best_value:.6f}")
print(f"Best params: {xgb_study.best_params}")

# ── 9c. CatBoost Optuna ──
print("\n--- CatBoost Optuna (50 trials) ---")

def cb_objective(trial):
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'MVS', 'Bernoulli'])
    params = {
        'iterations': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'random_seed': SEED,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 50,
        'verbose': 0,
        'bootstrap_type': bootstrap_type,
    }
    
    if bootstrap_type == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 5.0)
    elif bootstrap_type in ['Bernoulli', 'MVS']:
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    
    tr_idx, val_idx = _holdout_tr_idx, _holdout_val_idx
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train_sel[tr_idx], y[tr_idx],
              eval_set=(X_train_sel[val_idx], y[val_idx]), verbose=0)
    val_preds = np.clip(model.predict(X_train_sel[val_idx]), 0, 1)
    
    return r2_score(y[val_idx], val_preds)

cb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.optimize(cb_objective, n_trials=50, show_progress_bar=True)
print(f"Best CatBoost R²: {cb_study.best_value:.6f}")
print(f"Best params: {cb_study.best_params}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RETRAIN TUNED MODELS WITH CV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RETRAINING TUNED MODELS")
print("=" * 70)

# ── LightGBM tuned ──
print("\n--- LightGBM Tuned ---")
best_lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'random_state': SEED,
    'verbosity': -1,
    **lgb_study.best_params,
}

def lgb_tuned(X_tr, y_tr, X_val, y_val, fold):
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(best_lgb_params, dtrain, num_boost_round=10000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    return model

lgb_t_models, lgb_t_oof, lgb_t_r2, _ = evaluate_cv(lgb_tuned, X_train_sel, y)

# ── XGBoost tuned ──
print("\n--- XGBoost Tuned ---")
best_xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'random_state': SEED,
    'verbosity': 0,
    **xgb_study.best_params,
}

def xgb_tuned(X_tr, y_tr, X_val, y_val, fold):
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dval = xgb.DMatrix(X_val, y_val)
    model = xgb.train(best_xgb_params, dtrain, num_boost_round=10000,
                      evals=[(dval, 'val')],
                      early_stopping_rounds=200, verbose_eval=False)
    return XGBWrapper(model)

xgb_t_models, xgb_t_oof, xgb_t_r2, _ = evaluate_cv(xgb_tuned, X_train_sel, y)

# ── CatBoost tuned ──
print("\n--- CatBoost Tuned ---")
best_cb_params_raw = cb_study.best_params.copy()
best_bootstrap = best_cb_params_raw.pop('bootstrap_type', 'Bayesian')
best_cb_params = {
    'iterations': 10000,
    'random_seed': SEED,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'verbose': 0,
    'bootstrap_type': best_bootstrap,
    **best_cb_params_raw,
}
# Remove incompatible params based on bootstrap type
if best_bootstrap == 'Bayesian':
    best_cb_params.pop('subsample', None)
elif best_bootstrap in ['Bernoulli', 'MVS']:
    best_cb_params.pop('bagging_temperature', None)

def cb_tuned(X_tr, y_tr, X_val, y_val, fold):
    model = cb.CatBoostRegressor(**best_cb_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model

cb_t_models, cb_t_oof, cb_t_r2, _ = evaluate_cv(cb_tuned, X_train_sel, y)

# ── ExtraTrees (rerun with selected features) ──
print("\n--- ExtraTrees (selected features) ---")
et_t_models, et_t_oof, et_t_r2, _ = evaluate_cv(et_baseline, X_train_sel, y)

# ── RandomForest (rerun with selected features) ──
print("\n--- RandomForest (selected features) ---")
rf_t_models, rf_t_oof, rf_t_r2, _ = evaluate_cv(rf_baseline, X_train_sel, y)

print("\n--- Tuned Summary ---")
print(f"LightGBM   R²: {lgb_t_r2:.6f}  (100*R²={max(0,100*lgb_t_r2):.4f})")
print(f"XGBoost    R²: {xgb_t_r2:.6f}  (100*R²={max(0,100*xgb_t_r2):.4f})")
print(f"CatBoost   R²: {cb_t_r2:.6f}  (100*R²={max(0,100*cb_t_r2):.4f})")
print(f"ExtraTrees R²: {et_t_r2:.6f}  (100*R²={max(0,100*et_t_r2):.4f})")
print(f"RandomFor  R²: {rf_t_r2:.6f}  (100*R²={max(0,100*rf_t_r2):.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. ENSEMBLE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ENSEMBLE OPTIMIZATION")
print("=" * 70)

# Predictions are val-fold only; use matching val targets
y_val = y[_holdout_val_idx]
all_oof = np.column_stack([lgb_t_oof, xgb_t_oof, cb_t_oof, et_t_oof, rf_t_oof])
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'ExtraTrees', 'RandomForest']

def ensemble_r2_neg(weights):
    w = np.abs(weights)
    w = w / w.sum()
    oof_blend = all_oof @ w
    oof_blend = np.clip(oof_blend, 0, 1)
    return -r2_score(y_val, oof_blend)

# Multiple random restarts for better optimization
best_result = None
best_neg_r2 = 0

for restart in range(50):
    w0 = np.random.dirichlet(np.ones(len(model_names)))
    result = minimize(ensemble_r2_neg, w0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    if best_result is None or result.fun < best_neg_r2:
        best_result = result
        best_neg_r2 = result.fun

opt_weights = np.abs(best_result.x)
opt_weights = opt_weights / opt_weights.sum()

oof_ensemble = all_oof @ opt_weights
oof_ensemble = np.clip(oof_ensemble, 0, 1)
ensemble_r2 = r2_score(y_val, oof_ensemble)

print(f"\nOptimized Ensemble Weights:")
for name, w in zip(model_names, opt_weights):
    print(f"  {name}: {w:.6f}")
print(f"\nEnsemble R²: {ensemble_r2:.6f}  (100*R²={max(0,100*ensemble_r2):.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. FINAL PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FINAL PREDICTIONS")
print("=" * 70)

# Predict test with each model
lgb_test_preds = np.zeros(len(X_test_sel))
for m in lgb_t_models:
    lgb_test_preds += m.predict(X_test_sel) / len(lgb_t_models)

xgb_test_preds = np.zeros(len(X_test_sel))
for m in xgb_t_models:
    xgb_test_preds += m.predict(X_test_sel) / len(xgb_t_models)

cb_test_preds = np.zeros(len(X_test_sel))
for m in cb_t_models:
    cb_test_preds += m.predict(X_test_sel) / len(cb_t_models)

et_test_preds = np.zeros(len(X_test_sel))
for m in et_t_models:
    et_test_preds += m.predict(X_test_sel) / len(et_t_models)

rf_test_preds = np.zeros(len(X_test_sel))
for m in rf_t_models:
    rf_test_preds += m.predict(X_test_sel) / len(rf_t_models)

all_test_preds = np.column_stack([lgb_test_preds, xgb_test_preds, cb_test_preds,
                                   et_test_preds, rf_test_preds])

final_preds = all_test_preds @ opt_weights
final_preds = np.clip(final_preds, 0, 1)

# Also train on full data and predict
print("\n--- Full-data retraining ---")

# LightGBM full
dtrain_full = lgb.Dataset(X_train_sel, y)
lgb_full = lgb.train(best_lgb_params, dtrain_full, num_boost_round=5000)
lgb_full_preds = lgb_full.predict(X_test_sel)

# XGBoost full
dtrain_full_xgb = xgb.DMatrix(X_train_sel, y)
xgb_full = xgb.train(best_xgb_params, dtrain_full_xgb, num_boost_round=5000)
xgb_full_preds = xgb_full.predict(xgb.DMatrix(X_test_sel))

# CatBoost full
cb_full = cb.CatBoostRegressor(**{k: v for k, v in best_cb_params.items() if k != 'early_stopping_rounds'})
cb_full.set_params(iterations=5000)
cb_full.fit(X_train_sel, y, verbose=0)
cb_full_preds = cb_full.predict(X_test_sel)

# ExtraTrees full
et_full = ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_split=10,
                               min_samples_leaf=4, max_features=0.7, n_jobs=2, random_state=SEED)
et_full.fit(X_train_sel, y)
et_full_preds = et_full.predict(X_test_sel)

# RandomForest full
rf_full = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=10,
                                 min_samples_leaf=4, max_features=0.7, n_jobs=2, random_state=SEED)
rf_full.fit(X_train_sel, y)
rf_full_preds = rf_full.predict(X_test_sel)

all_full_preds = np.column_stack([lgb_full_preds, xgb_full_preds, cb_full_preds,
                                   et_full_preds, rf_full_preds])
final_full_preds = all_full_preds @ opt_weights
final_full_preds = np.clip(final_full_preds, 0, 1)

# Average CV-ensemble and full-data ensemble (hedge)
final_submission_preds = 0.5 * final_preds + 0.5 * final_full_preds
final_submission_preds = np.clip(final_submission_preds, 0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# 13. SAVE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    'Index': test_index,
    'demand': final_submission_preds
})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved: submission.csv")
print(f"Shape: {submission.shape}")
print(f"Demand stats:\n{submission['demand'].describe()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n{'Model':<15} {'R²':>10} {'100*R²':>10}")
print("-" * 37)
print(f"{'LightGBM':<15} {lgb_t_r2:>10.6f} {max(0,100*lgb_t_r2):>10.4f}")
print(f"{'XGBoost':<15} {xgb_t_r2:>10.6f} {max(0,100*xgb_t_r2):>10.4f}")
print(f"{'CatBoost':<15} {cb_t_r2:>10.6f} {max(0,100*cb_t_r2):>10.4f}")
print(f"{'ExtraTrees':<15} {et_t_r2:>10.6f} {max(0,100*et_t_r2):>10.4f}")
print(f"{'RandomForest':<15} {rf_t_r2:>10.6f} {max(0,100*rf_t_r2):>10.4f}")
print(f"{'ENSEMBLE':<15} {ensemble_r2:>10.6f} {max(0,100*ensemble_r2):>10.4f}")
print("-" * 37)

print(f"\nTop 30 Features:")
print(feat_imp.head(30)[['feature', 'importance']].to_string())

print(f"\nEnsemble Weights:")
for name, w in zip(model_names, opt_weights):
    print(f"  {name}: {w:.4f}")

print(f"\n{'='*70}")
print(f"FINAL ENSEMBLE CV SCORE: {max(0, 100*ensemble_r2):.4f}")
print(f"{'='*70}")
print("\nDone! submission.csv is ready for upload.")