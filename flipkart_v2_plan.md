# Flipkart Gridlock v2 — Detailed Implementation Plan

---

## What the Results Are Telling You

Before touching any code, understand what the numbers mean:

**Baseline R² = 0.599 with geo_ts_mean in the feature set is too low.**

`geo_ts_mean` — the mean demand for each (geohash, timestamp) pair — should be an almost-perfect predictor on its own. Days 1 through 47 all feed into it, and demand at any location at any time of day is largely stable day-to-day. As a raw single predictor, it should give R² ≥ 0.75 on your validation fold. If the full model with 30+ features is only hitting 0.599, one of two things is true:

1. **Day 48 is genuinely anomalous** — demand on day 48 (your validation fold) is structurally different from days 1–47. This is a real distribution shift and nothing you do with feature engineering will fully close it, but you can partially compensate.

2. **The geo_ts_mean fallback chain is broken** — when a (geohash, timestamp) pair is unseen in tr_df, it falls back to `geo_mean`, which is much weaker. If many val pairs are unseen (due to the way your train fold is constructed), you're running on a degraded signal for large portions of the val fold.

**The single most important thing you can do right now is run this one line and share the result:**

```python
from sklearn.metrics import r2_score
print(r2_score(val_fe['demand'], val_fe['geo_ts_mean']))
```

This number determines the entire strategy going forward. Everything below is written assuming it comes back low (< 0.75), which is the most likely explanation for the 0.599 baseline.

---

## Why geo4_ts_mean and geo5_ts_mean Hurt

This is the most counterintuitive result and it's important you understand it before the next iteration.

**The problem is not the features themselves — it's how they were added.**

They were added as *additional independent features* on top of `geo_ts_mean`. But that's wrong. `geo4_ts_mean` is a coarser, noisier version of `geo_ts_mean`. For the ~90% of rows where `geo_ts_mean` is defined, `geo4_ts_mean` gives CatBoost a worse version of a signal it already has. CatBoost then has to decide which to trust; it splits its attention, and performance drops.

**The correct use of `geo4_ts_mean` and `geo5_ts_mean` is as fallbacks inside `geo_ts_mean`, not as separate columns.** You build a single feature that says: "use the finest grain available."

```
geo_ts_mean_v2 = geo_ts_mean if seen, else geo5_ts_mean if seen, else geo4_ts_mean if seen, else ts_mean
```

This is one feature, not three. It's strictly better than the current `geo_ts_mean` on its own.

**Same logic applies to `time_slot_cyclical` hurting:** you already have `sin_hour`/`cos_hour` and `ts_min`. Adding 96-slot cyclicals is correlated noise on top of 24-hour cyclicals, especially when the model already knows `ts_min` exactly.

---

## The v2 Plan — What to Build

### Step 0: Run the Diagnostic First (5 minutes)

Add a new cell immediately after `val_fe = build_features(val_df, tr_df)` and run it before anything else:

```python
# === DIAGNOSTIC: Pipeline Health Check ===
from sklearn.metrics import r2_score

print("=== PIPELINE DIAGNOSTIC ===")
print(f"val_fe rows: {len(val_fe)}, tr_fe rows: {len(tr_fe)}")

# How well does geo_ts_mean predict on its own?
raw_geo_ts_r2 = r2_score(val_fe['demand'], val_fe['geo_ts_mean'])
print(f"\ngeo_ts_mean as raw predictor R²: {raw_geo_ts_r2:.4f}")
# Expected: > 0.75 if pipeline is healthy
# If < 0.60: the fallback chain is broken, geo_ts_mean is filling with geo_mean too often

# What fraction of val rows have an exact geo_ts_mean match vs fell back?
ref_pairs = set(zip(tr_df['geohash'], tr_df['timestamp']))
val_pairs = list(zip(val_fe['geohash'], val_fe['timestamp']))
hit_rate = sum(1 for p in val_pairs if p in ref_pairs) / len(val_pairs)
print(f"geo×timestamp hit rate in val: {hit_rate:.3f}  ({hit_rate*100:.1f}% of val rows have exact match)")
# Expected: > 0.85 — if lower, you have a coverage problem

# Distribution of geo_ts_mean vs actual demand
print(f"\ngeo_ts_mean describe:\n{val_fe['geo_ts_mean'].describe().round(4)}")
print(f"\nval demand describe:\n{val_fe['demand'].describe().round(4)}")

# Check geo_mean as raw predictor (just location, no time)
raw_geo_r2 = r2_score(val_fe['demand'], val_fe['geo_mean'])
print(f"\ngeo_mean as raw predictor R²: {raw_geo_r2:.4f}")
# If geo_ts_mean >> geo_mean: time component is important
# If geo_ts_mean ≈ geo_mean: the timestamp signal isn't being captured

# Check ts_mean as raw predictor (just time, no location)
raw_ts_r2 = r2_score(val_fe['demand'], val_fe['ts_mean'])
print(f"ts_mean as raw predictor R²:  {raw_ts_r2:.4f}")

# Check if day 48 is anomalous vs historical
day48_demand = train[train['day'] == 48]['demand'].mean()
other_demand  = train[train['day'] < 48]['demand'].mean()
print(f"\nMean demand day 48:     {day48_demand:.4f}")
print(f"Mean demand days 1-47:  {other_demand:.4f}")
print(f"Ratio (48 / 1-47):      {day48_demand/other_demand:.4f}")
# If ratio is far from 1.0, day 48 is structurally different
```

**Read the output before doing anything else.** The numbers will tell you exactly what to fix.

---

### Step 1: Fix the `build_features` Function

Replace the current `geo_ts_mean` block and add improved fallback chains. This is the highest-impact change.

**Find this block in `build_features`:**

```python
# geo x timestamp, using the existing leakage-safe methodology.
df['geo_ts_mean'] = [geo_ts_dict.get((g, ts), np.nan)
                     for g, ts in zip(df['geohash'], df['timestamp'])]
df['geo_ts_mean'] = df['geo_ts_mean'].fillna(df['geo_mean'])

# geo4 x timestamp, fallback to geo4_mean.
df['geo4_ts_mean'] = [geo4_ts_dict.get((g4, ts), np.nan)
                      for g4, ts in zip(df['geo4'], df['timestamp'])]
df['geo4_ts_mean'] = df['geo4_ts_mean'].fillna(df['geo4_mean'])

# geo5 x timestamp, fallback to geo_ts_mean, then geo5_mean.
df['geo5_ts_mean'] = [geo5_ts_dict.get((g5, ts), np.nan)
                      for g5, ts in zip(df['geo5'], df['timestamp'])]
df['geo5_ts_mean'] = df['geo5_ts_mean'].fillna(df['geo_ts_mean']).fillna(df['geo5_mean'])
```

**Replace with:**

```python
# ── Hierarchical geo×timestamp target encoding (single feature, cascading fallback) ──
# geo_ts: exact geohash × timestamp match (finest grain, most signal)
geo_ts_raw = pd.Series(
    [geo_ts_dict.get((g, ts), np.nan) for g, ts in zip(df['geohash'], df['timestamp'])],
    index=df.index
)
# geo5_ts: geo5-prefix × timestamp (finer than geo4, coarser than exact)
geo5_ts_raw = pd.Series(
    [geo5_ts_dict.get((g5, ts), np.nan) for g5, ts in zip(df['geo5'], df['timestamp'])],
    index=df.index
)
# geo4_ts: geo4-prefix × timestamp (coarsest of the three, but most coverage)
geo4_ts_raw = pd.Series(
    [geo4_ts_dict.get((g4, ts), np.nan) for g4, ts in zip(df['geo4'], df['timestamp'])],
    index=df.index
)

# Single combined feature: use finest available grain, fall through to coarser
df['geo_ts_mean'] = (
    geo_ts_raw
    .fillna(geo5_ts_raw)       # if exact pair unseen, use geo5×ts
    .fillna(geo4_ts_raw)       # if geo5×ts also unseen, use geo4×ts
    .fillna(df['ts_mean']      # if all geohash-ts unseen, use time-only mean
            if 'ts_mean' in df.columns
            else global_mean)
    .fillna(global_mean)
)

# Keep the raw exact-match as a separate feature (confidence signal)
df['geo_ts_mean_exact'] = geo_ts_raw.fillna(-1)   # -1 = not seen in training
df['geo_ts_seen'] = (~geo_ts_raw.isna()).astype(int)  # binary: was this pair seen?

# Count feature: how many training samples back this estimate?
geo_ts_count_dict = ref_enc.groupby(['geohash', 'timestamp'])['demand'].count().to_dict()
geo4_ts_count_dict = ref_enc.groupby(['geo4', 'timestamp'])['demand'].count().to_dict()
df['geo_ts_count'] = [geo_ts_count_dict.get((g, ts), 0)
                      for g, ts in zip(df['geohash'], df['timestamp'])]
df['geo4_ts_count'] = [geo4_ts_count_dict.get((g4, ts), 0)
                       for g4, ts in zip(df['geo4'], df['timestamp'])]

# Delta: how much demand deviates at this time vs the geohash average
df['geo_ts_delta'] = df['geo_ts_mean'] - df['geo_mean']
```

**Why each piece matters:**

- `geo_ts_mean` now uses a 3-level fallback instead of falling straight to `geo_mean`. For any val pair where the exact geohash×timestamp is unseen, it uses the geo5-prefix mean instead of the much weaker geo_mean. This directly addresses the coverage gap.
- `geo_ts_seen` tells the model whether the current estimate came from an exact match or a fallback. This is extremely useful because CatBoost can learn: "if geo_ts_seen=1, trust geo_ts_mean; if 0, discount it."
- `geo_ts_count` tells the model how many samples back the estimate (confidence). A mean from 50 days is more reliable than a mean from 2 days.
- `geo_ts_mean_exact` keeps the raw exact value (with -1 for unseen) as a separate signal. The model can learn "this specific (geohash,timestamp) has been anomalously high/low historically."

---

### Step 2: Add Smoothed Target Encodings

Raw means are noisy when count is low. A geohash with 2 training samples has a very unreliable `geo_mean`. Smoothing pulls rare estimates toward the global mean.

**Add this block inside `build_features`, right after computing `geo_ts_mean`:**

```python
# ── Smoothed (m-estimate) target encodings ─────────────────────────────────
# Formula: smoothed = (sum + global_mean * m) / (count + m)
# m=20 means: a geohash needs 20+ samples before we trust its mean fully
M = 20  # smoothing factor; tune this if needed (try 10, 20, 50)

def smooth_mean(count_dict, sum_dict, global_mean, m=M):
    """Return a dict of smoothed means."""
    result = {}
    for key in count_dict:
        cnt = count_dict[key]
        s = sum_dict.get(key, 0)
        result[key] = (s + global_mean * m) / (cnt + m)
    return result

geo_ts_sum_dict   = ref_enc.groupby(['geohash', 'timestamp'])['demand'].sum().to_dict()
geo_hour_sum_dict = ref_enc.groupby(['geohash', 'hour'])['demand'].sum().to_dict()
geo_hour_count_dict = ref_enc.groupby(['geohash', 'hour'])['demand'].count().to_dict()
geo_sum_dict      = ref_enc.groupby('geohash')['demand'].sum().to_dict()
geo_cnt_dict      = ref_enc.groupby('geohash')['demand'].count().to_dict()

geo_ts_smooth_dict   = smooth_mean(geo_ts_count_dict, geo_ts_sum_dict, global_mean)
geo_hour_smooth_dict = smooth_mean(geo_hour_count_dict, geo_hour_sum_dict, global_mean)
geo_smooth_dict      = smooth_mean(geo_cnt_dict, geo_sum_dict, global_mean)

df['geo_ts_smooth']   = [geo_ts_smooth_dict.get((g, ts), global_mean)
                         for g, ts in zip(df['geohash'], df['timestamp'])]
df['geo_hour_smooth'] = [geo_hour_smooth_dict.get((g, h), global_mean)
                         for g, h in zip(df['geohash'], df['hour'])]
df['geo_smooth']      = df['geohash'].map(geo_smooth_dict).fillna(global_mean)
```

**Why this helps:** For a geohash×timestamp pair with only 2 samples in training, the raw mean is highly sensitive to outliers. With M=20, you need 20 samples for the estimate to be near-unbiased. This is especially important for the 2:15–13:45 timestamps that are rarer in the training distribution.

---

### Step 3: Add Day-48 Recency Features (No Leakage)

Day 48 is the last day before the test day (49). Its demand patterns are the most recent signal available and may be closer to day 49 than the historical average.

**Important: compute these only from the non-val portion of day 48 (i.e., tr_df day-48 rows), so there is zero leakage into the val fold.**

```python
# ── Day-48 recency features ─────────────────────────────────────────────────
# Only use rows from ref_df that belong to day 48. For the validation run,
# ref_df = tr_df which already excludes the val timestamps of day 48.
# For the final model, ref_df = full train, which includes all of day 48.
ref_day48 = ref_enc[ref_enc['day'] == 48] if 'day' in ref_enc.columns else pd.DataFrame()

if len(ref_day48) > 10:
    # Mean demand per geohash on day 48 (only hours outside the test window)
    geo_day48_mean = ref_day48.groupby('geohash')['demand'].mean().to_dict()
    geo_day48_std  = ref_day48.groupby('geohash')['demand'].std().fillna(0).to_dict()
    
    df['geo_day48_mean'] = df['geohash'].map(geo_day48_mean).fillna(df['geo_mean'])
    df['geo_day48_std']  = df['geohash'].map(geo_day48_std).fillna(df['geo_std'])
    
    # Ratio: how does day-48 demand compare to the long-run mean for this location?
    df['geo_day48_ratio'] = df['geo_day48_mean'] / (df['geo_mean'] + 1e-8)
else:
    df['geo_day48_mean']  = df['geo_mean']
    df['geo_day48_std']   = df['geo_std']
    df['geo_day48_ratio'] = 1.0
```

**Why this matters:** If demand is drifting up or down across days (trend), the most recent day's signal is more predictive than a 47-day average. The ratio feature specifically tells the model: "this location is currently running 15% above its historical average."

**Leakage check:** `ref_day48 = ref_enc[ref_enc['day'] == 48]`. When `ref_enc = tr_df`, tr_df contains day-48 rows only OUTSIDE the test timestamp window (i.e., the overnight/early-morning hours, 0:00–2:00). The val fold is the 2:15–13:45 window of day 48. No overlap. Zero leakage.

---

### Step 4: Add geo5 as a CatBoost Categorical

Currently your categorical features are: `geohash`, `geo4`, `road_type_filled`, `weather_filled`.

Add `geo5`. It sits between `geohash` (exact, ~N unique) and `geo4` (~N/32 unique) in the hierarchy, giving CatBoost another level of spatial aggregation to work with natively.

```python
# In BASE_CAT_FEATURES:
BASE_CAT_FEATURES = ['geohash', 'geo4', 'geo5', 'road_type_filled', 'weather_filled']

# In BASE_FEATURES, add 'geo5':
# Already present: 'geo4'
# Add:             'geo5'
```

This is a 2-minute change with potentially meaningful gain because CatBoost's target-stats encoding (its internal TE for categoricals) will compute its own geo5×feature statistics during training.

---

### Step 5: Updated Feature Lists

After all changes, your feature lists become:

```python
BASE_FEATURES = [
    # Time
    'ts_min', 'hour', 'minute_slot',
    'is_rush_am', 'is_rush_pm', 'is_night',
    'sin_hour', 'cos_hour',
    # Road / infrastructure
    'NumberofLanes', 'large_veh_bin', 'landmark_bin', 'lanes_x_road',
    'temp_filled', 'road_type_ord',
    # Missing indicators (already kept from v1)
    'temp_missing', 'weather_missing', 'rt_missing',
    # Frequency (already kept from v1)
    'geo_freq', 'geo4_freq', 'geo5_freq',
    # Core target encodings
    'geo_mean', 'geo4_mean', 'geo5_mean',
    'geo_ts_mean',          # now uses hierarchical fallback — DO NOT add geo4_ts_mean/geo5_ts_mean separately
    'geo_ts_smooth',        # smoothed version of the above
    'geo_ts_mean_exact',    # raw exact match, -1 if unseen
    'geo_ts_seen',          # binary: was this pair seen in training?
    'geo_ts_count',         # how many samples back the geo_ts estimate
    'geo4_ts_count',        # how many geo4×ts samples
    'geo_hour_mean',
    'geo_hour_smooth',      # smoothed version
    'geo_smooth',           # smoothed geo mean
    'rt_ts_mean', 'rt_hour_mean', 'ts_mean',
    'geo_std', 'geo_count',
    'geo_ts_delta',
    # Recency features
    'geo_day48_mean', 'geo_day48_std', 'geo_day48_ratio',
    # Categoricals (CatBoost native)
    'geohash', 'geo4', 'geo5', 'road_type_filled', 'weather_filled',
]

BASE_CAT_FEATURES = ['geohash', 'geo4', 'geo5', 'road_type_filled', 'weather_filled']

# Remove FEATURE_GROUPS for geo4_ts_mean, geo5_ts_mean, time_slot_cyclical
# These are now either baked into geo_ts_mean or dropped
# Keep FEATURE_GROUPS for anything you still want to test incrementally
FEATURE_GROUPS = [
    # Spatial features: test individually since they hurt in v1
    {'name': 'geohash_spatial', 'features': ['latitude', 'longitude', 'lat_sq', 'lon_sq', 'lat_lon'], 'cat_features': []},
    {'name': 'geo_clusters',    'features': ['geo_cluster10', 'geo_cluster20'], 'cat_features': ['geo_cluster10', 'geo_cluster20']},
]
```

**Note:** Do NOT put `geo4_ts_mean` or `geo5_ts_mean` back as separate features. They are now baked into `geo_ts_mean` via the fallback chain. If you add them separately again, they will hurt again.

---

### Step 6: LightGBM — Fix the Categorical Encoding

The LightGBM model is already in the notebook, which is great. But there's a subtlety: your `make_lgbm_matrices` function label-encodes categoricals using the training set's unique values. This is correct for `road_type_filled` and `weather_filled`, but for `geohash` it creates a very high-cardinality integer feature that LightGBM can't use effectively without special tuning.

**For LightGBM, drop the raw `geohash`, `geo4`, `geo5` from the categorical list and rely on the target-encoding features instead:**

```python
# In your LGB params / make_lgbm_matrices call:
LGB_CAT_FEATURES_OVERRIDE = ['road_type_filled', 'weather_filled']
# Do NOT pass geohash/geo4/geo5 as categoricals to LightGBM
# They are already represented by geo_ts_mean, geo_mean, geo_freq, etc.

# Updated LGB params:
LGB_PARAMS = dict(
    n_estimators=10000,
    learning_rate=0.05,
    num_leaves=255,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='regression',
    metric='rmse',
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

# In the LGB fit call, use:
X_lgb_tr, X_lgb_val, lgb_cat_features = make_lgbm_matrices(
    tr_fe, val_fe, active_features, LGB_CAT_FEATURES_OVERRIDE
)
```

---

### Step 7: CatBoost Hyperparameter Tuning

Based on your grid search results (`depth=6, l2=7` was best), the model prefers shallow-and-regularized. This is expected because the dominant signal is already encoded in `geo_ts_mean`; the tree just needs to combine it with a few other features.

**Keep `depth=6, l2_leaf_reg=7`. Also test:**

```python
# Add to your tuning grid:
additional_params = [
    {'depth': 6, 'l2_leaf_reg': 7, 'learning_rate': 0.03},   # slower LR, more iterations
    {'depth': 6, 'l2_leaf_reg': 7, 'min_data_in_leaf': 10},  # allow more splits
    {'depth': 7, 'l2_leaf_reg': 7},                           # one more level
]
```

With `learning_rate=0.03`, increase `early_stopping_rounds` to 300 and allow up to 8000 iterations.

---

## The Complete Updated `build_features` Function

Here is the full replacement function. Copy this verbatim:

```python
def build_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    global_mean = ref_df['demand'].mean()
    M = 20  # smoothing factor for m-estimate TEs

    # ── Missing indicators (before imputation) ────────────────────────────────
    df['temp_missing']    = df['Temperature'].isna().astype(int)
    df['weather_missing'] = df['Weather'].isna().astype(int)
    df['rt_missing']      = df['RoadType'].isna().astype(int)

    # ── Timestamp features ────────────────────────────────────────────────────
    df['ts_min']      = df['timestamp'].apply(ts_to_min)
    df['hour']        = df['ts_min'] // 60
    df['minute_slot'] = (df['ts_min'] % 60) // 15
    df['time_slot']   = df['ts_min'] // 15
    df['is_rush_am']  = ((df['hour'] >= 7)  & (df['hour'] <= 9)).astype(int)
    df['is_rush_pm']  = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night']    = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['sin_hour']    = np.sin(2 * np.pi * df['ts_min'] / 1440)
    df['cos_hour']    = np.cos(2 * np.pi * df['ts_min'] / 1440)

    # ── Geohash prefix features ───────────────────────────────────────────────
    df['geo4'] = df['geohash'].str[:4]
    df['geo5'] = df['geohash'].str[:5]

    # ── Spatial features ──────────────────────────────────────────────────────
    df = add_geohash_coordinates(df)
    df['lat_sq']  = df['latitude'] ** 2
    df['lon_sq']  = df['longitude'] ** 2
    df['lat_lon'] = df['latitude'] * df['longitude']
    df = add_geo_clusters(df, ref_df)

    # ── Road type imputation + ordinal ────────────────────────────────────────
    rt_order     = {'Residential': 0, 'Street': 1, 'Highway': 2}
    geo_rt_mode  = (
        ref_df.groupby('geohash')['RoadType']
              .agg(lambda x: x.dropna().mode()[0] if not x.dropna().empty else 'Residential')
              .to_dict()
    )
    df['road_type_filled'] = df['RoadType'].copy()
    rt_na = df['road_type_filled'].isna()
    df.loc[rt_na, 'road_type_filled'] = df.loc[rt_na, 'geohash'].map(geo_rt_mode)
    df['road_type_filled'] = df['road_type_filled'].fillna('Residential')
    df['road_type_ord']    = df['road_type_filled'].map(rt_order).fillna(0).astype(int)

    # ── Binary flags & interactions ───────────────────────────────────────────
    df['large_veh_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['landmark_bin']  = (df['Landmarks'] == 'Yes').astype(int)
    df['lanes_x_road']  = df['NumberofLanes'] * df['road_type_ord']

    # ── Temperature imputation ────────────────────────────────────────────────
    weather_temp_mean = ref_df.groupby('Weather')['Temperature'].mean().to_dict()
    global_temp       = ref_df['Temperature'].mean()
    df['temp_filled']    = df['Temperature'].copy()
    t_na = df['temp_filled'].isna()
    df.loc[t_na, 'temp_filled'] = df.loc[t_na, 'Weather'].map(weather_temp_mean)
    df['temp_filled']    = df['temp_filled'].fillna(global_temp)
    df['weather_filled'] = df['Weather'].fillna('Sunny')

    # ── Prepare ref_df with derived columns ──────────────────────────────────
    ref_enc = ref_df.copy()
    ref_enc['geo4'] = ref_enc['geohash'].str[:4]
    ref_enc['geo5'] = ref_enc['geohash'].str[:5]
    ref_enc['hour'] = ref_enc['timestamp'].apply(ts_to_min) // 60

    # ── Basic stats dicts ─────────────────────────────────────────────────────
    geo_mean_d   = ref_enc.groupby('geohash')['demand'].mean().to_dict()
    geo4_mean_d  = ref_enc.groupby('geo4')['demand'].mean().to_dict()
    geo5_mean_d  = ref_enc.groupby('geo5')['demand'].mean().to_dict()
    geo_std_d    = ref_enc.groupby('geohash')['demand'].std().fillna(0).to_dict()
    geo_count_d  = ref_enc.groupby('geohash')['demand'].count().to_dict()
    ts_mean_d    = ref_enc.groupby('timestamp')['demand'].mean().to_dict()

    df['geo_mean']   = df['geohash'].map(geo_mean_d).fillna(global_mean)
    df['geo4_mean']  = df['geo4'].map(geo4_mean_d).fillna(global_mean)
    df['geo5_mean']  = df['geo5'].map(geo5_mean_d).fillna(df['geo4_mean'])
    df['ts_mean']    = df['timestamp'].map(ts_mean_d).fillna(global_mean)
    df['geo_std']    = df['geohash'].map(geo_std_d).fillna(0)
    df['geo_count']  = df['geohash'].map(geo_count_d).fillna(0)

    # ── Frequency encoding ────────────────────────────────────────────────────
    geo_freq  = ref_enc['geohash'].value_counts(normalize=True).to_dict()
    geo4_freq = ref_enc['geo4'].value_counts(normalize=True).to_dict()
    geo5_freq = ref_enc['geo5'].value_counts(normalize=True).to_dict()
    df['geo_freq']  = df['geohash'].map(geo_freq).fillna(0)
    df['geo4_freq'] = df['geo4'].map(geo4_freq).fillna(0)
    df['geo5_freq'] = df['geo5'].map(geo5_freq).fillna(0)

    # ── Hierarchical geo×timestamp TE (single feature, 3-level fallback) ──────
    geo_ts_cnt_d   = ref_enc.groupby(['geohash', 'timestamp'])['demand'].count().to_dict()
    geo_ts_sum_d   = ref_enc.groupby(['geohash', 'timestamp'])['demand'].sum().to_dict()
    geo5_ts_cnt_d  = ref_enc.groupby(['geo5', 'timestamp'])['demand'].count().to_dict()
    geo5_ts_sum_d  = ref_enc.groupby(['geo5', 'timestamp'])['demand'].sum().to_dict()
    geo4_ts_cnt_d  = ref_enc.groupby(['geo4', 'timestamp'])['demand'].count().to_dict()
    geo4_ts_sum_d  = ref_enc.groupby(['geo4', 'timestamp'])['demand'].sum().to_dict()

    # Raw mean lookups
    geo_ts_raw_mean  = {k: v/geo_ts_cnt_d[k]  for k, v in geo_ts_sum_d.items()}
    geo5_ts_raw_mean = {k: v/geo5_ts_cnt_d[k] for k, v in geo5_ts_sum_d.items()}
    geo4_ts_raw_mean = {k: v/geo4_ts_cnt_d[k] for k, v in geo4_ts_sum_d.items()}

    # Smoothed (m-estimate) lookups
    geo_ts_smooth_d  = {k: (s + global_mean*M)/(geo_ts_cnt_d[k]+M)
                         for k, s in geo_ts_sum_d.items()}
    geo5_ts_smooth_d = {k: (s + global_mean*M)/(geo5_ts_cnt_d[k]+M)
                         for k, s in geo5_ts_sum_d.items()}
    geo4_ts_smooth_d = {k: (s + global_mean*M)/(geo4_ts_cnt_d[k]+M)
                         for k, s in geo4_ts_sum_d.items()}

    keys_geo_ts  = list(zip(df['geohash'], df['timestamp']))
    keys_geo5_ts = list(zip(df['geo5'],    df['timestamp']))
    keys_geo4_ts = list(zip(df['geo4'],    df['timestamp']))

    geo_ts_exact  = pd.Series([geo_ts_raw_mean.get(k,  np.nan) for k in keys_geo_ts],  index=df.index)
    geo5_ts_exact = pd.Series([geo5_ts_raw_mean.get(k, np.nan) for k in keys_geo5_ts], index=df.index)
    geo4_ts_exact = pd.Series([geo4_ts_raw_mean.get(k, np.nan) for k in keys_geo4_ts], index=df.index)

    # Combined feature: finest available grain
    df['geo_ts_mean'] = (
        geo_ts_exact
        .fillna(geo5_ts_exact)
        .fillna(geo4_ts_exact)
        .fillna(df['ts_mean'])
        .fillna(global_mean)
    )

    # Smoothed version (same hierarchy but with shrinkage)
    geo_ts_smooth_raw  = pd.Series([geo_ts_smooth_d.get(k,  np.nan) for k in keys_geo_ts],  index=df.index)
    geo5_ts_smooth_raw = pd.Series([geo5_ts_smooth_d.get(k, np.nan) for k in keys_geo5_ts], index=df.index)
    geo4_ts_smooth_raw = pd.Series([geo4_ts_smooth_d.get(k, np.nan) for k in keys_geo4_ts], index=df.index)
    df['geo_ts_smooth'] = (
        geo_ts_smooth_raw
        .fillna(geo5_ts_smooth_raw)
        .fillna(geo4_ts_smooth_raw)
        .fillna(df['ts_mean'])
        .fillna(global_mean)
    )

    # Confidence/coverage features
    df['geo_ts_seen']  = (~geo_ts_exact.isna()).astype(int)
    df['geo_ts_count'] = [geo_ts_cnt_d.get(k, 0) for k in keys_geo_ts]
    df['geo4_ts_count']= [geo4_ts_cnt_d.get(k, 0) for k in keys_geo4_ts]
    df['geo_ts_mean_exact'] = geo_ts_exact.fillna(-1)  # -1 = not seen in training

    # Delta features
    df['geo_ts_delta']       = df['geo_ts_mean']   - df['geo_mean']
    df['geo_ts_smooth_delta']= df['geo_ts_smooth'] - df['geo_mean']

    # ── geo×hour TE ───────────────────────────────────────────────────────────
    geo_hour_cnt_d = ref_enc.groupby(['geohash', 'hour'])['demand'].count().to_dict()
    geo_hour_sum_d = ref_enc.groupby(['geohash', 'hour'])['demand'].sum().to_dict()
    geo_hour_raw   = {k: v/geo_hour_cnt_d[k] for k, v in geo_hour_sum_d.items()}
    geo_hour_smo   = {k: (s+global_mean*M)/(geo_hour_cnt_d[k]+M)
                      for k, s in geo_hour_sum_d.items()}

    keys_geo_h = list(zip(df['geohash'], df['hour']))
    df['geo_hour_mean']   = pd.Series([geo_hour_raw.get(k, np.nan) for k in keys_geo_h],
                                       index=df.index).fillna(df['geo_mean'])
    df['geo_hour_smooth'] = pd.Series([geo_hour_smo.get(k,  np.nan) for k in keys_geo_h],
                                       index=df.index).fillna(df['geo_mean'])

    # ── RoadType×timestamp TE ─────────────────────────────────────────────────
    rt_ts_d   = ref_enc.groupby(['RoadType', 'timestamp'])['demand'].mean().to_dict()
    rt_hour_d = ref_enc.groupby(['RoadType', 'hour'])['demand'].mean().to_dict()

    df['rt_ts_mean']   = [rt_ts_d.get((rt, ts), np.nan)
                          for rt, ts in zip(df['road_type_filled'], df['timestamp'])]
    df['rt_ts_mean']   = df['rt_ts_mean'].fillna(global_mean)
    df['rt_hour_mean'] = [rt_hour_d.get((rt, h), np.nan)
                          for rt, h in zip(df['road_type_filled'], df['hour'])]
    df['rt_hour_mean'] = df['rt_hour_mean'].fillna(global_mean)

    # ── Smoothed geo mean ─────────────────────────────────────────────────────
    geo_sum_d = ref_enc.groupby('geohash')['demand'].sum().to_dict()
    df['geo_smooth'] = df['geohash'].apply(
        lambda g: (geo_sum_d.get(g, 0) + global_mean*M) / (geo_count_d.get(g, 0) + M)
    )

    # ── Day-48 recency features ───────────────────────────────────────────────
    ref_day48 = ref_enc[ref_enc['day'] == 48] if 'day' in ref_enc.columns else pd.DataFrame()
    if len(ref_day48) > 10:
        geo_d48_mean_d = ref_day48.groupby('geohash')['demand'].mean().to_dict()
        geo_d48_std_d  = ref_day48.groupby('geohash')['demand'].std().fillna(0).to_dict()
        df['geo_day48_mean']  = df['geohash'].map(geo_d48_mean_d).fillna(df['geo_mean'])
        df['geo_day48_std']   = df['geohash'].map(geo_d48_std_d).fillna(df['geo_std'])
        df['geo_day48_ratio'] = df['geo_day48_mean'] / (df['geo_mean'] + 1e-8)
    else:
        df['geo_day48_mean']  = df['geo_mean']
        df['geo_day48_std']   = df['geo_std']
        df['geo_day48_ratio'] = 1.0

    return df
```

---

## What NOT to Change

| Item | Why Leave It |
|---|---|
| Validation framework (temporal holdout) | Still correct; do not touch |
| `geo_ts_mean` as a single feature | Fixed in v2; adding geo4/geo5 separately will hurt again |
| Random KFold anywhere | Never add this |
| day%7, day%31, is_weekend | day is a sequential integer; these are noise |
| time_slot_cyclical as separate features | Already have ts_min exactly; these add noise |
| Lat/lon as separate features (FEATURE_GROUPS) | They consistently hurt; may retest after v2 is stable |

---

## Expected Outcomes

| Change | Expected Val R² Gain |
|---|---|
| Hierarchical fallback in geo_ts_mean | +0.02 to +0.10 |
| geo_ts_smooth (m-estimate) | +0.01 to +0.05 |
| geo_ts_seen + geo_ts_count | +0.01 to +0.04 |
| geo_ts_mean_exact | +0.01 to +0.03 |
| geo5 as CatBoost categorical | +0.005 to +0.02 |
| Day-48 recency ratio | +0.01 to +0.05 |
| LightGBM fix (no geohash categorical) | +0.01 to +0.03 |
| **Total stacked** | **+0.05 to +0.20** |

At a baseline of 0.664, these changes should bring you to **0.72–0.85 local R²**, which with your temporal validation should translate meaningfully to the leaderboard.

---

## Order of Operations

1. **Run the diagnostic cell** (5 min). Share the output — especially `geo_ts_mean raw R²` and hit rate.
2. **Replace `build_features`** with the v2 version above.
3. **Update `BASE_FEATURES` and `BASE_CAT_FEATURES`** as shown.
4. **Remove `geo4_ts_mean` and `geo5_ts_mean` from `FEATURE_GROUPS`** (they are now baked in).
5. **Fix LightGBM categorical handling** — pass only `road_type_filled`, `weather_filled` as categoricals.
6. **Run the full notebook** and check if baseline R² improves from 0.664.
7. **Submit to leaderboard** immediately after. You need the LB score to calibrate everything.
8. Only after getting an LB score: decide whether to continue feature engineering or focus on hyperparameter tuning.
