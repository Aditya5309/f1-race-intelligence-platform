# EDA Summary — F1 Race Winner Prediction

_Generated: 2026-06-08 | Source: `notebooks/01_eda_raw_data.ipynb`_

---

## Data Scope

| Property | Value |
|----------|-------|
| Source | `data/interim/results.parquet` |
| Rows | 27,279 |
| Year range | 1950–2026 (2026 races added) |
| Unique races | 1,171 |
| Drivers | 858 |
| Constructors | 210 |
| Winner entries | 1,157 (4.24% of all entries) |

---

## Key Findings

### 1. Grid Position vs Win Rate

Pole position (P1) wins **43.5%** of all races — by far the strongest single pre-race predictor.

| Grid | Win Rate |
|------|----------|
| P1 (pole) | 43.5% |
| P2 | 23.8% |
| P3 | 12.0% |
| P4 | 6.0% |
| P5 | 4.3% |
| P6–20 | < 4% each |

- Top-3 grid positions account for **79%** of all wins.
- Top-5 grid positions account for **89%** of all wins.
- Win rate signal is consistent across all eras — **grid position is the most reliable feature**.
- Figures: `reports/fig01_grid_vs_win_rate.png`

### 2. Constructor Dominance by Season

- **Ferrari** holds the all-time win record (249 wins), followed by McLaren (199), Mercedes (137), Red Bull (130).
- Clear **dominant eras**: Ferrari (1950s), Lotus (1963–73), McLaren/Williams (1984–97), Ferrari (1999–2004), Red Bull (2010–13, 2021–24), Mercedes (2014–20).
- HHI (championship concentration) by era:
  - 1950–1989: 0.397 (moderate)
  - 1990–2013: 0.455 (moderate-high)
  - **2014–2024: 0.594** (highest — Mercedes/Red Bull near-monopoly periods)
- Constructor form is highly predictive; **rolling win counts capture this without leakage**.
- Figures: `reports/fig02_constructor_dominance.png`

### 3. Driver Win Rate Distribution

- 161 drivers have ≥50 career starts; their win rate distribution is **heavily right-skewed**.
- Mean win rate: 4.17% | Median: 0.92% — the median driver almost never wins.
- Top champions: Fangio 47.1%, Jim Clark 34.7%, Verstappen 29.7%, Schumacher 29.5%, Hamilton 27.2%.
- **Driver identity cannot be used as a feature** — it memorises historical champions and breaks for rookies.
- Rolling form stats (wins/podiums in last N races) generalise to any driver including new entrants.
- Figures: `reports/fig03_driver_win_rates.png`

### 4. Result Status Over Time

Mean "Finished" rate by era:

| Era | Finished Rate |
|-----|--------------|
| 1950–1979 | 50.4% |
| 1980–2009 | 53.8% |
| **2010–2024** | **83.8%** |

- Finished rate has nearly doubled from early F1 to the modern era.
- Training on pre-1990 data would bias the model toward over-weighting retirements.
- The **2010 lower bound** for training data removes the reliability era mismatch.
- DNS (Did Not Start) entries are rare post-2000 but significant in 1970s–80s pre-qualifying era.
- Figures: `reports/fig04_result_status_over_time.png`

### 5. Structural Changes

| Era | Races | Pole Win% | Top-3 Grid Win% | Finish Rate% |
|-----|-------|-----------|-----------------|--------------|
| Pre-modern (1950–1982) | 373 | 35.5% | 23.8% | 53.6% |
| Turbo era (1983–1988) | 95 | 31.6% | 22.5% | 46.8% |
| V10/V8 (1989–2009) | 352 | 46.3% | 27.4% | 60.5% |
| **Modern (2010–2024)** | 305 | **51.1%** | **29.2%** | **84.0%** |
| Hybrid (2014–2024) | 228 | 52.2% | 29.0% | 84.6% |

- Field stabilised at exactly **20 drivers per race** post-2010.
- **Points system changed in 2010** (max 10 → 25 per race) — use standing rank, not raw points.
- Pole-to-win conversion is slightly higher in modern era (51%) than earlier (~35–46%).
- **2022 ground-effect regulations** represent a structural shift in aerodynamic philosophy — valid distribution shift for validation set.
- Figures: `reports/fig05_structural_changes.png`

---

## Recommended Train / Validation / Test Split

| Split | Years | Races | Entries | Winners |
|-------|-------|-------|---------|---------|
| **Train** | 2010–2021 | 237 | 5,077 | 237 |
| **Validation** | 2022–2023 | 44 | 880 | 44 |
| **Test** | 2024 | 24 | 479 | 24 |

**Rationale:**
- 2010 lower bound: stable 20-car field, modern points system, ~84% finish rate, no pre-hybrid physics
- 2022 validation: new ground-effect regulations = moderate intentional distribution shift
- 2024 test: fully held-out; never used for tuning
- **No random shuffle** — always split strictly by year (rolling features leak on shuffled splits)

**Class imbalance:** 4.67% positive rate → `scale_pos_weight ≈ 20` for XGBoost/LightGBM

---

## Candidate Predictive Features

| Rank | Feature | Expected Signal | Source |
|------|---------|----------------|--------|
| 1 | `grid_position` | **Very strong** | `results.parquet` |
| 2 | `qualifying_position` | **Very strong** | `qualifying.csv` |
| 3 | `driver_wins_last_3` | Strong | Rolling on results |
| 4 | `driver_wins_last_5` | Strong | Rolling on results |
| 5 | `constructor_wins_last_5` | Strong | Rolling on results |
| 6 | `driver_championship_position` | Strong | `driver_standings.csv` (round N-1) |
| 7 | `constructor_championship_position` | Strong | `constructor_standings.csv` (round N-1) |
| 8 | `qualifying_time_delta_pct` | Moderate | `qualifying.csv` Q1/Q2/Q3 |
| 9 | `circuit_wins_driver` | Moderate | results + circuits join |
| 10 | `circuit_wins_constructor` | Moderate | results + circuits join |
| 11 | `driver_podiums_last_5` | Moderate | Rolling on results |
| 12 | `driver_dnf_rate_last_5` | Weak-moderate | Rolling on results |
| 13 | `grid_position_norm` | Moderate | grid / field_size |
| 14 | `is_home_circuit` | Weak | drivers + circuits nationality |

---

## Data Leakage Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Rolling features on full dataset | Computing `driver_wins_last_5` over all rows includes future races | Compute using only rows prior to target race (strict temporal window) |
| Championship standings | `driver_standings` contains standing after each race | Use round N-1 standings only |
| Raw driver/constructor ID | Encodes historical champions; breaks for new drivers | Use rolling form stats, not identity keys |
| Race-day columns | `milliseconds`, `laps`, `fastestLapTime`, `fastestLapSpeed` are post-race | Exclude entirely from feature matrix |
| Season fraction | `round / total_rounds` requires knowing season length upfront | Use `round` number as-is |

---

## Phase 3 Next Steps

1. Record Decision 008 in `context/decisions.md`: train 2010–2021 / val 2022–2023 / test 2024
2. Add `clean_qualifying()` to build `data/interim/qualifying.parquet`
3. Build `data/interim/standings.parquet` from `driver_standings.csv` and `constructor_standings.csv`, lagged one round
4. Implement feature engineering (completed in modular `src/features/` modules):
   - `add_rolling_driver_stats(df, windows=[3, 5, 10])`
   - `add_rolling_constructor_stats(df, windows=[3, 5])`
   - `add_qualifying_delta(df, qualifying_df)`
   - `add_circuit_history(df)`
   - `add_standings_features(df, standings_df)`
5. Write `src/features/pipeline.py` — sklearn `Pipeline` wrapping all transforms
6. Write `tests/test_features.py` — unit tests including temporal leakage checks
7. Save `data/processed/features.parquet`
