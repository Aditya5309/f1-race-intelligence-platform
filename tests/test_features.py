"""
Tests for src/features/ (Phase 3b — feature engineering).

Every leakage risk documented in reports/master_dataset_design.md Section 6
has an explicit test here, as that document requires:

  6.1  Post-race outcome columns never appear as features
       -> test_feature_columns_disjoint_from_post_race_outcomes
       -> test_validate_features_rejects_post_race_column
  6.2  Rolling windows / standings use strict prior-race ordering
       -> test_rolling_wins_exclude_current_race
       -> test_rolling_window_spans_season_boundary
       -> test_first_career_race_rolling_is_nan
       -> test_standings_lagged_by_one_round
       -> test_round1_uses_prior_season_final_standings
       -> test_constructor_form_excludes_teammate_same_race
       -> test_constructor_circuit_wins_exclude_teammate_same_race
  6.3  is_home_circuit deferred (no direct join key)
       -> test_home_circuit_deferred
  6.4  Sprint enrichment deferred
       -> test_sprint_features_deferred
  6.5  grid == 0 pit-lane sentinel handled explicitly
       -> test_pit_lane_start_sentinel
  6.6  Driver form intentionally spans mid-season constructor changes
       -> test_driver_form_spans_constructor_change
  6.7  Per-race field-size normalization
       -> test_grid_position_norm_uses_per_race_field_size

Plus unit tests per module and an end-to-end smoke test on the real data.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.circuit_history import add_circuit_history_features
from src.features.constructor_form import add_constructor_form_features
from src.features.driver_form import add_driver_form_features
from src.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURES_DATASET_COLUMNS,
    MASTER_DATASET_PATH,
    build_features,
    validate_features,
)
from src.features.qualifying import add_qualifying_features, parse_qualifying_time
from src.features.standings import add_standings_features, build_prev_race_map
from src.features.weather import (
    WEATHER_CSV_PATH,
    WEATHER_FEATURES,
    add_weather_features,
)
from src.features.wet_form import SHRINKAGE_K, add_wet_form_features
from src.integration.build_master_dataset import POST_RACE_OUTCOME_COLUMNS

# ---------------------------------------------------------------------------
# Fixture builders — one synthetic master-dataset row per (raceId, driverId).
# ---------------------------------------------------------------------------

def _entry(**overrides) -> dict:
    base = {
        "raceId": 1, "driverId": 1, "constructorId": 1, "circuitId": 1,
        "year": 2020, "round": 1,
        "grid": 1, "qualifying_position": 1,
        "q1": "1:25.000", "q2": "1:24.000", "q3": "1:23.000",
        "position": 1, "positionText": "1", "positionOrder": 1,
        "points": 25.0, "laps": 58, "milliseconds": 5_000_000, "rank": 1,
        "fastestLap": 40, "fastestLapTime": "1:20.000", "fastestLapSpeed": 220.0,
        "statusId": 1, "result_status": "Finished", "finished": True,
        "winner": 0,
    }
    base.update(overrides)
    return base


def _driver_standing(race_id, driver_id, points, position, wins) -> dict:
    return {
        "driverStandingsId": race_id * 100 + driver_id,
        "raceId": race_id, "driverId": driver_id,
        "points": points, "position": position,
        "positionText": str(position), "wins": wins,
    }


def _constructor_standing(race_id, constructor_id, points, position, wins) -> dict:
    return {
        "constructorStandingsId": race_id * 100 + constructor_id,
        "raceId": race_id, "constructorId": constructor_id,
        "points": points, "position": position,
        "positionText": str(position), "wins": wins,
    }


def _season(driver_id=1, year=2020, n_races=4, winner_all=False, **overrides):
    """One driver's season: raceId == round == 1..n_races."""
    return [
        _entry(
            raceId=r, round=r, year=year, driverId=driver_id,
            winner=1 if winner_all else 0,
            positionOrder=1 if winner_all else 5,
            **overrides,
        )
        for r in range(1, n_races + 1)
    ]


def _empty_weather() -> pd.DataFrame:
    """No weather rows -- add_weather_features left-joins, so every row just
    gets NaN weather columns. Fine for tests that don't care about weather
    values, matching how a race missing from race_weather.csv behaves."""
    return pd.DataFrame(columns=["raceId", *WEATHER_FEATURES])


# ---------------------------------------------------------------------------
# 6.1 — post-race outcome columns are never features
# ---------------------------------------------------------------------------

def test_feature_columns_disjoint_from_post_race_outcomes():
    leaked = set(FEATURE_COLUMNS) & POST_RACE_OUTCOME_COLUMNS
    assert not leaked, f"Post-race column(s) in FEATURE_COLUMNS: {sorted(leaked)}"
    # Raw hazard columns are excluded too: grid carries the pit-lane sentinel
    # and q1/q2/q3 are unparsed strings — only engineered forms are features.
    for raw in ("grid", "q1", "q2", "q3"):
        assert raw not in FEATURE_COLUMNS
    # And no post-race column sneaks into the output schema at all.
    assert not set(FEATURES_DATASET_COLUMNS) & POST_RACE_OUTCOME_COLUMNS


def test_validate_features_rejects_post_race_column():
    master = pd.DataFrame(_season(winner_all=True))
    ds = pd.DataFrame([_driver_standing(1, 1, 25, 1, 1)])
    cs = pd.DataFrame([_constructor_standing(1, 1, 25, 1, 1)])
    features = build_features(master, ds, cs, _empty_weather())
    poisoned = features.assign(position=1)
    result = validate_features(poisoned, expected_row_count=len(master))
    assert not result.passed
    assert any("position" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 6.2 — rolling windows: strict prior-race ordering, exclusive of current race
# ---------------------------------------------------------------------------

def test_rolling_wins_exclude_current_race():
    # Driver wins every one of 4 races. If the window leaked the current
    # race, race k would show min(k, 3) wins; prior-only shows min(k-1, 3).
    df = add_driver_form_features(pd.DataFrame(_season(winner_all=True)))
    df = df.sort_values("raceId")
    assert pd.isna(df["driver_wins_last_3"].iloc[0])
    assert df["driver_wins_last_3"].iloc[1:].tolist() == [1.0, 2.0, 3.0]
    assert df["driver_wins_last_10"].iloc[3] == 3.0


def test_rolling_window_spans_season_boundary():
    # Two wins at the end of 2020, then round 1 of 2021: the window must look
    # back across the season boundary (ordered by (year, round), not raceId).
    rows = [
        _entry(raceId=10, year=2020, round=1, winner=1, positionOrder=1),
        _entry(raceId=11, year=2020, round=2, winner=1, positionOrder=1),
        _entry(raceId=12, year=2021, round=1, winner=0, positionOrder=5),
    ]
    df = add_driver_form_features(pd.DataFrame(rows)).sort_values("raceId")
    assert df["driver_wins_last_5"].iloc[2] == 2.0


def test_first_career_race_rolling_is_nan():
    # No prior history is NaN, not 0 — "unknown" and "0 wins lately" differ.
    df = add_driver_form_features(pd.DataFrame(_season(n_races=2)))
    first = df.sort_values("raceId").iloc[0]
    for col in ("driver_wins_last_5", "driver_avg_finish_last_5",
                "driver_dnf_rate_last_5", "driver_points_last_5"):
        assert pd.isna(first[col]), col
    assert first["driver_experience_races"] == 0


def test_driver_form_values():
    rows = [
        _entry(raceId=1, round=1, winner=1, positionOrder=1, points=25.0, finished=True),
        _entry(raceId=2, round=2, winner=0, positionOrder=3, points=15.0, finished=True),
        _entry(raceId=3, round=3, winner=0, positionOrder=18, points=0.0, finished=False),
        _entry(raceId=4, round=4, winner=0, positionOrder=5, points=10.0, finished=True),
    ]
    df = add_driver_form_features(pd.DataFrame(rows)).sort_values("raceId")
    last = df.iloc[3]
    assert last["driver_wins_last_5"] == 1.0
    assert last["driver_podiums_last_5"] == 2.0            # P1, P3
    assert last["driver_avg_finish_last_5"] == pytest.approx((1 + 3 + 18) / 3)
    assert last["driver_dnf_rate_last_5"] == pytest.approx(1 / 3)
    assert last["driver_points_last_5"] == pytest.approx(40.0)
    assert last["driver_experience_races"] == 3


# ---------------------------------------------------------------------------
# 6.2 — standings: lagged to round N-1, round-1 rule, no same-race standings
# ---------------------------------------------------------------------------

def test_standings_lagged_by_one_round():
    master = pd.DataFrame([
        _entry(raceId=1, round=1),
        _entry(raceId=2, round=2),
    ])
    # Standings AFTER race 1 vs AFTER race 2 differ — race 2's row must show
    # the post-race-1 value, never its own post-race-2 value.
    ds = pd.DataFrame([
        _driver_standing(1, 1, points=25, position=1, wins=1),
        _driver_standing(2, 1, points=43, position=2, wins=1),
    ])
    cs = pd.DataFrame([
        _constructor_standing(1, 1, points=25, position=1, wins=1),
        _constructor_standing(2, 1, points=43, position=3, wins=1),
    ])
    out = add_standings_features(master, ds, cs).sort_values("raceId")

    race1, race2 = out.iloc[0], out.iloc[1]
    assert pd.isna(race1["driver_standing_points_prev"])   # no prior race
    assert race2["driver_standing_points_prev"] == 25      # post-race-1, not 43
    assert race2["driver_standing_position_prev"] == 1
    assert race2["driver_standing_wins_prev"] == 1
    assert race2["constructor_standing_points_prev"] == 25
    assert race2["constructor_standing_position_prev"] == 1


def test_round1_uses_prior_season_final_standings():
    master = pd.DataFrame([
        _entry(raceId=1, year=2020, round=1),
        _entry(raceId=2, year=2020, round=2),
        _entry(raceId=3, year=2021, round=1),
    ])
    ds = pd.DataFrame([
        _driver_standing(1, 1, points=25, position=1, wins=1),
        _driver_standing(2, 1, points=40, position=2, wins=1),   # 2020 final
        _driver_standing(3, 1, points=18, position=4, wins=0),
    ])
    cs = pd.DataFrame([
        _constructor_standing(1, 1, points=25, position=1, wins=1),
        _constructor_standing(2, 1, points=40, position=2, wins=1),
        _constructor_standing(3, 1, points=18, position=4, wins=0),
    ])
    out = add_standings_features(master, ds, cs).sort_values("raceId")
    season2_round1 = out.iloc[2]
    # Round 1 of 2021 must carry the FINAL 2020 standing (round 2), not its
    # own post-race standing and not a null.
    assert season2_round1["driver_standing_points_prev"] == 40
    assert season2_round1["driver_standing_position_prev"] == 2
    assert season2_round1["constructor_standing_points_prev"] == 40


def test_first_appearance_has_null_standings():
    # Driver 2 debuts at race 2: there is no prior standings row for them,
    # so the lagged features are null — never a same-race lookup.
    master = pd.DataFrame([
        _entry(raceId=1, round=1, driverId=1),
        _entry(raceId=2, round=2, driverId=1),
        _entry(raceId=2, round=2, driverId=2, constructorId=2),
    ])
    ds = pd.DataFrame([
        _driver_standing(1, 1, points=25, position=1, wins=1),
        _driver_standing(2, 1, points=43, position=1, wins=2),
        _driver_standing(2, 2, points=10, position=5, wins=0),
    ])
    cs = pd.DataFrame([
        _constructor_standing(1, 1, points=25, position=1, wins=1),
        _constructor_standing(2, 1, points=43, position=1, wins=2),
        _constructor_standing(2, 2, points=10, position=5, wins=0),
    ])
    out = add_standings_features(master, ds, cs)
    debut = out[(out["raceId"] == 2) & (out["driverId"] == 2)].iloc[0]
    assert pd.isna(debut["driver_standing_position_prev"])
    assert pd.isna(debut["constructor_standing_position_prev"])


def test_prev_race_map_rejects_ambiguous_calendar():
    df = pd.DataFrame([
        _entry(raceId=1, year=2020, round=1),
        _entry(raceId=2, year=2020, round=1),   # same (year, round) twice
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        build_prev_race_map(df)


# ---------------------------------------------------------------------------
# 6.2 — constructor form: race-grain windows, teammate same-race exclusion
# ---------------------------------------------------------------------------

def _two_car_team(race_id, rnd, year=2020, winner_driver=None):
    """Two rows for constructor 1 in one race; driver 1 or 2 may win."""
    return [
        _entry(raceId=race_id, round=rnd, year=year, driverId=d,
               constructorId=1,
               winner=int(winner_driver == d),
               positionOrder=1 if winner_driver == d else 8)
        for d in (1, 2)
    ]


def test_constructor_form_excludes_teammate_same_race():
    # Driver 1 wins race 1. In race 1 itself, NEITHER car may see that win
    # (it is the outcome of the race being predicted); in race 2, BOTH must.
    rows = _two_car_team(1, 1, winner_driver=1) + _two_car_team(2, 2)
    out = add_constructor_form_features(pd.DataFrame(rows))
    race1 = out[out["raceId"] == 1]
    race2 = out[out["raceId"] == 2]
    assert race1["constructor_wins_last_5"].isna().all()
    assert (race2["constructor_wins_last_5"] == 1.0).all()


def test_constructor_window_counts_races_not_rows():
    # Constructor wins races 1-3 (two cars each = 6 rows). A row-based
    # "last 3" would see only ~1.5 races; a race-based one sees 3 wins.
    rows = []
    for r in (1, 2, 3):
        rows += _two_car_team(r, r, winner_driver=1)
    rows += _two_car_team(4, 4)
    out = add_constructor_form_features(pd.DataFrame(rows))
    race4 = out[out["raceId"] == 4]
    assert (race4["constructor_wins_last_3"] == 3.0).all()


def test_constructor_dnf_rate_prior_races_only():
    rows = _two_car_team(1, 1)
    for row in rows:
        row.update(finished=False, positionOrder=15, result_status="Retired")
    rows += _two_car_team(2, 2)
    out = add_constructor_form_features(pd.DataFrame(rows))
    race2 = out[out["raceId"] == 2]
    assert (race2["constructor_dnf_rate_last_5"] == 1.0).all()
    assert out[out["raceId"] == 1]["constructor_dnf_rate_last_5"].isna().all()


# ---------------------------------------------------------------------------
# 6.6 — driver form intentionally spans mid-season constructor changes
# ---------------------------------------------------------------------------

def test_driver_form_spans_constructor_change():
    rows = [
        _entry(raceId=1, round=1, constructorId=1, winner=1, positionOrder=1),
        _entry(raceId=2, round=2, constructorId=1, winner=1, positionOrder=1),
        _entry(raceId=3, round=3, constructorId=2, winner=0, positionOrder=6),
    ]
    out = add_driver_form_features(pd.DataFrame(rows)).sort_values("raceId")
    # Wins at the old team still count toward the DRIVER's form at the new team.
    assert out["driver_wins_last_5"].iloc[2] == 2.0

    # ...but the new constructor's form does NOT inherit them.
    out2 = add_constructor_form_features(pd.DataFrame(rows))
    race3 = out2[out2["raceId"] == 3].iloc[0]
    assert pd.isna(race3["constructor_wins_last_5"])   # constructor 2's first race


# ---------------------------------------------------------------------------
# Circuit history — prior visits to the same circuit only
# ---------------------------------------------------------------------------

def test_circuit_history_prior_visits_only():
    rows = [
        _entry(raceId=1, year=2019, round=1, circuitId=7, winner=1, positionOrder=1),
        _entry(raceId=2, year=2019, round=2, circuitId=9, winner=1, positionOrder=1),
        _entry(raceId=3, year=2020, round=1, circuitId=7, winner=0, positionOrder=4),
        _entry(raceId=4, year=2021, round=1, circuitId=7, winner=0, positionOrder=6),
    ]
    out = add_circuit_history_features(pd.DataFrame(rows)).set_index("raceId")

    # First-ever visit to circuit 7: no history at all.
    assert out.loc[1, "driver_circuit_starts"] == 0
    assert out.loc[1, "driver_circuit_wins"] == 0
    assert pd.isna(out.loc[1, "driver_circuit_avg_finish"])

    # Third visit: two prior visits at circuit 7 only — the circuit-9 win
    # must NOT bleed in, and the current race's own result must not count.
    assert out.loc[4, "driver_circuit_starts"] == 2
    assert out.loc[4, "driver_circuit_wins"] == 1
    assert out.loc[4, "driver_circuit_avg_finish"] == pytest.approx((1 + 4) / 2)


def test_constructor_circuit_wins_exclude_teammate_same_race():
    rows = _two_car_team(1, 1, winner_driver=1)                      # circuit 1
    rows += [
        _entry(raceId=2, round=2, driverId=d, constructorId=1, circuitId=1,
               winner=0, positionOrder=8)
        for d in (1, 2)
    ]
    out = add_circuit_history_features(pd.DataFrame(rows))
    race1 = out[out["raceId"] == 1]
    race2 = out[out["raceId"] == 2]
    # Same race: teammate's win at this circuit must not appear.
    assert (race1["constructor_circuit_wins"] == 0).all()
    # Next visit: it must.
    assert (race2["constructor_circuit_wins"] == 1).all()


# ---------------------------------------------------------------------------
# 6.5 / 6.7 — grid sentinel and per-race normalization
# ---------------------------------------------------------------------------

def test_pit_lane_start_sentinel():
    rows = [
        _entry(raceId=1, driverId=1, grid=1),
        _entry(raceId=1, driverId=2, grid=2),
        _entry(raceId=1, driverId=3, grid=0),    # pit-lane start
    ]
    out = add_qualifying_features(pd.DataFrame(rows)).set_index("driverId")
    assert bool(out.loc[3, "pit_lane_start"]) is True
    assert out.loc[3, "grid_adjusted"] == 4                 # field_size + 1
    assert out.loc[3, "grid_position_norm"] == pytest.approx(4 / 3)
    assert bool(out.loc[1, "pit_lane_start"]) is False
    assert out.loc[1, "grid_adjusted"] == 1


def test_grid_position_norm_uses_per_race_field_size():
    rows = [_entry(raceId=1, driverId=d, grid=d) for d in (1, 2)]
    rows += [_entry(raceId=2, driverId=d, grid=d) for d in (1, 2, 3, 4)]
    out = add_qualifying_features(pd.DataFrame(rows))
    small = out[(out["raceId"] == 1) & (out["driverId"] == 2)].iloc[0]
    large = out[(out["raceId"] == 2) & (out["driverId"] == 2)].iloc[0]
    assert small["grid_position_norm"] == pytest.approx(2 / 2)
    assert large["grid_position_norm"] == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# Qualifying times
# ---------------------------------------------------------------------------

def test_parse_qualifying_time():
    assert parse_qualifying_time("1:23.456") == pytest.approx(83.456)
    assert parse_qualifying_time("59.5") == pytest.approx(59.5)
    assert parse_qualifying_time("2:05.000") == pytest.approx(125.0)
    assert np.isnan(parse_qualifying_time(None))
    assert np.isnan(parse_qualifying_time("DNF"))
    assert np.isnan(parse_qualifying_time(""))


def test_gap_to_pole_uses_best_available_session():
    rows = [
        _entry(raceId=1, driverId=1, q1="1:26.000", q2="1:25.000", q3="1:24.000"),
        _entry(raceId=1, driverId=2, q1="1:25.200", q2=None, q3=None),  # out in Q1
    ]
    out = add_qualifying_features(pd.DataFrame(rows)).set_index("driverId")
    assert out.loc[1, "qualifying_gap_to_pole_pct"] == pytest.approx(0.0)
    expected = (85.2 - 84.0) / 84.0 * 100
    assert out.loc[2, "qualifying_gap_to_pole_pct"] == pytest.approx(expected)
    assert bool(out.loc[2, "reached_q2"]) is False
    assert bool(out.loc[1, "reached_q3"]) is True
    # Informative nulls are preserved, not imputed (design doc Section 5.3).
    assert pd.isna(out.loc[2, "q3_sec"])


# ---------------------------------------------------------------------------
# 6.3 / 6.4 — explicitly deferred features stay deferred
# ---------------------------------------------------------------------------

def test_home_circuit_deferred():
    # is_home_circuit needs a hand-built nationality->country mapping (6.3);
    # it must not appear until that mapping exists and is tested.
    assert "is_home_circuit" not in FEATURES_DATASET_COLUMNS


def test_sprint_features_deferred():
    assert not any(c.startswith("sprint") or "sprint" in c
                   for c in FEATURES_DATASET_COLUMNS)


# ---------------------------------------------------------------------------
# Weather (Phase 4 Tranche B): per-race left-join, missing raceId -> NaN
# ---------------------------------------------------------------------------

def test_weather_broadcasts_to_every_driver_in_the_race():
    master = pd.DataFrame(_two_car_team(1, 1, winner_driver=1))
    weather = pd.DataFrame([{
        "raceId": 1, "race_precip_mm": 2.5, "race_temp_c": 18.0,
        "quali_precip_mm": 0.0, "conditions_changed": True,
    }])
    out = add_weather_features(master, weather)
    assert len(out) == len(master)
    assert (out["race_precip_mm"] == 2.5).all()
    assert (out["conditions_changed"] == True).all()  # noqa: E712


def test_weather_missing_raceid_is_nan_not_an_error():
    master = pd.DataFrame(_two_car_team(1, 1, winner_driver=1))
    out = add_weather_features(master, _empty_weather())
    assert len(out) == len(master)
    for col in WEATHER_FEATURES:
        assert out[col].isna().all()


def test_weather_merge_rejects_duplicate_raceid():
    master = pd.DataFrame(_two_car_team(1, 1, winner_driver=1))
    dup_weather = pd.DataFrame([
        {"raceId": 1, "race_precip_mm": 0.0, "race_temp_c": 20.0,
         "quali_precip_mm": None, "conditions_changed": None},
        {"raceId": 1, "race_precip_mm": 5.0, "race_temp_c": 15.0,
         "quali_precip_mm": None, "conditions_changed": None},
    ])
    with pytest.raises(Exception, match="many_to_one|Merge keys"):
        add_weather_features(master, dup_weather)


# ---------------------------------------------------------------------------
# Wet-condition form (Phase 4 Tranche B item 2): shrunk wet-minus-dry delta
# ---------------------------------------------------------------------------

def _solo_driver_race(race_id, rnd, driver_id, position_order, precip_mm, year=2020):
    """One row: driver is their own single-car constructor (id == driverId),
    so constructor-level wet-form collapses to the single-car case and
    doesn't complicate verifying the driver-level math."""
    return _entry(
        raceId=race_id, round=rnd, year=year, driverId=driver_id,
        constructorId=driver_id, positionOrder=position_order,
        race_precip_mm=precip_mm, race_temp_c=20.0,
        quali_precip_mm=None, conditions_changed=None,
    )


def test_wet_dry_delta_hand_computed():
    # Driver A: much better in the wet (finish 1) than the dry (finish 10).
    # Driver B: no difference at all (finish 5 in both). Same 4 races, so
    # field_wide_delta (pooled across A and B) is checkable independently.
    rows = []
    for r, (precip, order_a, order_b) in enumerate(
        [(1.0, 1, 5), (0.0, 10, 5), (1.0, 1, 5), (0.0, 10, 5)], start=1
    ):
        rows.append(_solo_driver_race(r, r, driver_id=1, position_order=order_a, precip_mm=precip))
        rows.append(_solo_driver_race(r, r, driver_id=2, position_order=order_b, precip_mm=precip))
    out = add_wet_form_features(pd.DataFrame(rows)).set_index(["raceId", "driverId"])

    # Races 1-2: driver A/B each have at most one of (wet, dry) history ->
    # raw delta undefined for both -> shrinkage collapses to field_wide_delta,
    # which is itself undefined (no driver has a defined raw delta yet) -> NaN.
    assert pd.isna(out.loc[(1, 1), "driver_wet_dry_delta"])
    assert pd.isna(out.loc[(2, 1), "driver_wet_dry_delta"])

    # Race 3: A's raw = wet_avg(1) - dry_avg(10) = -9, wet_n=1.
    # B's raw = wet_avg(5) - dry_avg(5) = 0, wet_n=1.
    # field_wide_delta = mean(-9, 0) = -4.5. weight = 1/(1+K).
    weight = 1.0 / (1.0 + SHRINKAGE_K)
    expected_a = weight * -9.0 + (1 - weight) * -4.5
    expected_b = weight * 0.0 + (1 - weight) * -4.5
    assert out.loc[(3, 1), "driver_wet_dry_delta"] == pytest.approx(expected_a)
    assert out.loc[(3, 2), "driver_wet_dry_delta"] == pytest.approx(expected_b)


def test_wet_dry_delta_zero_wet_history_equals_field_average():
    # Driver C never races in the wet at all (n_wet stays 0 forever) while
    # A/B (same fixture as above) do -- C's delta must equal field_wide_delta
    # exactly (weight=0), not 0.0 and not NaN, once the field has any signal.
    rows = []
    for r, (precip, order_a, order_b) in enumerate(
        [(1.0, 1, 5), (0.0, 10, 5), (1.0, 1, 5), (0.0, 10, 5)], start=1
    ):
        rows.append(_solo_driver_race(r, r, driver_id=1, position_order=order_a, precip_mm=precip))
        rows.append(_solo_driver_race(r, r, driver_id=2, position_order=order_b, precip_mm=precip))
        # Driver C only ever races in the dry.
        rows.append(_solo_driver_race(r, r, driver_id=3, position_order=6, precip_mm=0.0))
    out = add_wet_form_features(pd.DataFrame(rows)).set_index(["raceId", "driverId"])

    # Expected field_wide_delta at race 4, from A/B's own raw deltas at that
    # row (both wet_n=2 by then).
    weight_ab = 2.0 / (2.0 + SHRINKAGE_K)
    # A: wet_avg([1,1])=1, dry_avg([10])=10 -> raw=-9. B: wet_avg([5,5])=5, dry_avg([5])=5 -> raw=0.
    field_wide = (-9.0 + 0.0) / 2
    assert out.loc[(4, 1), "driver_wet_dry_delta"] == pytest.approx(
        weight_ab * -9.0 + (1 - weight_ab) * field_wide
    )
    # Driver C: n_wet=0 forever -> weight=0 -> delta collapses exactly to
    # the field-wide prior, not 0.0 and not NaN.
    assert out.loc[(4, 3), "driver_wet_dry_delta"] == pytest.approx(field_wide)
    assert out.loc[(4, 3), "driver_wet_dry_delta"] != 0.0


def test_wet_form_missing_precip_excluded_from_both_averages():
    # A race with unknown precipitation (NaN) must contribute to NEITHER the
    # wet nor the dry running average for that driver -- not silently "dry".
    rows = [
        _solo_driver_race(1, 1, driver_id=1, position_order=1, precip_mm=1.0),   # wet
        _solo_driver_race(2, 2, driver_id=1, position_order=99, precip_mm=None),  # unknown -- must be ignored
        _solo_driver_race(3, 3, driver_id=1, position_order=10, precip_mm=0.0),  # dry
        _solo_driver_race(4, 4, driver_id=1, position_order=2, precip_mm=1.0),   # wet again
    ]
    out = add_wet_form_features(pd.DataFrame(rows)).set_index("raceId")
    # At race 4: prior wet=[1] (race 1 only; race 2's NaN excluded), avg=1.
    # prior dry=[10] (race 3), avg=10. raw = 1 - 10 = -9, wet_n=1.
    # Single driver here -> field_wide_delta == the driver's own raw delta,
    # so the shrinkage weight is a no-op regardless of its value (covered by
    # test_wet_dry_delta_zero_wet_history_equals_field_average instead).
    assert out.loc[4, "driver_wet_dry_delta"] == pytest.approx(-9.0)


def test_constructor_wet_form_excludes_teammate_same_race():
    # Two cars, same constructor: teammate's SAME-RACE finish must never
    # appear in either car's own row for that race.
    rows = []
    for r, (precip, order_1, order_2) in enumerate(
        [(1.0, 1, 2), (0.0, 10, 11), (1.0, 3, 4), (0.0, 12, 13)], start=1
    ):
        rows.append(_entry(raceId=r, round=r, driverId=1, constructorId=1,
                           positionOrder=order_1, race_precip_mm=precip,
                           race_temp_c=20.0, quali_precip_mm=None, conditions_changed=None))
        rows.append(_entry(raceId=r, round=r, driverId=2, constructorId=1,
                           positionOrder=order_2, race_precip_mm=precip,
                           race_temp_c=20.0, quali_precip_mm=None, conditions_changed=None))
    out = add_wet_form_features(pd.DataFrame(rows))
    # Races 1-2 are before the constructor has BOTH a prior wet and a prior
    # dry race on record -> undefined raw delta, no other constructor to
    # pool a field-wide prior from either -> NaN, not a leaked same-race value.
    assert out.loc[out["raceId"].isin([1, 2]), "constructor_wet_dry_delta"].isna().all()
    # From race 3 on (1+ prior race of each type exists), both teammates
    # share one identical, non-NaN constructor-level value per race.
    for race_id in (3, 4):
        vals = out.loc[out["raceId"] == race_id, "constructor_wet_dry_delta"]
        assert vals.notna().all()
        assert vals.nunique() == 1


# ---------------------------------------------------------------------------
# Pipeline composition and validation
# ---------------------------------------------------------------------------

def _small_universe():
    master = pd.DataFrame(
        _two_car_team(1, 1, winner_driver=1) + _two_car_team(2, 2, winner_driver=2)
    )
    ds = pd.DataFrame([
        _driver_standing(1, 1, 25, 1, 1), _driver_standing(1, 2, 0, 2, 0),
        _driver_standing(2, 1, 25, 2, 1), _driver_standing(2, 2, 25, 1, 1),
    ])
    cs = pd.DataFrame([
        _constructor_standing(1, 1, 25, 1, 1),
        _constructor_standing(2, 1, 50, 1, 2),
    ])
    return master, ds, cs


def test_build_features_schema_and_row_count():
    master, ds, cs = _small_universe()
    features = build_features(master, ds, cs, _empty_weather())
    assert list(features.columns) == list(FEATURES_DATASET_COLUMNS)
    assert len(features) == len(master)
    result = validate_features(features, expected_row_count=len(master))
    assert result.passed, result.errors


def test_validate_features_catches_duplicates_and_row_count():
    master, ds, cs = _small_universe()
    features = build_features(master, ds, cs, _empty_weather())

    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    result = validate_features(duplicated, expected_row_count=len(master))
    assert not result.passed

    result = validate_features(features, expected_row_count=len(master) + 1)
    assert not result.passed


# ---------------------------------------------------------------------------
# End-to-end smoke test against the real project data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MASTER_DATASET_PATH.exists(),
    reason="master_dataset.parquet not built (run src.pipelines.build_dataset)",
)
@pytest.mark.skipif(
    not WEATHER_CSV_PATH.exists(),
    reason="race_weather.csv not built (run scripts/backfill_weather.py)",
)
def test_end_to_end_smoke_real_data():
    from src.features.standings import load_standings
    from src.features.weather import load_race_weather

    master = pd.read_parquet(MASTER_DATASET_PATH)
    ds, cs = load_standings()
    weather = load_race_weather()
    features = build_features(master, ds, cs, weather)

    assert len(features) == len(master)
    assert list(features.columns) == list(FEATURES_DATASET_COLUMNS)
    result = validate_features(features, expected_row_count=len(master))
    assert result.passed, result.errors
    # Target must survive the pipeline unchanged.
    assert features["winner"].sum() == master["winner"].sum()
    # Spot leakage check on real data: nobody's rolling window may exceed
    # its own size, and standing positions are >= 1 where present.
    assert features["driver_wins_last_5"].max() <= 5
    valid_standing = features["driver_standing_position_prev"].dropna()
    assert (valid_standing >= 1).all()
