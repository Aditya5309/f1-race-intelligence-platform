"""
src/models/eras.py

Formula 1 regulation-era definitions (Decision 019) — the code-level single
source of truth for era boundaries, mirroring context/domain_knowledge.md
Section 1 (the prose source of truth) and following the Decision-013
precedent of src/features/metadata.py.

WHY ERAS MATTER (concept drift in F1)
-------------------------------------
The FIA periodically rewrites the technical regulations. A major rewrite
changes aerodynamic philosophy, engine architecture, and tyre behaviour at
once, which resets team competitiveness: the car is the dominant performance
factor (~80-90% of results variance), so a constructor dominant under one
ruleset can be midfield under the next (Mercedes 2021 -> 2022). Relationships
a model learns from one era — constructor form persistence, dominance
concentration, even the qualifying-to-race conversion rate — therefore drift
or break at era boundaries. This is CONCEPT DRIFT with known, pre-announced
breakpoints, which is why this project's split strategies are era-aware
(src/models/splits.py, Decisions 008/018/019) and why Decision 013 classifies
features as era-sensitive or stable.

Eras are defined only from 2010 (MODELING_WINDOW_START, Decision 008):
earlier seasons are structurally different (field size, ~50% finish rates,
points systems) and are outside every split strategy. `era_of()` returns
None for pre-2010 years by design.

Era boundaries are public years in advance (regulations are published ahead
of time), so era information is never leakage (domain_knowledge.md Section 1).

MAINTENANCE
-----------
When the FIA announces or starts a new regulation cycle:
1. Close the current final era (set its `end_year`).
2. Append the new era (usually with `end_year=None` while ongoing).
3. Update context/domain_knowledge.md Section 1 in the same change.
Nothing in the splitting logic needs to change; within-era presets for the
new era become available automatically once the era is closed (or has enough
seasons) — see `within_era_strategy` in src/models/splits.py.
"""

from __future__ import annotations

from dataclasses import dataclass

# Decision 008 — the modeling window starts here; no era is defined earlier.
MODELING_WINDOW_START: int = 2010


@dataclass(frozen=True)
class RegulationEra:
    """
    One technical-regulation cycle, inclusive of both boundary years.

    `end_year=None` marks an ongoing era (no closed range yet) — only the
    final entry of REGULATION_ERAS may be ongoing.
    """

    name: str            # stable slug used in code and logs
    label: str           # human-readable name for reports
    start_year: int
    end_year: int | None
    description: str

    @property
    def is_ongoing(self) -> bool:
        return self.end_year is None

    @property
    def year_range(self) -> tuple[int, int]:
        """Inclusive (start, end); loud for ongoing eras — callers that need
        a closed range (e.g. within-era splits) must not guess an end year."""
        if self.end_year is None:
            raise ValueError(
                f"Regulation era '{self.name}' is ongoing — it has no closed "
                "year range yet. Close the era in src/models/eras.py once its "
                "final season is known."
            )
        return (self.start_year, self.end_year)

    def contains(self, year: int) -> bool:
        return year >= self.start_year and (
            self.end_year is None or year <= self.end_year
        )


# ---------------------------------------------------------------------------
# The era table (domain_knowledge.md Section 1; segmentation verified in the
# Decision-012 Section 14 era-nonstationarity analysis and Decision 013).
# ---------------------------------------------------------------------------

V8 = RegulationEra(
    name="v8",
    label="V8 / post-refuelling era",
    start_year=2010,
    end_year=2013,
    description=(
        "Modern-era start: refuelling ban, 25-point wins, ~20-car field, "
        "~84% finish rates. Pirelli sole supplier and DRS from 2011."
    ),
)

HYBRID = RegulationEra(
    name="hybrid",
    label="Hybrid era (V6 turbo-hybrid power units)",
    start_year=2014,
    end_year=2021,
    description=(
        "1.6L V6 turbo-hybrid power units; Mercedes won all eight "
        "constructors' titles — the dataset's most extreme dominance streak. "
        "2017 wider-car reset and the 2021 cost cap land inside this era. "
        "Caveat for within-era work: 2020 ran a COVID-shortened calendar."
    ),
)

GROUND_EFFECT = RegulationEra(
    name="ground_effect",
    label="Ground-effect era",
    start_year=2022,
    end_year=2025,
    description=(
        "Venturi-floor aerodynamics, 18-inch wheels; order reset (Red Bull "
        "dominance 2022-2023, convergence from 2024 under the cost cap). "
        "2025 rows are the project's forward holdout (Decision 012 S13.1)."
    ),
)

FUTURE_ENGINE = RegulationEra(
    name="future_engine",
    label="2026 engine/chassis regulations",
    start_year=2026,
    end_year=None,
    description=(
        "Largest combined chassis + power-unit reset since 2014: ~50% "
        "electric power share, sustainable fuel, active aerodynamics, "
        "22-car field (Audi, Cadillac). Ongoing — no closed range yet."
    ),
)

REGULATION_ERAS: tuple[RegulationEra, ...] = (
    V8,
    HYBRID,
    GROUND_EFFECT,
    FUTURE_ENGINE,
)


def get_era(era: RegulationEra | str) -> RegulationEra:
    """Resolve an era object or slug; loud on unknown names."""
    if isinstance(era, RegulationEra):
        return era
    for candidate in REGULATION_ERAS:
        if candidate.name == era:
            return candidate
    raise KeyError(
        f"Unknown regulation era '{era}'. Known eras: "
        f"{[e.name for e in REGULATION_ERAS]}."
    )


def era_of(year: int) -> RegulationEra | None:
    """The era containing `year`, or None before the 2010 modeling window."""
    for era in REGULATION_ERAS:
        if era.contains(year):
            return era
    return None


# ---------------------------------------------------------------------------
# Import-time integrity (same discipline as src/features/metadata.py):
# the table must start at the modeling window, be contiguous and
# non-overlapping, and only its final entry may be ongoing.
# ---------------------------------------------------------------------------

assert REGULATION_ERAS[0].start_year == MODELING_WINDOW_START, (
    "First regulation era must start at the Decision-008 modeling window."
)
for _prev, _next in zip(REGULATION_ERAS, REGULATION_ERAS[1:]):
    assert _prev.end_year is not None, (
        f"Era '{_prev.name}' is ongoing but is not the final era — close it."
    )
    assert _next.start_year == _prev.end_year + 1, (
        f"Eras '{_prev.name}' and '{_next.name}' are not contiguous — every "
        "modeling-window year must belong to exactly one era."
    )
del _prev, _next
