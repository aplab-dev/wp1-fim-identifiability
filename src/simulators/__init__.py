"""Tumor dynamics simulators.

Phase 2 implementation (started 2026-05-02 session 14).

Currently exposed:
- ``lv_2pop_multdeath`` : 2-population L-V with multiplicative drug-death.
  Theory tribe (matches Strobl 2021 / Gallagher 2025 / Wang & Lei 2025).
  See Derivation 1 in `docs/notes/derivations.md`.
- ``lv_3pop_kshift`` : 3-population K-shift L-V. Clinical tribe
  (matches Zhang 2017 / Cunningham 2020 / West 2020 lineage). T+, TP, T-
  cell types with carrying-capacity-shift drug entry.
- ``psa_dynamics`` : First-order linear ODE filter mapping cell counts
  to serum PSA. Zhang 2017 / Brady-Nicholls 2020 share this structure.
"""

from .lv_2pop_multdeath import LV2PopMultDeath, LV2PopParams, LV2PopResult  # noqa: F401
from .lv_3pop_kshift import LV3PopKShift, LV3PopParams, LV3PopResult  # noqa: F401
from .psa_dynamics import PSAParams, integrate_psa_from_cells, psa_steady_state  # noqa: F401
