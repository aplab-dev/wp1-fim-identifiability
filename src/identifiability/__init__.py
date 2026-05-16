"""Identifiability analysis — Phase 2 Stage 2.5b.

Tools for assessing how identifiable L-V parameters are from PSA-only
observation (the practical clinical setting). This is the technical
foundation for "given identifiability limits, what is the narrowest
parameter regime consistent with each patient's PSA?" — a candidate Phase
3 research question.

The core tool is the **Fisher Information Matrix (FIM)**:

    FIM[i, j] = ∫ (∂y(t)/∂θ_i)(∂y(t)/∂θ_j) / σ²(t) dt

For PSA-only observation, the FIM is typically rank-deficient — multiple
parameter combinations produce identical PSA trajectories. The rank tells
us how many *effective* parameters we can fit.

Exports:
- ``fim`` — finite-difference FIM computation.
- ``fim_eigendecomposition`` — SVD-based identifiability decomposition.
"""

from .fim import (  # noqa: F401
    FIMResult,
    compute_fim,
    fim_eigendecomposition,
)
