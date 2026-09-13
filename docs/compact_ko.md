# Compact Kreiss-Oliger dissipation

Select Compact KO independently of the first-derivative implementation:

```toml
"dsolve::SOLVER_DISSIPATION_METHOD" = "compact_ko"
"dsolve::SOLVER_COMPACT_KO_SCHEME" = "radius1"
"dsolve::KO_DISS_SIGMA" = 0.4
"dsolve::SOLVER_FD_ORDER" = 6
```

`compact_ko` uses the new Compact KO implementation; `explicit_ko` uses
Dendro's existing explicit KO filters; `none` disables KO dissipation.
`radius1` is currently the only supported Compact KO scheme. Radius2 is not
supported. `KO_DISS_SIGMA` controls the dissipation strength.

To select explicit KO instead, replace the Compact KO selection with:

```toml
"dsolve::SOLVER_DISSIPATION_METHOD" = "explicit_ko"
"dsolve::SOLVER_EXPLICIT_KO_ORDER" = 6
"dsolve::KO_DISS_SIGMA" = 0.4
```

The existing explicit KO6 filter requires padding width 4 or 5 (for example,
element order 8 supplies padding width 4). The simplified Compact KO example
uses element order 6 and padding width 3, so selecting explicit KO6 there also
requires a compatible element order.

If `SOLVER_DISSIPATION_METHOD` is omitted,
each CPU RHS path preserves its previous method: `solverrhs` uses Compact KO,
and `solverrhs_compact_derivs` uses explicit KO. The latter is normally selected
by `EM4_ENABLE_COMPACT_DERIVS=ON`. CUDA does not support these runtime overrides.
Selecting a Compact KO scheme alone does not enable Compact KO.

`SOLVER_EXPLICIT_KO_ORDER` optionally selects the existing KO2/KO4/KO6/KO8
filter. When omitted, Dendro uses its existing element-order-based default.
It never selects a Compact KO scheme. `KO_DISS_SIGMA` is unchanged.

Deprecated parser-only aliases are `SOLVER_KO_DISS_ORDER`,
`SOLVER_FD_DERIV_ORDER`, and `SOLVER_HERMITE_KO_VARIANT`, all with the same
`dsolve::` prefix. Supplying both old and new names is an error. Only old
selector 1 maps to `radius1`; selector 2 is rejected because its formula has
not been validated. No radius-2 scheme or accuracy-order label is advertised.

## Gradients and boundaries

Dendro owns the fixed radius-one formula and adds it to one field using three
read-only directional first derivatives. EM4 supplies its eight evolution
fields after RHS evaluation and physical boundary conditions. Explicit KO and
Compact KO are mutually exclusive: only explicit KO may overwrite gradients
as scratch storage after RHS/BC use. No per-evaluation operator allocation,
matrix construction, or global function-pointer selection is performed.

The field is unzipped, with ghost field values populated by Dendro (including
AMR interpolation). Derivatives are computed per block; EM4 does not exchange
their ghosts. Thus field padding does not establish gradient validity there.
EM4 conservatively supplies `gradient_padding_width = padding_width`.
For each direction, the Compact KO stencil must stay inside that valid region.
Only the normal term on the first/last interior layer is omitted; valid
tangential terms remain. This replaces the old blanket `PW+2` exclusion.
Physical boundary planes indicated by `bflag` remain unchanged, preserving
the solver's boundary conditions. No one-sided Compact KO closure is provided.

This is still a restriction at internal and coarse/fine block interfaces.
Full normal dissipation there requires computing/exchanging valid first-
derivative halos, with a consistent AMR transfer policy. The library supports
that via a smaller `gradient_padding_width`; this refactor does not introduce
a new MPI gradient exchange or assume interpolated field ghosts are gradients.
Full AMR-interface accuracy, decomposition independence, and long-time
stability remain unvalidated.

## Legacy finite differences

`SOLVER_FD_ORDER` retains the behavior of commit 4d8daa1: order 4 accepts padding
2/3/4, order 6 accepts 3/4/5, and order 8 accepts 4/5. Order 6 still uses
fourth-order second derivatives. Compared with main, order-6/padding-2 no longer
falls back to order 4, and eighth-order z dispatch and padding conditionals are
corrected. This setting does not replace `SOLVER_DERIVTYPE_FIRST/SECOND` in the
Dendro derivative path. Existing compile-time order flags may still appear in
legacy diagnostic output; the parameter dump reports the runtime FD order.
