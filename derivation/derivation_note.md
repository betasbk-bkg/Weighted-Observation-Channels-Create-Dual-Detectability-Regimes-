# Derivation note: mean- and variance-channel behaviour under self-referential clip-cosine weighting

Companion note to the manuscript *"Weighted observation channels split coordinated attacks
into detectability regimes in crowd-sourced continuous control"*. This note records the
first-principles derivations behind Results, with the numerical checks used to validate
each step. Notation follows the manuscript.

## 1. Setting (identities taken from the simulation code)

Eight unit directions V_d, d = 0..7. Weighted aggregation: w_i = max(V_i . m-hat, 0)^2,
gamma_C = |sum w_i V_i| / sum w_i, with m-hat the unit sample-mean direction of the
current tick (self-referential). While the honest majority holds, m-hat ~ +x, and the
clip passes only three directions with non-zero weight: (w, V_x) = dir0 (1, 1),
dir+/-1 (0.5, cos 45deg).

Per-voter expected moments (derived from the vote-generation distributions; five decimals):

| population | E[V_x] | E[w] | E[w V_x] |
|---|---|---|---|
| honest (0.7368 accurate + 0.2105 lagged + 0.0527 wide) | 0.99614 | 0.99341 | 0.99148 |
| uniform troll (and the marginal of an aligned scatter troll) | 0 | 0.25 | 0.21339 |
| aligned opposite troll (dirs {3,4,5}, each 1/3) | -0.80474 | **0 (clip identity)** | **0** |

## 2. Closed forms (mean level)

- gamma_F(opposite; p, c) = (1-p)*0.99614 - 0.80474*p*c
- gamma_F(scatter; p, c)  = (1-p)*0.99614   (c-independent)
- gamma_C(opposite; p, c) = [(1-p)*0.99148 + p(1-c)*0.21339] / [(1-p)*0.99341 + p(1-c)*0.25]
- gamma_C(scatter; p, c)  ~ the c = 0 form of the line above at mean level (c-independent in expectation)

Baselines (p = 0.05, c = 0): gamma_F0 = 0.9463, gamma_C0 = 0.9962.

Engine validation: eight comparison points, six with absolute error <= 0.0010; the
c = 1 opposite prediction matches to four decimals (0.9981 / 0.9981).

## 3. Mean-channel blindness (coherent attack) - an identity

At c = 1 (opposite), every aligned vote is anti-parallel to m-hat, so its clip weight is
exactly zero. The malicious block contributes to neither numerator nor denominator of
gamma_C, so gamma_C equals the honest-only value 0.9981 > baseline 0.9962, i.e.
Delta-gamma_C < 0 for all p in the honest-majority regime. Verified insensitive to the
detector variance floor from 1e-4 down to 1e-10.

## 4. Variance-channel sign asymmetry

Law-of-total-variance decomposition (verified at p = 0.4, c = 0.8): the between-support
term contributes about 90% of Var[gamma_C]; the within-support (multinomial allocation)
term about 10%.

- **Opposite:** Var[gamma_C] proportional to (1-c) (linear fit R^2 = 0.977 at p = 0.4).
  Mechanism: the clip removes the aligned block identically, leaving only the shrinking
  uniform population, of size proportional to (1-c). At c = 1 the aligned fluctuation
  source vanishes - the variance counterpart of the identity in section 3.
- **Scatter:** Var[gamma_C] increases monotonically and superlinearly in c, with a
  leading quadratic term (the aligned count enters linearly and the between-support
  variance quadratically). The exact exponent is regime-dependent: near 2 at moderate p,
  closer to 1.3 at strong p, because of an additive baseline and higher-order wobble;
  a universal square law is *not* claimed. An additive fit Var = a + b*c^2 gives
  R^2 = 0.998 at p = 0.3, degrading to 0.89 at p = 0.5.
- Scatter-to-opposite variance ratio: ~1.2 at c = 0.1 rising to ~150 at c = 1 (p = 0.4).

A closed-form finite sum over the 56 three-direction support sets reproduces the
moderate-attack variance and its scaling but underestimates the strong-attack corner
(see section 5).

## 5. Partial closure of the strong-attack corner (mean-direction wobble)

Decomposing the variance with the reference direction frozen at +x versus recomputed
per tick shows the wobble contribution grows from ~9% of Var[gamma_C] at moderate
attack to ~83% at the strongest corner tested (p = 0.5, c = 1).

**Closed (angular variance).** The angular variance of m-hat is
Var[theta] ~ [n_al^2 * Var_sup(s-bar) + n_al * v_w + n_uni * 1/2] / (N^2 * E[m_x]^2),
with the purely geometric constants Var_sup(s-bar) = 0.11905 (between-support variance
of the mean sine over the 56 support sets) and v_w = 0.38095 (mean within-support
variance of the sine). Prediction/measurement ratio 1.03-1.14 across p in {0.3, 0.4,
0.5} and c in {0.4, 0.8, 1.0}.

**Not closed (propagation).** First-order propagation, (d gamma_C / d theta)^2 *
Var[theta], captures only 7-34% of the wobble contribution and degrades at strong
attack, because the same support draw drives both the wobble and the weighted sum: the
two are coupled and do not separate as a product. The strong-attack absolute variance
is therefore left as an unresolved stopping-time-class approximation problem. Only the
sign, the scaling, and the angular variance are claimed as closed.

## 6. Pre-registered predictions (fixed before the 200-replicate campaign)

- PV1: scatter Var[gamma_C](c) monotone increasing and superlinear (log-log slope > 1).
  *Note:* an earlier registration of a pure power law with slope in [1.5, 2.0] was
  falsified by the additive baseline at low c and corrected to the present form; the
  correction is recorded here for transparency.
- PV2: opposite Var[gamma_C](c) proportional to (1-c), monotone decreasing (linear
  R^2 > 0.9).
- PV3: scatter/opposite variance ratio >= 50 near c = 1.
- PV4: the component-wise (weak-axis) observer either shows a c-structure under scatter
  (supporting it as a new channel) or is uniformly censored (rejecting it).

Outcome at MC = 200: PV1-PV3 held; PV4 resolved as rejection (uniformly censored - a
dead channel), reported as a negative result in the manuscript.

## 7. Known limitations of this note

The absolute detection boundary p* is a calibrated quantity (CUSUM constants, including
the warm-up variance floor); the closed-form stopping-time approximation is left open.
The strong-attack absolute variance is unresolved (section 5). All derivations assume
the honest-majority regime (m-hat ~ +x), eight-direction quantization, and N = 120.
