import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", app_title="Ethiopia Rainfall Regime Clustering")


# ============================================================================
# 0. Setup
# ============================================================================
@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from pathlib import Path
    import time as _time
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.io.shapereader as shpreader
    from shapely import contains_xy

    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    return (
        GaussianMixture,
        KMeans,
        Path,
        _time,
        contains_xy,
        mo,
        mpl,
        np,
        plt,
        shpreader,
        xr,
    )


@app.cell
def _intro(mo):
    mo.md(
        r"""
# Ethiopia Rainfall Regime Clustering
### Comparing K-means (Euclidean), K-means (soft-DTW), and EM Gaussian Mixtures

**Problem.** Ethiopia's rainfall is famously heterogeneous: the highlands have a
single boreal-summer rainy season, the south has a bimodal regime, and the
eastern lowlands are arid. Many downstream products (early warning, agronomy,
index insurance) need a *consistent* partition of the country into regions whose
**rainy-season behaviour is internally similar**.

**Goal.** Find a small number of clusters where the cluster *centroid* is a
faithful summary of the rainy-season climatology of every grid cell assigned to
it. Faithful means two things:

1. The shape (rise, peak, decay) of the cluster mean matches the cell's own.
2. **Onset and cessation timing** is preserved — these are the most operationally
   important parts of the curve, so we cannot let a model silently smooth them
   away.

**Three solutions** we compare in this notebook:

| Model | Distance / likelihood | Assignment | Library |
|---|---|---|---|
| K-means Euclidean | $\sum_t (x_t - \mu_t)^2$ | hard | `sklearn.cluster.KMeans` |
| K-means soft-DTW  | differentiable DTW       | hard | `tslearn.clustering.TimeSeriesKMeans` |
| EM Gaussian Mixture | mixture of $K$ Gaussians | soft | `sklearn.mixture.GaussianMixture` |

The notebook is structured as a guided answer to six questions:

1. What is the right data preparation for each model (scaled vs raw, and *why*)?
2. Why use soft-DTW with `tslearn`, and why is its efficiency so bad?
3. What is the best metric for picking $K$ — and where do time shifts make a real difference?
4. How can EM/GMM be applied here, and is it reasonable?
5. How do we measure performance, given there are no ground-truth labels?
6. How do the three models compare, and which is preferable for this problem?
"""
    )
    return


# ============================================================================
# 1. Parameters
# ============================================================================
@app.cell
def _params(mo):
    data_dir = mo.ui.text(value="data/CHIRPS", label="CHIRPS daily directory")
    start_year = mo.ui.number(value=1998, label="Start year")
    end_year = mo.ui.number(value=2024, label="End year")
    var_name = mo.ui.text(value="precip", label="Precip variable")

    onset_start_week = mo.ui.number(value=15, label="Onset window start (ISO week)")
    onset_end_week = mo.ui.number(value=46, label="Onset window end (ISO week)")
    season_start_week = mo.ui.number(value=22, label="Core season start week")
    season_end_week = mo.ui.number(value=36, label="Core season end week")
    min_season_fraction = mo.ui.number(
        value=0.10, step=0.01, label="Min fraction of annual rainfall in core season"
    )

    selected_k = mo.ui.number(value=5, label="K for the headline runs")
    elbow_max_k = mo.ui.number(value=10, label="Max K to scan for elbow / silhouette / BIC")

    softdtw_subsample = mo.ui.number(
        value=400,
        label="Soft-DTW subsample size (cells; full grid is ~3k)",
    )
    softdtw_gamma = mo.ui.number(
        value=1.0, step=0.1, label="Soft-DTW gamma (smoothing of the min)"
    )

    mo.md("### Parameters")
    mo.vstack(
        [
            data_dir,
            mo.hstack([start_year, end_year, var_name]),
            mo.md("**Onset window** (the slice of the year fed into the clusterer):"),
            mo.hstack([onset_start_week, onset_end_week]),
            mo.md("**Core season** (used to drop dry cells before clustering):"),
            mo.hstack([season_start_week, season_end_week, min_season_fraction]),
            mo.md("**Clustering controls:**"),
            mo.hstack([selected_k, elbow_max_k]),
            mo.md("**Soft-DTW controls** (it's expensive — start small):"),
            mo.hstack([softdtw_subsample, softdtw_gamma]),
        ]
    )
    return (
        data_dir,
        elbow_max_k,
        end_year,
        min_season_fraction,
        onset_end_week,
        onset_start_week,
        season_end_week,
        season_start_week,
        selected_k,
        softdtw_gamma,
        softdtw_subsample,
        start_year,
        var_name,
    )


# ============================================================================
# 2. Data loading and the masked weekly climatology
# ============================================================================
@app.cell
def _data_md(mo):
    mo.md(
        r"""
## 1. Data preparation: from daily rainfall to a per-cell weekly climatology

We want each grid cell $i$ to be summarised by a single vector
$x_i \in \mathbb{R}^{L}$ — its long-run **mean weekly rainfall** across the onset
window (length $L$). That vector is what every clusterer below will see.

Pipeline (no choices about it — this is fixed):

1. Load CHIRPS daily totals for `start_year` .. `end_year`.
2. Resample to weekly totals.
3. Group by ISO week-of-year and average across years → weekly climatology.
4. Restrict to the onset window (default ISO weeks 15–46, i.e. mid-April through
   mid-November — long enough to contain *kiremt*, *belg* and *deyr* shoulders).
5. Drop cells outside Ethiopia and cells that get less than `min_season_fraction`
   of their annual rainfall inside the core season (default 10%) — these are the
   truly arid cells where any seasonal clustering would just be noise.

After this, all three models start from the same matrix $X$ of shape
$(n_{\text{cells}}, L)$.
"""
    )
    return


@app.cell
def _load(Path, data_dir, end_year, mo, start_year, var_name, xr):
    _years = list(range(int(start_year.value), int(end_year.value) + 1))
    _files = [Path(data_dir.value) / f"{y}.nc" for y in _years]
    _existing = [str(f) for f in _files if f.exists()]
    _missing = [str(f) for f in _files if not f.exists()]

    if not _existing:
        raise FileNotFoundError(
            f"No CHIRPS files found under {data_dir.value!r}. "
            "Set the path in the parameters cell."
        )

    ds = xr.open_mfdataset(
        _existing,
        combine="by_coords",
        parallel=True,
        engine="netcdf4",
        chunks={"time": 365},
    )

    _rename = {}
    for _old, _new in [
        ("LATITUDE", "latitude"),
        ("LONGITUDE", "longitude"),
        ("lat", "latitude"),
        ("lon", "longitude"),
    ]:
        if _old in ds.dims or _old in ds.coords:
            _rename[_old] = _new
    if _rename:
        ds = ds.rename(_rename)

    if var_name.value not in ds.data_vars:
        raise KeyError(
            f"Variable {var_name.value!r} not in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[var_name.value]

    mo.vstack(
        [
            mo.md(
                f"Loaded **{len(_existing)}** annual files "
                f"({min(_years)}–{max(_years)}); missing **{len(_missing)}**."
            ),
            mo.plain_text(
                f"Grid: {ds.sizes.get('latitude', '?')} lat × "
                f"{ds.sizes.get('longitude', '?')} lon, "
                f"{ds.sizes.get('time', '?')} daily slices."
            ),
        ]
    )
    return (da,)


@app.cell
def _weekly_clim(da, mo, np, onset_end_week, onset_start_week):
    _weekly_total = da.resample(time="7D").sum(skipna=False)
    _wk = _weekly_total.time.dt.strftime("%V").astype(int)
    _weekly_total = _weekly_total.assign_coords(week=("time", _wk.data))
    weekly_clim = _weekly_total.groupby("week").mean("time").sortby("week")

    X_weekly = weekly_clim.transpose("latitude", "longitude", "week").values
    n_lat, n_lon, _n_week = X_weekly.shape
    annual_total = np.nansum(X_weekly, axis=2)
    annual_total = np.where(annual_total == 0, np.nan, annual_total)

    weeks = weekly_clim["week"].values
    onset_window = (weeks >= int(onset_start_week.value)) & (
        weeks <= int(onset_end_week.value)
    )
    weeks_onset = weeks[onset_window]

    if weeks_onset.size == 0:
        raise ValueError("Onset window selects zero weeks.")

    mo.plain_text(
        f"Weekly climatology grid: {n_lat}×{n_lon}, "
        f"{_n_week} weeks total, {weeks_onset.size} weeks inside the onset window."
    )
    return (
        X_weekly,
        annual_total,
        n_lat,
        n_lon,
        onset_window,
        weekly_clim,
        weeks,
        weeks_onset,
    )


@app.cell
def _ethiopia_mask(contains_xy, np, shpreader, weekly_clim):
    _shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    ethiopia_geom = None
    for _rec in shpreader.Reader(_shp).records():
        if _rec.attributes["NAME"] == "Ethiopia":
            ethiopia_geom = _rec.geometry
            break
    if ethiopia_geom is None:
        raise RuntimeError("Could not find Ethiopia geometry.")

    lon = weekly_clim["longitude"].values
    lat = weekly_clim["latitude"].values
    _lon2d, _lat2d = np.meshgrid(lon, lat)
    ethiopia_mask_2d = contains_xy(
        ethiopia_geom, _lon2d.ravel(), _lat2d.ravel()
    ).reshape(lat.size, lon.size)
    return ethiopia_geom, ethiopia_mask_2d, lat, lon


@app.cell
def _season_mask(
    X_weekly,
    annual_total,
    min_season_fraction,
    np,
    season_end_week,
    season_start_week,
    weeks,
):
    _season = (weeks >= int(season_start_week.value)) & (
        weeks <= int(season_end_week.value)
    )
    _rain_season = np.nansum(X_weekly[:, :, _season], axis=2)
    _frac = _rain_season / annual_total
    keep_mask_2d = _frac >= float(min_season_fraction.value)
    return (keep_mask_2d,)


@app.cell
def _build_X(
    X_weekly,
    ethiopia_mask_2d,
    keep_mask_2d,
    mo,
    np,
    onset_window,
):
    _onset = X_weekly[:, :, onset_window]
    _flat = _onset.reshape(-1, _onset.shape[2])

    valid_mask = (
        np.isfinite(_flat).all(axis=1)
        & keep_mask_2d.reshape(-1)
        & ethiopia_mask_2d.reshape(-1)
    )
    X_raw = _flat[valid_mask]

    if X_raw.shape[0] == 0:
        raise ValueError("No valid grid cells remain after masking.")

    _mu = X_raw.mean(axis=1, keepdims=True)
    _sd = X_raw.std(axis=1, keepdims=True)
    _sd_safe = np.where(_sd == 0, 1.0, _sd)
    X_zscore = (X_raw - _mu) / _sd_safe

    _total_safe = X_raw.sum(axis=1, keepdims=True)
    _total_safe = np.where(_total_safe == 0, 1.0, _total_safe)
    X_unitsum = X_raw / _total_safe

    mo.plain_text(
        f"X built. n_cells = {X_raw.shape[0]}, weeks per cell = {X_raw.shape[1]}.\n"
        f"Three preps available: X_raw (mm/week), X_zscore (per-cell standardised), "
        f"X_unitsum (per-cell L1-normalised)."
    )
    return X_raw, X_unitsum, X_zscore, valid_mask


@app.cell
def _prep_math(mo):
    mo.md(
        r"""
### Three data preparations, and what they do mathematically

For each cell $i$ we have a raw vector $x_i \in \mathbb{R}^L_{\geq 0}$ (mm of rain
per week). We can feed the clusterer one of:

- **Raw** $x_i$ — preserves both **shape** (when does it rain) and **magnitude**
  (how much). Wet highlands and dry lowlands become very far apart even if their
  seasonal *shape* is identical.
- **Z-score per cell** $\tilde{x}_i = (x_i - \bar{x}_i\mathbf{1}) / s_i$ — subtract
  the cell's own mean, divide by its own standard deviation. Removes magnitude
  entirely. Two cells whose curves are constant multiples of each other become
  identical.
- **L1-normalised per cell** $u_i = x_i / \sum_t x_{i,t}$ — turns each curve into a
  probability distribution over the weeks. Removes total volume but keeps the
  relative weights, so a "70% of rain in July" cell still looks different from a
  "30% in July, 30% in October" cell.

**Pairing each prep with its model:**

- **K-means Euclidean** uses squared $\ell_2$ distance, so anything that isn't
  normalised will be dominated by absolute volume — usually you want z-score or
  L1 here, depending on whether you care about *all* shape or *only the shape of
  the share*.
- **K-means soft-DTW** also uses squared distances but along the warped alignment,
  so the same logic applies; the per-cell standardisation is essential or wet/dry
  contrast will swamp the soft-DTW signal.
- **GMM** assumes Gaussian components. Z-score does not change Gaussian-ness per
  cell — but globally, the raw rainfall distribution is heavy-tailed (lots of
  low-rainfall cells, a long tail of wet ones). Centring each cell to zero and
  unit variance gets each *feature* (each week) much closer to a unimodal
  Gaussian across cells, which is what the GMM cares about.

We will use **z-score** as the default for all three models so that we are
comparing *shape* clusters, and we will explicitly demo the raw version for
Euclidean K-means to show what magnitude-driven clustering looks like.
"""
    )
    return


# ============================================================================
# 3. K-means with Euclidean distance
# ============================================================================
@app.cell
def _kmeans_md(mo):
    mo.md(
        r"""
## 2. K-means with Euclidean distance

### Math
Given $n$ vectors $\{x_i\}$, K-means chooses $K$ centroids $\{\mu_k\}$ and an
assignment $c: i \mapsto k$ that minimise
$$
J(\mu, c) \;=\; \sum_{i=1}^{n}\, \lVert x_i - \mu_{c(i)} \rVert_2^2 .
$$
Lloyd's algorithm alternates: assign each point to its nearest centroid, then
recompute each centroid as the mean of its members. Converges to a **local**
optimum, hence we use `n_init=10` random restarts.

### Assumptions, and how they map to this problem
K-means implicitly assumes:

1. Clusters are **roughly spherical** in the feature space — equal variance in
   every direction (every week here). After per-cell z-scoring, every grid cell
   has equal energy, which makes this far more reasonable than clustering on
   raw mm.
2. **All weeks are equally important.** Euclidean distance gives equal weight to
   mismatch in week 17 (start of *belg*) and week 30 (peak of *kiremt*). For this
   problem that is acceptable: the operational quantity we want to preserve is
   the *shape* of the seasonal curve in time.
3. **No tolerance for time shifts.** A cell whose curve is the cluster mean
   shifted by one week looks "as different" from the centroid as a cell with a
   totally different shape. **For the rainfall onset/cessation problem this is a
   feature, not a bug** — see Q3b below.

Under these assumptions, the centroid is the **best representative** of its
members in the squared-error sense, which directly answers the goal stated at the
top.
"""
    )
    return


@app.cell
def _scale_choice(mo):
    feature_space = mo.ui.dropdown(
        options=["Z-score per cell", "Raw weekly rainfall", "L1-normalised per cell"],
        value="Z-score per cell",
        label="Feature space for the headline K-means run",
    )
    feature_space
    return (feature_space,)


@app.cell
def _select_X(X_raw, X_unitsum, X_zscore, feature_space):
    if feature_space.value == "Z-score per cell":
        X_for_k = X_zscore
        feature_label = "z-score"
    elif feature_space.value == "L1-normalised per cell":
        X_for_k = X_unitsum
        feature_label = "share of annual"
    else:
        X_for_k = X_raw
        feature_label = "mm/week"
    return X_for_k, feature_label


@app.cell
def _kmeans_fit(KMeans, X_for_k, X_raw, mo, np, selected_k):
    K_kmeans = int(selected_k.value)
    _km = KMeans(n_clusters=K_kmeans, random_state=0, n_init=10)
    km_labels = _km.fit_predict(X_for_k)

    cluster_ids_km = np.unique(km_labels)
    centroids_km_feat = np.array(
        [X_for_k[km_labels == _c].mean(axis=0) for _c in cluster_ids_km]
    )
    centroids_km_raw = np.array(
        [X_raw[km_labels == _c].mean(axis=0) for _c in cluster_ids_km]
    )
    mo.plain_text(
        f"Fitted K-means (Euclidean) with K={K_kmeans}, "
        f"inertia={_km.inertia_:,.1f}."
    )
    return centroids_km_feat, centroids_km_raw, cluster_ids_km, km_labels


@app.cell
def _kmeans_plots(
    centroids_km_feat,
    centroids_km_raw,
    cluster_ids_km,
    ethiopia_geom,
    feature_label,
    km_labels,
    lat,
    lon,
    mo,
    mpl,
    n_lat,
    n_lon,
    np,
    plt,
    valid_mask,
    weeks_onset,
):
    _flat = np.full(n_lat * n_lon, np.nan)
    _flat[valid_mask] = km_labels
    label_grid_km = _flat.reshape(n_lat, n_lon)

    _cmap = plt.get_cmap("tab10", len(cluster_ids_km))
    _norm = mpl.colors.BoundaryNorm(
        np.arange(-0.5, len(cluster_ids_km) + 0.5, 1), _cmap.N
    )

    _fig_map, _ax_map = plt.subplots(figsize=(7.5, 6.5))
    _ax_map.pcolormesh(lon, lat, label_grid_km, shading="auto", cmap=_cmap, norm=_norm)
    _geoms = ethiopia_geom.geoms if hasattr(ethiopia_geom, "geoms") else [ethiopia_geom]
    for _g in _geoms:
        _x, _y = _g.exterior.xy
        _ax_map.plot(_x, _y, color="black", linewidth=1.0)
    _ax_map.set_aspect("equal")
    _ax_map.set_title(f"K-means (Euclidean), K={len(cluster_ids_km)}")

    _fig_c, (_ax_f, _ax_r) = plt.subplots(1, 2, figsize=(11, 4))
    for _i, _c in enumerate(cluster_ids_km):
        _ax_f.plot(weeks_onset, centroids_km_feat[_i], color=_cmap(_i), label=f"C{_c}")
        _ax_r.plot(weeks_onset, centroids_km_raw[_i], color=_cmap(_i), label=f"C{_c}")
    _ax_f.set_title(f"Centroids in feature space ({feature_label})")
    _ax_r.set_title("Same clusters, mean raw rainfall (mm/week)")
    for _ax in (_ax_f, _ax_r):
        _ax.set_xlabel("ISO week")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=8)

    mo.vstack([mo.mpl.interactive(_fig_map), mo.mpl.interactive(_fig_c)])
    return (label_grid_km,)


# ============================================================================
# 4. Q3 — picking K
# ============================================================================
@app.cell
def _pickk_md(mo):
    mo.md(
        r"""
### Q3. Picking $K$ — elbow vs silhouette

Two standard heuristics, both unsupervised:

- **Elbow on inertia.** $J(K) = \sum_i \min_k \lVert x_i - \mu_k \rVert^2$ is
  monotonically non-increasing in $K$. We look for a "knee" where adding another
  cluster stops yielding much improvement.
- **Silhouette score.** For each point $i$, let $a(i)$ be its mean distance to
  its own cluster and $b(i)$ its mean distance to the nearest other cluster.
  Silhouette is $s(i) = (b(i) - a(i)) / \max(a(i), b(i)) \in [-1, 1]$. The mean
  over all points is the silhouette score; values near 1 mean tight,
  well-separated clusters. **Silhouette has a maximum**, so unlike the elbow it
  directly nominates a $K$.

Implementation note: silhouette here uses `tslearn.clustering.silhouette_score`
with `metric="euclidean"`. We use the same machinery in section 3 with
`metric="softdtw"` — using the *same* metric you fit with is the only way the
score is internally consistent.
"""
    )
    return


@app.cell
def _pickk_run(KMeans, X_for_k, _time, elbow_max_k, mo, np):
    from tslearn.clustering import silhouette_score as ts_silhouette_score

    K_grid = list(range(2, int(elbow_max_k.value) + 1))
    inertias = []
    sils_eucl = []

    _rng = np.random.default_rng(0)
    _n_sample = min(2000, X_for_k.shape[0])
    _idx = _rng.choice(X_for_k.shape[0], size=_n_sample, replace=False)
    _X_sil = X_for_k[_idx]

    for _k in K_grid:
        _t0 = _time.time()
        _km = KMeans(n_clusters=_k, random_state=0, n_init=10).fit(X_for_k)
        inertias.append(_km.inertia_)
        _labels_sil = _km.predict(_X_sil)
        sils_eucl.append(
            float(ts_silhouette_score(_X_sil, _labels_sil, metric="euclidean"))
        )

    mo.plain_text(
        "Scanned K=" + ", ".join(str(_k) for _k in K_grid)
        + f". Sampled {_n_sample} cells for silhouette."
    )
    return K_grid, inertias, sils_eucl, ts_silhouette_score


@app.cell
def _pickk_plot(K_grid, inertias, mo, plt, sils_eucl):
    _fig, (_ax_elbow, _ax_sil) = plt.subplots(1, 2, figsize=(11, 4))
    _ax_elbow.plot(K_grid, inertias, marker="o")
    _ax_elbow.set_xlabel("K")
    _ax_elbow.set_ylabel("Inertia (within-cluster SS)")
    _ax_elbow.set_title("Elbow")
    _ax_elbow.grid(alpha=0.3)

    _ax_sil.plot(K_grid, sils_eucl, marker="o", color="C1")
    _best_k = K_grid[int(max(range(len(sils_eucl)), key=lambda i: sils_eucl[i]))]
    _ax_sil.axvline(_best_k, color="C1", linestyle="--", alpha=0.5)
    _ax_sil.set_xlabel("K")
    _ax_sil.set_ylabel("Mean silhouette (Euclidean)")
    _ax_sil.set_title(f"Silhouette — argmax at K={_best_k}")
    _ax_sil.grid(alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _pickk_recommendation(mo):
    mo.md(
        r"""
**Recommendation.** For this problem, prefer **silhouette over elbow**:

- Inertia is in mm² (or z-score² depending on prep) and its absolute decrease
  has no scale to compare against — the elbow is a vibes call.
- Silhouette has a hard maximum and a well-defined zero point.
- Silhouette also penalises *too many* clusters: when $K$ is too high,
  neighbouring clusters' $b(i)$ collapses toward $a(i)$ and the score plummets —
  exactly the regime we want to avoid for an interpretable regime map.

**But: silhouette is a sanity check, not a verdict.** For a deliverable regime
map, domain knowledge (Ethiopia has roughly 4–6 well-known rainfall regimes)
should override a silhouette argmax of, say, $K=2$.
"""
    )
    return


# ============================================================================
# Q3b — time shifts
# ============================================================================
@app.cell
def _shift_md(mo):
    mo.md(
        r"""
### Q3b. When time shifts *should not* be ignored

Soft-DTW's selling point is tolerance to time-warping: a curve that arrives a
week early and leaves a week early is treated as "the same shape". For many
problems (gesture recognition, ECG morphology) that is exactly what you want.

**For Ethiopian rainfall regimes it is exactly what you do not want.** A grid
cell whose *kiremt* onset is in week 22 versus week 26 is agronomically a very
different cell — different planting calendar, different crop choice, different
insurance trigger. We *want* the clusterer to call them different. The whole
point of separating the country into rainfall regions is to capture timing
differences.

The demo below illustrates this. We take the Euclidean-K-means centroid for one
cluster, shift it by ±1, ±2 weeks, and ask: how does the Euclidean distance
between the original and the shifted version compare to soft-DTW distance?
Soft-DTW collapses the shifts, Euclidean does not.
"""
    )
    return


@app.cell
def _shift_demo(centroids_km_feat, mo, np, plt, weeks_onset):
    from tslearn.metrics import soft_dtw

    _base = centroids_km_feat[0]

    def _shifted(curve, k):
        if k == 0:
            return curve
        out = np.empty_like(curve)
        if k > 0:
            out[:k] = curve[0]
            out[k:] = curve[:-k]
        else:
            kk = -k
            out[-kk:] = curve[-1]
            out[:-kk] = curve[kk:]
        return out

    _shifts = [-2, -1, 0, 1, 2]
    _rows = []
    for _s in _shifts:
        _x_s = _shifted(_base, _s)
        _e = float(np.sqrt(np.sum((_base - _x_s) ** 2)))
        _d = float(soft_dtw(_base.reshape(-1, 1), _x_s.reshape(-1, 1), gamma=1.0))
        _rows.append((_s, _e, _d))

    _fig, (_ax_curves, _ax_table) = plt.subplots(1, 2, figsize=(11, 4))
    for _s in _shifts:
        _ax_curves.plot(weeks_onset, _shifted(_base, _s), label=f"shift {_s:+d}", alpha=0.8)
    _ax_curves.set_title("One centroid, time-shifted")
    _ax_curves.set_xlabel("ISO week")
    _ax_curves.legend(fontsize=8)
    _ax_curves.grid(alpha=0.3)

    _ax_table.axis("off")
    _txt = "shift | Euclidean | soft-DTW\n" + "-" * 30 + "\n"
    for _s, _e, _d in _rows:
        _txt += f"{_s:+d}    | {_e:8.3f}  | {_d:8.3f}\n"
    _ax_table.text(0.05, 0.5, _txt, family="monospace", fontsize=11)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.mpl.interactive(_fig),
            mo.md(
                "**Read the table:** Euclidean distance grows roughly quadratically "
                "with the shift, soft-DTW grows much more slowly (it 'forgives' the "
                "shift). For *this* problem we want the Euclidean behaviour."
            ),
        ]
    )
    return


# ============================================================================
# 5. K-means with soft-DTW (tslearn)
# ============================================================================
@app.cell
def _softdtw_md(mo):
    mo.md(
        r"""
## 3. K-means with soft-DTW (tslearn)

### Math
Classical DTW between two series $x, y$ of length $L$ finds the alignment path
$\pi$ minimising $\sum_{(t, t') \in \pi} (x_t - y_{t'})^2$ subject to boundary,
monotonicity and continuity constraints. The minimum is taken over an exponential
set of paths but computed in $\mathcal{O}(L^2)$ with dynamic programming.

**Soft-DTW** replaces the hard min by a smooth approximation
$$
\min{}^{\gamma}(a) \;=\; -\gamma \log \sum_i \exp(-a_i / \gamma),
$$
with $\gamma > 0$ controlling the smoothing. As $\gamma \to 0$ this recovers
ordinary DTW; for $\gamma > 0$ it is *differentiable* in the inputs, which is
what lets `tslearn` optimise centroids by gradient descent.

### Assumptions vs this problem
Soft-DTW K-means makes the same Euclidean-spherical-cluster assumption as plain
K-means, but additionally **assumes you want time-warp invariance**. Per the
discussion above, that assumption *does not hold here* — onset timing is a
feature, not noise. We include the model anyway because showing the failure mode
is informative.

### Q2. Why is the efficiency so bad?
Per-iteration cost: each of the $n$ samples must compare to each of $K$
centroids, and each comparison runs the DTW DP, which is $\mathcal{O}(L^2)$. So
one iteration is $\mathcal{O}(n K L^2)$ versus Euclidean K-means'
$\mathcal{O}(n K L)$.

Worse, **centroid updates** in soft-DTW K-means are not closed-form. Each update
solves an inner optimisation (`tslearn` uses L-BFGS) that itself evaluates the
soft-DTW gradient — another factor of $L^2$ per evaluation, with multiple
line-search evaluations per inner iteration.

Concretely for our grid: $n \approx 3000$ cells, $L \approx 32$ weeks, $K = 5$.
Plain K-means converges in seconds. Soft-DTW K-means with the same $K$ takes
minutes per restart. We therefore subsample (`softdtw_subsample` cells) for the
demo.
"""
    )
    return


@app.cell
def _softdtw_run(
    X_zscore,
    _time,
    mo,
    np,
    selected_k,
    softdtw_gamma,
    softdtw_subsample,
):
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

    _rng = np.random.default_rng(1)
    _n_full = X_zscore.shape[0]
    _n_sub = min(int(softdtw_subsample.value), _n_full)
    sub_idx = _rng.choice(_n_full, size=_n_sub, replace=False)
    X_sub = X_zscore[sub_idx]
    X_sub_ts = to_time_series_dataset(X_sub)

    K_sd = int(selected_k.value)

    _t0 = _time.time()
    _ = TimeSeriesKMeans(
        n_clusters=K_sd, metric="euclidean", random_state=0, n_init=2
    ).fit(X_sub_ts)
    t_eucl = _time.time() - _t0

    _t0 = _time.time()
    _km_sdtw = TimeSeriesKMeans(
        n_clusters=K_sd,
        metric="softdtw",
        metric_params={"gamma": float(softdtw_gamma.value)},
        random_state=0,
        n_init=1,
        max_iter=20,
        max_iter_barycenter=20,
        n_jobs=-1,
    ).fit(X_sub_ts)
    t_sdtw = _time.time() - _t0

    sdtw_labels = _km_sdtw.labels_
    sdtw_centroids = _km_sdtw.cluster_centers_.squeeze(-1)

    mo.md(
        f"**Wall-clock comparison** on {_n_sub} cells, K={K_sd}, L={X_sub.shape[1]}:\n\n"
        f"- Euclidean K-means (tslearn, n_init=2): **{t_eucl:.2f}s**\n"
        f"- Soft-DTW K-means (n_init=1, max_iter=20, n_jobs=-1, γ={float(softdtw_gamma.value)}): "
        f"**{t_sdtw:.2f}s**\n\n"
        f"Ratio: soft-DTW is **{t_sdtw / max(t_eucl, 1e-6):.1f}×** slower on the same subset, "
        f"and the gap grows with $L$."
    )
    return X_sub, X_sub_ts, sdtw_centroids, sdtw_labels, sub_idx, t_eucl, t_sdtw


@app.cell
def _softdtw_silhouette(X_sub_ts, mo, sdtw_labels, ts_silhouette_score):
    _sil_sdtw = float(
        ts_silhouette_score(X_sub_ts, sdtw_labels, metric="softdtw")
    )
    _sil_eucl = float(
        ts_silhouette_score(X_sub_ts, sdtw_labels, metric="euclidean")
    )
    mo.md(
        f"Silhouette of the soft-DTW partition (using metric='softdtw'): "
        f"**{_sil_sdtw:.3f}**\n\n"
        f"Same partition scored under Euclidean: **{_sil_eucl:.3f}**\n\n"
        "These are not directly comparable across metrics — what matters is "
        "comparing soft-DTW's score to soft-DTW scores at *other* K values, which "
        "you can do by re-running this section with a different `selected_k`."
    )
    return


@app.cell
def _softdtw_centroids_plot(mo, plt, sdtw_centroids, weeks_onset):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _cmap = plt.get_cmap("tab10", sdtw_centroids.shape[0])
    for _i in range(sdtw_centroids.shape[0]):
        _ax.plot(weeks_onset, sdtw_centroids[_i], color=_cmap(_i), label=f"C{_i}")
    _ax.set_title("Soft-DTW K-means centroids (DBA-style barycenters, z-score units)")
    _ax.set_xlabel("ISO week")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _softdtw_takeaway(mo):
    mo.md(
        r"""
**Takeaway on soft-DTW for this problem.**

- It is genuinely slower by an order of magnitude or more, and the constant
  factor matters because we want to scan $K$ to pick the best.
- Its inductive bias (warp invariance) actively *removes the signal* we care
  about — onset timing differences between regions.
- Its centroids (DBA / soft-DBA barycenters) are also not interpretable as "the
  average week-by-week curve" the way Euclidean centroids are.

**Verdict: do not use soft-DTW for the rainfall regime problem.** Use it when
you genuinely have time-warping nuisance (e.g. comparing daily growth curves
across different planting dates).
"""
    )
    return


# ============================================================================
# 6. EM Gaussian Mixture
# ============================================================================
@app.cell
def _gmm_md(mo):
    mo.md(
        r"""
## 4. EM Gaussian Mixture Model

### Math
We model each cell's profile $x_i$ as a draw from a mixture
$$
p(x) \;=\; \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k),
$$
with mixing weights $\pi_k$ summing to 1. Latent variable $z_i \in \{1, \ldots, K\}$
is the unknown component. Expectation-Maximisation alternates:

- **E-step:** compute the posterior responsibility
  $\gamma_{ik} = p(z_i = k \mid x_i, \theta) \propto \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$.
- **M-step:** update $\pi_k = \tfrac{1}{n}\sum_i \gamma_{ik}$,
  $\mu_k = \tfrac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$, and the analogous
  weighted-covariance update for $\Sigma_k$.

Converges to a local maximum of the log-likelihood; we use `n_init=5`.

### Assumptions vs this problem
- Each cluster's profiles are **multivariate Gaussian around a centroid.** With
  $L \approx 32$ weeks, a `full` covariance has $\sim 528$ parameters *per
  component* — easy to overfit on 3000 cells. We use `covariance_type="diag"`
  (one variance per week per component, ~32 parameters) which is the right bias
  for this scale.
- **Soft assignments.** Unlike K-means' hard partition, GMM gives a probability
  distribution over clusters per cell. This is genuinely useful for rainfall
  regimes: there are obvious transition zones between the unimodal kiremt-only
  highlands and the bimodal southern lowlands, and a hard label there is
  misleading.

### Q3c. Is EM/GMM reasonable here?
Yes, with caveats:

- The Gaussian assumption per *feature* (a single week) is plausible after
  z-scoring — we're modelling a centred, scaled vector of weekly anomalies.
- The diagonal covariance is conservative; we are *not* modelling week-to-week
  correlation. That's a deliberate underfit to keep the model identifiable on
  ~3000 samples.
- The headline gain over K-means is the **soft membership map** — see the
  "ambiguity" plot below.
"""
    )
    return


@app.cell
def _gmm_fit(GaussianMixture, X_zscore, mo, np, selected_k):
    K_gmm = int(selected_k.value)
    _gmm = GaussianMixture(
        n_components=K_gmm,
        covariance_type="diag",
        random_state=0,
        n_init=5,
        max_iter=300,
    )
    gmm_labels = _gmm.fit_predict(X_zscore)
    gmm_probs = _gmm.predict_proba(X_zscore)

    cluster_ids_gmm = np.unique(gmm_labels)
    centroids_gmm = _gmm.means_
    mo.plain_text(
        f"Fitted GMM (K={K_gmm}, diag). Final log-likelihood per sample: "
        f"{_gmm.score(X_zscore):.3f}; converged: {_gmm.converged_}."
    )
    return K_gmm, centroids_gmm, cluster_ids_gmm, gmm_labels, gmm_probs


@app.cell
def _gmm_bic(GaussianMixture, X_zscore, elbow_max_k, mo, np, plt):
    _K_grid = list(range(2, int(elbow_max_k.value) + 1))
    _bics = []
    _aics = []
    for _k in _K_grid:
        _g = GaussianMixture(
            n_components=_k,
            covariance_type="diag",
            random_state=0,
            n_init=2,
            max_iter=200,
        ).fit(X_zscore)
        _bics.append(_g.bic(X_zscore))
        _aics.append(_g.aic(X_zscore))

    _best_bic_k = _K_grid[int(np.argmin(_bics))]
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_K_grid, _bics, marker="o", label="BIC (lower is better)")
    _ax.plot(_K_grid, _aics, marker="s", label="AIC")
    _ax.axvline(_best_bic_k, color="C0", linestyle="--", alpha=0.5,
                label=f"BIC argmin K={_best_bic_k}")
    _ax.set_xlabel("K")
    _ax.set_ylabel("Information criterion")
    _ax.set_title("GMM model selection")
    _ax.legend()
    _ax.grid(alpha=0.3)
    mo.vstack(
        [
            mo.mpl.interactive(_fig),
            mo.md(
                "**BIC penalises model complexity** (number of free parameters), so "
                "it usually selects a smaller K than AIC. Either is a more "
                "principled answer than the K-means elbow."
            ),
        ]
    )
    return


@app.cell
def _gmm_plots(
    K_gmm,
    centroids_gmm,
    cluster_ids_gmm,
    ethiopia_geom,
    gmm_labels,
    gmm_probs,
    lat,
    lon,
    mo,
    mpl,
    n_lat,
    n_lon,
    np,
    plt,
    valid_mask,
    weeks_onset,
):
    _flat = np.full(n_lat * n_lon, np.nan)
    _flat[valid_mask] = gmm_labels
    label_grid_gmm = _flat.reshape(n_lat, n_lon)

    _cmap = plt.get_cmap("tab10", len(cluster_ids_gmm))
    _norm = mpl.colors.BoundaryNorm(
        np.arange(-0.5, len(cluster_ids_gmm) + 0.5, 1), _cmap.N
    )

    _entropy = -(gmm_probs * np.log(gmm_probs + 1e-12)).sum(axis=1) / np.log(K_gmm)
    _ent_flat = np.full(n_lat * n_lon, np.nan)
    _ent_flat[valid_mask] = _entropy
    _ent_grid = _ent_flat.reshape(n_lat, n_lon)

    _geoms = ethiopia_geom.geoms if hasattr(ethiopia_geom, "geoms") else [ethiopia_geom]

    _fig, (_ax_lab, _ax_ent) = plt.subplots(1, 2, figsize=(13, 6))
    _ax_lab.pcolormesh(lon, lat, label_grid_gmm, shading="auto", cmap=_cmap, norm=_norm)
    for _g in _geoms:
        _x, _y = _g.exterior.xy
        _ax_lab.plot(_x, _y, color="black", linewidth=1.0)
        _ax_ent.plot(_x, _y, color="black", linewidth=1.0)
    _ax_lab.set_aspect("equal")
    _ax_lab.set_title(f"GMM hard assignment (argmax), K={K_gmm}")

    _mesh = _ax_ent.pcolormesh(lon, lat, _ent_grid, shading="auto", cmap="magma", vmin=0, vmax=1)
    _ax_ent.set_aspect("equal")
    _ax_ent.set_title("Membership entropy (0 = confident, 1 = uniform)")
    _fig.colorbar(_mesh, ax=_ax_ent, label="normalised entropy")

    _fig_c, _ax_c = plt.subplots(figsize=(9, 4))
    for _i, _c in enumerate(cluster_ids_gmm):
        _ax_c.plot(weeks_onset, centroids_gmm[_i], color=_cmap(_i), label=f"C{_c}")
    _ax_c.set_title("GMM component means (z-score units)")
    _ax_c.set_xlabel("ISO week")
    _ax_c.legend(fontsize=8)
    _ax_c.grid(alpha=0.3)

    mo.vstack([mo.mpl.interactive(_fig), mo.mpl.interactive(_fig_c)])
    return (label_grid_gmm,)


# ============================================================================
# 7. Q4 — measuring performance and Q5 — comparing models
# ============================================================================
@app.cell
def _perf_md(mo):
    mo.md(
        r"""
## 5. How to measure performance — and how to compare models

We have no labels. So "performance" splits into two kinds of question:

**Internal validity** (model talking to itself):

- K-means: inertia (used in elbow), silhouette (already shown).
- GMM: log-likelihood, BIC, AIC (already shown).

**External validity** (does the partition behave like a *useful* regime map?). For
this problem the operational tests are:

- **Centroid faithfulness.** For each cell, how big is the gap between its own
  seasonal curve and the centroid it's assigned to? We measure per-cell RMSE in
  mm/week against the *raw* cluster mean, then look at its distribution per
  cluster — fat tails mean a cluster is too coarse.
- **Onset/peak agreement.** For each cell compute the week of the seasonal peak;
  for each cluster centroid do the same. Cells should mostly agree with their
  centroid's peak week to within a small tolerance. This is the *direct*
  operational metric for the goal stated at the top.

Both of these are computed below for K-means (Euclidean) and GMM. We omit
soft-DTW from the headline comparison because we already concluded its inductive
bias is wrong here — but the same metrics could be applied.
"""
    )
    return


@app.cell
def _faithfulness(
    X_raw,
    gmm_labels,
    km_labels,
    mo,
    np,
    plt,
):
    def _faith(X, labels):
        _ids = np.unique(labels)
        _means = np.array([X[labels == c].mean(axis=0) for c in _ids])
        _rmse = np.sqrt(np.mean((X - _means[labels]) ** 2, axis=1))
        _peak_cell = X.argmax(axis=1)
        _peak_centroid = _means[labels].argmax(axis=1)
        _peak_diff = np.abs(_peak_cell - _peak_centroid)
        return _rmse, _peak_diff

    _rmse_km, _peak_km = _faith(X_raw, km_labels)
    _rmse_gmm, _peak_gmm = _faith(X_raw, gmm_labels)

    _fig, (_ax_r, _ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax_r.boxplot(
        [_rmse_km, _rmse_gmm],
        labels=["K-means (Eucl.)", "GMM"],
        showfliers=False,
    )
    _ax_r.set_ylabel("Per-cell RMSE vs cluster mean (mm/week)")
    _ax_r.set_title("Centroid faithfulness")
    _ax_r.grid(alpha=0.3)

    _bins = np.arange(0, max(_peak_km.max(), _peak_gmm.max()) + 2) - 0.5
    _ax_p.hist(
        [_peak_km, _peak_gmm],
        bins=_bins,
        label=["K-means", "GMM"],
        density=True,
    )
    _ax_p.set_xlabel("|peak week − centroid peak week|")
    _ax_p.set_ylabel("Density")
    _ax_p.set_title("Onset/peak-week agreement")
    _ax_p.legend()
    _ax_p.grid(alpha=0.3)
    _fig.tight_layout()

    _summary = (
        f"**K-means** — median per-cell RMSE: {np.median(_rmse_km):.2f} mm/wk, "
        f"95th: {np.quantile(_rmse_km, 0.95):.2f}; "
        f"share of cells with peak-week mismatch ≤ 1: "
        f"{(_peak_km <= 1).mean():.1%}\n\n"
        f"**GMM** — median per-cell RMSE: {np.median(_rmse_gmm):.2f} mm/wk, "
        f"95th: {np.quantile(_rmse_gmm, 0.95):.2f}; "
        f"share of cells with peak-week mismatch ≤ 1: "
        f"{(_peak_gmm <= 1).mean():.1%}"
    )
    mo.vstack([mo.mpl.interactive(_fig), mo.md(_summary)])
    return


@app.cell
def _side_by_side(
    ethiopia_geom,
    label_grid_gmm,
    label_grid_km,
    lat,
    lon,
    mo,
    mpl,
    np,
    plt,
):
    from sklearn.metrics import adjusted_rand_score

    _flat_km = label_grid_km.ravel()
    _flat_gmm = label_grid_gmm.ravel()
    _common = ~(np.isnan(_flat_km) | np.isnan(_flat_gmm))
    _ari = adjusted_rand_score(
        _flat_km[_common].astype(int), _flat_gmm[_common].astype(int)
    )

    _geoms = ethiopia_geom.geoms if hasattr(ethiopia_geom, "geoms") else [ethiopia_geom]

    _n_clusters = (
        max(int(np.nanmax(label_grid_km)), int(np.nanmax(label_grid_gmm))) + 1
    )
    _cmap = plt.get_cmap("tab10", _n_clusters)
    _norm = mpl.colors.BoundaryNorm(np.arange(-0.5, _n_clusters + 0.5, 1), _cmap.N)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for _ax, _grid, _title in [
        (_ax1, label_grid_km, "K-means (Euclidean)"),
        (_ax2, label_grid_gmm, "GMM (diag, EM)"),
    ]:
        _ax.pcolormesh(lon, lat, _grid, shading="auto", cmap=_cmap, norm=_norm)
        for _g in _geoms:
            _x, _y = _g.exterior.xy
            _ax.plot(_x, _y, color="black", linewidth=1.0)
        _ax.set_aspect("equal")
        _ax.set_title(_title)

    _fig.suptitle(f"Cluster maps side-by-side — Adjusted Rand Index = {_ari:.3f}")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.mpl.interactive(_fig),
            mo.md(
                "**ARI** measures partition similarity, ignoring label permutations. "
                "1.0 = identical, 0 = no agreement beyond chance. A high ARI between "
                "K-means and GMM here is reassuring: two algorithms with very "
                "different inductive biases agree on the gross regime structure."
            ),
        ]
    )
    return


# ============================================================================
# 8. Conclusion
# ============================================================================
@app.cell
def _conclusion(mo):
    mo.md(
        r"""
## 6. Conclusions

- **Best data prep** for this problem is **per-cell z-scoring**. Raw mm makes
  K-means cluster wet vs dry instead of clustering *shape*; L1 normalisation
  works too but is harder to interpret because the units are shares.
- **Best metric for picking $K$** is silhouette for K-means and BIC for GMM.
  Both are bounded, principled, and put the answer where domain knowledge can
  argue with it (typically 4–6 regimes for Ethiopia).
- **Soft-DTW is the wrong tool** for the rainfall regime problem. Its
  warp-invariance is a virtue elsewhere but a vice here, because onset timing
  differences between cells are *the* signal we want to surface. On top of that
  its $\mathcal{O}(L^2)$ per-pair cost makes K-scans expensive.
- **GMM adds value beyond K-means** through its membership entropy map, which
  directly highlights the transitional zones between unimodal and bimodal
  regimes — useful for downstream products that need to handle uncertainty.
- **The two recommended methods (K-means Euclidean and GMM-diag) agree well on
  the gross structure** (high ARI) while disagreeing in interesting ways at the
  boundaries — exactly the right kind of ensemble disagreement.

**Recommended deliverable:** publish the K-means partition as the canonical
regime map, and publish the GMM membership entropy as a companion uncertainty
layer. Use silhouette + BIC + the operational peak-week mismatch metric to set
$K$.
"""
    )
    return


if __name__ == "__main__":
    app.run()
