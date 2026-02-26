import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from pathlib import Path
    import matplotlib.pyplot as plt
    import cartopy.io.shapereader as shpreader
    from shapely import contains_xy

    return Path, contains_xy, mo, np, plt, shpreader, xr


@app.cell
def _(mo):
    mo.md("# Ethiopia Seasonal Mask (CHIRPS v3)")

    data_dir = mo.ui.text(
        value="data/chirps_v3_monthly", label="CHIRPS directory"
    )
    output_dir = mo.ui.text(value="data", label="Output directory")
    start_year = mo.ui.number(value=1998, label="Start year")
    end_year = mo.ui.number(value=2013, label="End year")
    var_name = mo.ui.text(value="precip", label="Precip variable")
    season_choice = mo.ui.dropdown(
        options=["MAM", "JJAS", "OND"],
        value="JJAS",
        label="Season",
    )
    mask_mode = mo.ui.dropdown(
        options=[
            "Rainfall in season >= threshold (mm)",
            "Majority of annual rainfall occurs in season",
        ],
        value="Rainfall in season >= threshold (mm)",
        label="Mask logic",
    )
    threshold_mm = mo.ui.number(value=100.0, label="Threshold total (mm)")
    majority_fraction = mo.ui.number(value=0.5, label="Majority threshold (0-1)")

    mo.vstack(
        [
            data_dir,
            output_dir,
            mo.hstack([start_year, end_year]),
            var_name,
            season_choice,
            mask_mode,
            mo.hstack([threshold_mm, majority_fraction]),
        ]
    )
    return (
        data_dir,
        end_year,
        majority_fraction,
        mask_mode,
        season_choice,
        start_year,
        threshold_mm,
        var_name,
    )


@app.cell
def _(Path, data_dir, end_year, mo, start_year):
    years = list(range(int(start_year.value), int(end_year.value) + 1))
    files = [Path(data_dir.value) / f"chirps-v3.0.{y}.monthly.nc" for y in years]
    existing_files = [str(f) for f in files if f.exists()]
    missing_files = [str(f) for f in files if not f.exists()]

    mo.vstack(
        [
            mo.md(f"Found **{len(existing_files)}** files, missing **{len(missing_files)}**."),
            mo.plain_text("\n".join(missing_files[:12]) if missing_files else "No missing files."),
        ]
    )
    return existing_files, years


@app.cell
def _(existing_files, mo, xr):
    if not existing_files:
        raise ValueError("No CHIRPS files found in selected range/path.")

    ds_chirps = xr.open_mfdataset(
        existing_files,
        combine="by_coords",
        parallel=True,
        engine="netcdf4",
        chunks={"time": 12},
    )

    rename_map = {}
    for old, new in [
        ("TIME", "time"),
        ("LATITUDE", "lat"),
        ("LONGITUDE", "lon"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ]:
        if old in ds_chirps.dims or old in ds_chirps.coords:
            rename_map[old] = new
    if rename_map:
        ds_chirps = ds_chirps.rename(rename_map)

    mo.plain_text(
        "Dims: "
        + str(ds_chirps.dims)
        + "\nVars: "
        + ", ".join(list(ds_chirps.data_vars))
    )
    return (ds_chirps,)


@app.cell
def _(
    ds_chirps,
    majority_fraction,
    mask_mode,
    mo,
    np,
    season_choice,
    threshold_mm,
    var_name,
    years,
):
    if var_name.value not in ds_chirps.data_vars:
        raise KeyError(
            f"Variable '{var_name.value}' not in dataset. "
            f"Available: {list(ds_chirps.data_vars)}"
        )
    if not (0.0 <= float(majority_fraction.value) <= 1.0):
        raise ValueError("Majority threshold must be between 0 and 1.")

    ds_years = ds_chirps.sel(time=ds_chirps["time.year"].isin(np.array(years)))
    da = ds_years[var_name.value]

    season_to_months = {"MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "OND": [10, 11, 12]}
    selected_months = season_to_months[season_choice.value]
    season_da = da.sel(time=da["time.month"].isin(selected_months))
    seasonal_total_clim = season_da.groupby("time.year").sum("time").mean("year")
    annual_total_clim = da.groupby("time.year").sum("time").mean("year")
    seasonal_fraction = seasonal_total_clim / annual_total_clim

    if mask_mode.value == "Rainfall in season >= threshold (mm)":
        seasonal_mask = seasonal_total_clim >= float(threshold_mm.value)
        logic_text = (
            f"Logic: climatological seasonal total for months {selected_months} "
            f">= {float(threshold_mm.value):.1f} mm."
        )
        logic_slug = f"total-ge-{int(round(float(threshold_mm.value)))}mm"
    else:
        seasonal_mask = seasonal_fraction >= float(majority_fraction.value)
        logic_text = (
            f"Logic: seasonal share for months {selected_months} "
            f">= {float(majority_fraction.value):.2f} of annual rainfall."
        )
        logic_slug = f"majority-ge-{str(float(majority_fraction.value)).replace('.', 'p')}"

    season_slug = season_choice.value.lower()
    filename_slug = (
        f"chirps_v3_ethiopia_mask_{season_slug}_{logic_slug}_{years[0]}-{years[-1]}.nc"
    )

    mo.md(f"Seasonal mask computed. {logic_text}")
    return logic_text, seasonal_mask, selected_months


@app.cell
def _(contains_xy, np, seasonal_mask, shpreader):
    ethiopia_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    ethiopia_geom = None
    for country in shpreader.Reader(ethiopia_shp).records():
        if country.attributes["NAME"] == "Ethiopia":
            ethiopia_geom = country.geometry
            break
    if ethiopia_geom is None:
        raise RuntimeError("Could not find Ethiopia geometry in Natural Earth data.")

    lons, lats = np.meshgrid(seasonal_mask.lon.values, seasonal_mask.lat.values)
    mask_flat = contains_xy(ethiopia_geom, lons.ravel(), lats.ravel())
    ethiopia_grid_mask = mask_flat.reshape(lons.shape)
    seasonal_mask_ethiopia = seasonal_mask.where(ethiopia_grid_mask)
    return ethiopia_geom, seasonal_mask_ethiopia


@app.cell
def _(ds_chirps, np, season_choice, seasonal_mask_ethiopia, var_name, years):
    def _():
        season_to_months = {"MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "OND": [10, 11, 12]}
        selected_months = season_to_months[season_choice.value]

        ds_years = ds_chirps.sel(time=ds_chirps["time.year"].isin(np.array(years)))
        da = ds_years[var_name.value]
        season_da = da.sel(time=da["time.month"].isin(selected_months))

        # CHIRPS monthly files are totals, so convert to mm/day before climatological mean.
        daily_mean_precip = (season_da / season_da["time"].dt.days_in_month).mean("time")

        # Keep precip only where mask is active (inside Ethiopia + logic=True).
        mask_true = seasonal_mask_ethiopia.fillna(0) > 0
        daily_mean_precip_masked = daily_mean_precip.where(mask_true)
        return daily_mean_precip_masked


    daily_mean_precip_masked = _()
    return (daily_mean_precip_masked,)


@app.cell
def _(daily_mean_precip_masked, ethiopia_geom, mo, plt, season_choice, years):
    fig, ax = plt.subplots(figsize=(8, 6))
    daily_mean_precip_masked.plot(
        ax=ax,
        cmap="Blues",
        robust=True,
        cbar_kwargs={"label": "Daily mean precipitation (mm/day)"},
    )

    # Overlay Ethiopia boundary.
    geoms = ethiopia_geom.geoms if hasattr(ethiopia_geom, "geoms") else [ethiopia_geom]
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.plot(x, y, color="black", linewidth=1.2)

    # Zoom to Ethiopia by default.
    minx, miny, maxx, maxy = ethiopia_geom.bounds
    pad = 1.0
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    ax.set_title(
        f"Masked Daily Mean Precipitation ({season_choice.value}, {years[0]}-{years[-1]})"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(
    daily_mean_precip_masked,
    majority_fraction,
    mask_mode,
    season_choice,
    seasonal_mask_ethiopia,
    selected_months,
    threshold_mm,
    years,
):
    print("years:", years[0], years[-1], "n_years:", len(years))
    print("season months:", selected_months if "selected_months" in globals() else season_choice.value)
    print("mask mode:", mask_mode)
    print("threshold_mm:", threshold_mm, "majority_fraction:", majority_fraction)

    print("mask true count:", int((seasonal_mask_ethiopia.fillna(0) > 0).sum()))
    print("daily mean min/max:", float(daily_mean_precip_masked.min(skipna=True)),
          float(daily_mean_precip_masked.max(skipna=True)))
    print("daily mean area mean:", float(daily_mean_precip_masked.mean(skipna=True)))

    return


@app.cell
def _(Path, logic_text, mo, season_choice, seasonal_mask_ethiopia, xr):
    seasonal_mask_ethiopia_ds = xr.Dataset(
        {"seasonal_mask": (["lat", "lon"], seasonal_mask_ethiopia.astype("float32").values)},
        coords={"lat": seasonal_mask_ethiopia.lat, "lon": seasonal_mask_ethiopia.lon},
    )

    seasonal_mask_ethiopia_ds["seasonal_mask"].attrs["long_name"] = (
        f"{season_choice.value} seasonal mask for Ethiopia"
    )
    seasonal_mask_ethiopia_ds["seasonal_mask"].attrs["description"] = (
        "Binary mask where 1 indicates cells meeting selected logic within Ethiopia boundaries, "
        "0 or NaN elsewhere."
    )
    seasonal_mask_ethiopia_ds.attrs["source"] = "Derived from CHIRPS v3 seasonal analysis"
    seasonal_mask_ethiopia_ds.attrs["resolution"] = "0.25 degrees"
    seasonal_mask_ethiopia_ds.attrs["mask_logic"] = logic_text

    out = "data/data.nc"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    seasonal_mask_ethiopia_ds.to_netcdf(out)
    mo.md(f"Saved mask to `{out}`")
    return


if __name__ == "__main__":
    app.run()
