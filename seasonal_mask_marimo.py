import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from pathlib import Path
    import cartopy.io.shapereader as shpreader
    from shapely import contains_xy
    return Path, contains_xy, mo, np, shpreader, xr


@app.cell
def __(Path, mo):
    mo.md("# Ethiopia JJAS Seasonal Mask (CHIRPS v3)")

    data_dir = mo.ui.text(
        value="data/chirps_v3_monthly", label="CHIRPS directory"
    )
    start_year = mo.ui.number(value=1998, label="Start year")
    end_year = mo.ui.number(value=2013, label="End year")
    var_name = mo.ui.text(value="precip", label="Precip variable")
    mask_mode = mo.ui.dropdown(
        options=[
            "JJAS dominates MAM and OND",
            "Seasonal precip >= threshold (mm)",
        ],
        value="JJAS dominates MAM and OND",
        label="Mask logic",
    )
    threshold_season = mo.ui.dropdown(
        options={"MAM": "MAM", "JJAS": "JJAS", "OND": "OND"},
        value="JJAS",
        label="Threshold season",
    )
    threshold_mm = mo.ui.number(value=100.0, label="Threshold (mm)")
    output_file = mo.ui.text(
        value="data/chirps_jjas_seasonal_mask_ethiopia_0p25.nc",
        label="Output NetCDF",
    )

    mo.vstack(
        [
            data_dir,
            mo.hstack([start_year, end_year]),
            var_name,
            mask_mode,
            mo.hstack([threshold_season, threshold_mm]),
            output_file,
        ]
    )
    return (
        data_dir,
        end_year,
        mask_mode,
        output_file,
        start_year,
        threshold_mm,
        threshold_season,
        var_name,
    )


@app.cell
def __(Path, data_dir, end_year, mo, start_year):
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
def __(existing_files, mo, xr):
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
    return ds_chirps


@app.cell
def __(ds_chirps, mask_mode, mo, np, threshold_mm, threshold_season, var_name, years):
    seasons = {"MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "OND": [10, 11, 12]}

    if var_name.value not in ds_chirps.data_vars:
        raise KeyError(
            f"Variable '{var_name.value}' not in dataset. "
            f"Available: {list(ds_chirps.data_vars)}"
        )

    ds_years = ds_chirps.sel(time=ds_chirps["time.year"].isin(np.array(years)))
    da = ds_years[var_name.value]
    chirps_mam = da.sel(time=da["time.month"].isin(seasons["MAM"])).mean(dim="time")
    chirps_jjas = da.sel(time=da["time.month"].isin(seasons["JJAS"])).mean(dim="time")
    chirps_ond = da.sel(time=da["time.month"].isin(seasons["OND"])).mean(dim="time")
    clim_seasonal_total = {
        "MAM": da.sel(time=da["time.month"].isin(seasons["MAM"])).groupby("time.year").sum("time").mean("year"),
        "JJAS": da.sel(time=da["time.month"].isin(seasons["JJAS"])).groupby("time.year").sum("time").mean("year"),
        "OND": da.sel(time=da["time.month"].isin(seasons["OND"])).groupby("time.year").sum("time").mean("year"),
    }

    if mask_mode.value == "JJAS dominates MAM and OND":
        seasonal_mask = (chirps_jjas > chirps_mam) & (chirps_jjas > chirps_ond)
        logic_text = "Logic: JJAS mean precip > MAM and > OND."
    else:
        s = threshold_season.value
        seasonal_mask = clim_seasonal_total[s] >= float(threshold_mm.value)
        logic_text = f"Logic: {s} climatological seasonal total >= {float(threshold_mm.value):.1f} mm."

    mo.md(f"Seasonal mask computed. {logic_text}")
    return logic_text, seasonal_mask


@app.cell
def __(contains_xy, np, seasonal_mask, shpreader):
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
    return seasonal_mask_ethiopia


@app.cell
def __(Path, logic_text, mo, output_file, seasonal_mask_ethiopia, xr):
    seasonal_mask_ethiopia_ds = xr.Dataset(
        {"seasonal_mask": (["lat", "lon"], seasonal_mask_ethiopia.astype("float32").values)},
        coords={"lat": seasonal_mask_ethiopia.lat, "lon": seasonal_mask_ethiopia.lon},
    )

    seasonal_mask_ethiopia_ds["seasonal_mask"].attrs["long_name"] = (
        "JJAS-dominated seasonal mask for Ethiopia"
    )
    seasonal_mask_ethiopia_ds["seasonal_mask"].attrs["description"] = (
        "Binary mask where 1 indicates cells meeting selected logic within Ethiopia boundaries, "
        "0 or NaN elsewhere."
    )
    seasonal_mask_ethiopia_ds.attrs["source"] = "Derived from CHIRPS v3 seasonal analysis"
    seasonal_mask_ethiopia_ds.attrs["resolution"] = "0.25 degrees"
    seasonal_mask_ethiopia_ds.attrs["mask_logic"] = logic_text

    out = output_file.value
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    seasonal_mask_ethiopia_ds.to_netcdf(out)
    mo.md(f"Saved mask to `{out}`")
    return


if __name__ == "__main__":
    app.run()
