import csv, glob, os, re, sys
import numpy as np

COLS = ["ALLSKY_SFC_SW_DWN", "T2M", "T2M_RANGE", "PS", "WS10M", "RH2M", "PRECTOTCORR"]
IRR, T2M, TRNG, PS, WS, RH, PRC = range(7)

CLEAR_DAY = 5.0   # kWh/m2/day - clear day
RAINY_DAY = 1.0   # mm/day - rainy day

def save(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)


# READ SOLAR FARMS
def read_farms(path):
    raw = open(path, "rb").read()
    text = raw.decode("utf-8-sig")
    farms= []
    for r in csv.DictReader(text.splitlines()):
        for k in ("installed_mw", "lat", "lon"):
            r[k] = float(r[k])
        if r["installed_mw"] >= 1:
            farms.append(r)
    print(f"{len(farms)} farms")
    return farms


# GRID CELLS,, NASA MERRA-2 cell centers sit at multiples of 0.5lat/0.625lon, so round to the nearest center.
def cell_key(lat, lon):
    lat_center = -90 + 0.5 * round((lat + 90) / 0.5)
    lon_center = -180 + 0.625 * round((lon + 180) / 0.625)
    return f"{lat_center:.3f}_{lon_center:.3f}"


# PROCESS NASA POWER DATA
def read_power(path):
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines()

    location_match = re.search(
        r"latitude\s+([-\d.]+)\s+longitude\s+([-\d.]+)",
        "\n".join(lines[:10]),
    )
    if location_match is None:
        raise ValueError(f"Could not find latitude/longitude header in {path}")

    lat, lon = map(float, location_match.groups())

    start = next(i for i, line in enumerate(lines) if "-END HEADER-" in line) + 1
    dates = []
    rows = []
    for row in csv.DictReader(lines[start:]):
        dates.append((int(row["YEAR"]), int(row["MO"]), int(row["DY"])))
        rows.append([float(row[col]) for col in COLS])

    data = np.array(rows, dtype=float)
    data[data == -999] = np.nan

    for j in range(data.shape[1]):
        bad = np.isnan(data[:, j])
        if bad.any():
            valid_x = np.where(~bad)[0]
            bad_x = np.where(bad)[0]
            data[bad_x, j] = np.interp(bad_x, valid_x, data[valid_x, j])

    return np.array(dates), data, lat, lon

def summarize(dates, data):
    irr= data[:, IRR]
    monthly = [irr[dates[:, 1] == m].mean() for m in range(1, 13)]
    return {
        "irr_mean": irr.mean(),                          # how much sun avail on average
        "irr_std": irr.std(),                            # how steady is sun day to day
        "irr_season_gap": max(monthly) - min(monthly),   # highest monthly ave vs lowest
        "clear_day_frac": np.mean(irr >= CLEAR_DAY),     # fraction of clear days
        "temp_mean": data[:, T2M].mean(),                   # ave temp 
        "rh_mean": data[:, RH].mean(),                      # ave relative humidity
        "rain_mean": data[:, PRC].mean(),                   # ave rainfall
        "rainy_day_frac": np.mean(data[:, PRC] >= RAINY_DAY), # days with at least 1mm rain
        "wind_mean": data[:, WS].mean(),                    # ave wind speed
    }

def main(farms_csv, weather_dir):
    out_dir = "processed"
    os.makedirs(out_dir, exist_ok=True)

    farms = read_farms(farms_csv)
    cells = {}
    for farm in farms:
        cell_id = cell_key(farm["lat"], farm["lon"])
        farm["cell_key"] = cell_id

        counts_and_mw = cells.setdefault(cell_id, [0, 0.0])
        counts_and_mw[0] += 1
        counts_and_mw[1] += farm["installed_mw"]

    farm_fields = [
        "project",
        "province",
        "municipality",
        "installed_mw",
        "lat",
        "lon",
        "cell_key",
    ]
    save(
        os.path.join(out_dir, "farm_cells.csv"),
        farm_fields,
        [[farm[field] for field in farm_fields] for farm in farms],
    )

    rows = []
    names = None
    for path in sorted(glob.glob(os.path.join(weather_dir, "*.csv"))):
        dates, data, lat, lon = read_power(path)
        key = cell_key(lat, lon)
        summary = summarize(dates, data)
        names = list(summary)

        n_farms, total_mw = cells.get(key, [0, 0.0])
        rows.append([key, lat, lon, n_farms, round(total_mw, 2)] + list(summary.values()))

    save(
        os.path.join(out_dir, "site_table.csv"),
        ["cell_key", "lat", "lon", "n_farms", "total_mw"] + names,
        rows,
    )

    # z-scores: the columns have different units, so unscaled distances would be
    # dominated by the biggest numbers (for clustering and KNN)
    metrics = np.array([row[5:] for row in rows], dtype=float)
    z_scores = (metrics - metrics.mean(0)) / (metrics.std(0) + 1e-12)
    save(
        os.path.join(out_dir, "site_table_zscore.csv"),
        ["cell_key"] + names,
        [[row[0]] + z.tolist() for row, z in zip(rows, z_scores)],
    )

    print(f"{len(rows)} cells")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
