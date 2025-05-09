"""
Merge flight + crew data into **DataA.csv** and then split that file into
three roughly-equal chronological chunks (instance1.csv … instance3.csv).

Usage
-----
Just run the script from the folder that contains:

* DataA_Flight.csv
* DataA_Crew.csv

All output files are written to the same folder.
"""

import math
from pathlib import Path

import pandas as pd


def build_merged_dataset() -> Path:
    """Create DataA.csv by joining flight records with crew cost rates."""
    df_flt = pd.read_csv("DataA_Flight.csv")
    df_crew = pd.read_csv("DataA_Crew.csv")

    # One (Duty $/h, Pairing $/h) row per crew base
    cost_map = (
        df_crew[["Base", "DutyCostPerHour", "ParingCostPerHour"]]
        .drop_duplicates(subset="Base")
        .set_index("Base")
    )

    # Attach costs to each flight via the crew *base* (DptrStn)
    df = df_flt.merge(
        cost_map,
        left_on="DptrStn",
        right_index=True,
        how="left",
    )

    out_path = Path("DataA.csv")
    df.to_csv(out_path, index=False)
    print(f"✓ Saved merged data → {out_path.resolve()}  ({len(df):,} rows)")
    return out_path


def split_into_thirds(source_path: Path) -> None:
    """Split DataA.csv into three date-contiguous instance*.csv files."""
    df = pd.read_csv(source_path)

    # Ensure we can sort by actual dates
    df["DptrDate"] = pd.to_datetime(df["DptrDate"], format="%m/%d/%Y")

    unique_days = sorted(df["DptrDate"].dt.date.unique())
    chunk_size = math.ceil(len(unique_days) / 3)
    day_chunks = [
        unique_days[i : i + chunk_size]
        for i in range(0, len(unique_days), chunk_size)
    ]

    summary = []
    for idx, days in enumerate(day_chunks, start=1):
        chunk_df = df[df["DptrDate"].dt.date.isin(days)].copy()

        # --- force m/d/YYYY formatting (handles Unix & Windows) -------------
        try:
            chunk_df["DptrDate"] = chunk_df["DptrDate"].dt.strftime("%-m/%-d/%Y")
            date_fmt = "%-m/%-d/%Y"
        except ValueError:  # Windows has no %-m / %-d
            chunk_df["DptrDate"] = chunk_df["DptrDate"].apply(
                lambda x: f"{x.month}/{x.day}/{x.year}"
            )
            date_fmt = "%#m/%#d/%Y"
        # --------------------------------------------------------------------

        out_file = Path(f"instance{idx}.csv")
        chunk_df.to_csv(out_file, index=False)

        summary.append(
            {
                "File": out_file,
                "Rows": len(chunk_df),
                "From": days[0].strftime(date_fmt),
                "To": days[-1].strftime(date_fmt),
                "Unique Days": len(days),
            }
        )

    print("\n=== Split summary ===")
    for item in summary:
        print(
            f"{item['File']}: {item['Rows']} rows, "
            f"{item['Unique Days']} days  ({item['From']} → {item['To']})"
        )


def main() -> None:
    merged = build_merged_dataset()
    split_into_thirds(merged)


if __name__ == "__main__":
    main()