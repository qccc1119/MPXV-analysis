# MPXV-analysis
# PR2 bias analysis and visualization.

This script calculates the two coordinates used in a parity rule 2 (PR2) plot:

    x = G3 / (G3 + C3)
    y = A3 / (A3 + T3)

The dashed reference lines at x = 0.5 and y = 0.5 represent parity between
G3/C3 and A3/T3, respectively.

Species, order, T3s, C3s, A3s, G3s

The aliases "名称" and "类别" are accepted for "Species" and "order".
The input file may be an Excel workbook (.xlsx/.xls) or a CSV file.

python pr2_analysis.py \
    --input core_peripheral_agct.xlsx \
    --output-dir results

numpy, pandas, matplotlib, openpyxl (for .xlsx input)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("Species", "order", "T3s", "C3s", "A3s", "G3s")
NUCLEOTIDE_COLUMNS = ("T3s", "C3s", "A3s", "G3s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate PR2 bias coordinates and generate a PR2 plot."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input .xlsx, .xls, or .csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pr2_results"),
        help="Directory for output files (default: pr2_results).",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help=(
            "Excel sheet name or zero-based sheet index (default: 0). "
            "Ignored for CSV input."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="PR2_Plot_Core_Peripheral",
        help="Output filename prefix (default: PR2_Plot_Core_Peripheral).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    """Configure fonts and publication-friendly vector output."""
    for font_name in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_path = fm.findfont(font_name, fallback_to_default=False)
            if font_path:
                fm.fontManager.addfont(font_path)
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [font_name]
                break
        except ValueError:
            continue

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )


def parse_sheet_argument(sheet: object) -> object:
    """Convert a numeric sheet argument to an integer index."""
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def read_input(input_path: Path, sheet: object = 0) -> pd.DataFrame:
    """Read an Excel or CSV input table."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, sheet_name=parse_sheet_argument(sheet))

    raise ValueError(
        f"Unsupported input format '{suffix}'. Use .xlsx, .xls, or .csv."
    )


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and normalize supported Chinese aliases."""
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]

    aliases = {}
    if "名称" in data.columns and "Species" not in data.columns:
        aliases["名称"] = "Species"
    if "类别" in data.columns and "order" not in data.columns:
        aliases["类别"] = "order"

    return data.rename(columns=aliases)


def validate_and_prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields, calculate PR2 coordinates, and add groups."""
    data = normalize_columns(data)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    for column in NUCLEOTIDE_COLUMNS:
        original = data[column]
        data[column] = pd.to_numeric(original, errors="coerce")
        newly_missing = original.notna() & data[column].isna()
        if newly_missing.any():
            examples = original[newly_missing].astype(str).head(3).tolist()
            warnings.warn(
                f"Column '{column}' contains non-numeric value(s), which were "
                f"converted to missing values. Examples: {examples}",
                stacklevel=2,
            )

    at_denominator = data["A3s"] + data["T3s"]
    gc_denominator = data["G3s"] + data["C3s"]

    data["AT_bias"] = np.where(
        at_denominator != 0, data["A3s"] / at_denominator, np.nan
    )
    data["GC_bias"] = np.where(
        gc_denominator != 0, data["G3s"] / gc_denominator, np.nan
    )

    order_text = data["order"].fillna("").astype(str).str.strip().str.lower()
    species_text = data["Species"].fillna("").astype(str).str.lower()

    # Prefer explicit group labels in "order". If they are absent, look for the
    # labels in "Species"; all remaining rows are treated as host species.
    explicit_groups = order_text.str.contains("core|peripheral", regex=True).any()
    grouping_text = order_text if explicit_groups else species_text
    data["Group"] = np.select(
        [
            grouping_text.str.contains("core", regex=False),
            grouping_text.str.contains("peripheral", regex=False),
        ],
        ["Core", "Peripheral"],
        default="Host",
    )

    valid_coordinates = (
        data["AT_bias"].between(0, 1, inclusive="both")
        & data["GC_bias"].between(0, 1, inclusive="both")
    )
    invalid_count = int((~valid_coordinates).sum())
    if invalid_count:
        warnings.warn(
            f"{invalid_count} row(s) have missing or invalid PR2 coordinates "
            "and will not appear in the plot.",
            stacklevel=2,
        )

    return data


def padded_limits(values: pd.Series, padding: float = 0.04) -> tuple[float, float]:
    """Return rounded plot limits that include both the data and parity line."""
    finite = values[np.isfinite(values)].to_numpy(dtype=float)
    if finite.size == 0:
        return 0.0, 1.0

    lower = max(0.0, np.floor((finite.min() - padding) * 20) / 20)
    upper = min(1.0, np.ceil((finite.max() + padding) * 20) / 20)
    lower = min(lower, 0.42)
    upper = max(upper, 0.58)

    if np.isclose(lower, upper):
        lower, upper = max(0.0, lower - 0.1), min(1.0, upper + 0.1)
    return float(lower), float(upper)


def make_pr2_plot(
    data: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    dpi: int = 300,
) -> None:
    """Create and save the PR2 scatter plot."""
    valid = data.dropna(subset=["GC_bias", "AT_bias"]).copy()
    valid = valid[
        valid["GC_bias"].between(0, 1, inclusive="both")
        & valid["AT_bias"].between(0, 1, inclusive="both")
    ]
    if valid.empty:
        raise ValueError("No valid PR2 coordinates are available for plotting.")

    colors = {
        "Host": "#A8ADB4",
        "Peripheral": "#0072B5",
        "Core": "#BC3C29",
    }
    sizes = {"Host": 42, "Peripheral": 58, "Core": 58}
    alpha = {"Host": 0.78, "Peripheral": 0.86, "Core": 0.86}
    labels = {
        "Host": "Host species",
        "Peripheral": "Peripheral",
        "Core": "Core",
    }

    figure, axis = plt.subplots(figsize=(6.8, 6.4), dpi=dpi)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    axis.axvline(
        0.5,
        color="#6F747A",
        linestyle=(0, (4, 3)),
        linewidth=1.05,
        zorder=1,
    )
    axis.axhline(
        0.5,
        color="#6F747A",
        linestyle=(0, (4, 3)),
        linewidth=1.05,
        zorder=1,
    )

    for z_order, group in enumerate(("Host", "Peripheral", "Core"), start=2):
        subset = valid[valid["Group"] == group]
        if subset.empty:
            continue

        legend_label = labels[group]
        if group != "Host":
            legend_label += f" (n={len(subset)})"

        axis.scatter(
            subset["GC_bias"],
            subset["AT_bias"],
            s=sizes[group],
            marker="o",
            facecolor=colors[group],
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha[group],
            label=legend_label,
            zorder=z_order,
        )

    x_min, x_max = padded_limits(valid["GC_bias"])
    y_min, y_max = padded_limits(valid["AT_bias"])
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")

    axis.grid(color="#E6E8EB", linewidth=0.8, zorder=0)
    axis.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
        length=4,
        color="#333333",
    )
    axis.set_xlabel("G3s / (G3s + C3s)", fontsize=11, fontweight="bold")
    axis.set_ylabel("A3s / (A3s + T3s)", fontsize=11, fontweight="bold")

    axis.text(
        x_min + 0.012,
        y_max - 0.018,
        "A/T bias",
        color="#555555",
        fontsize=8.8,
        va="top",
    )
    axis.text(
        x_max - 0.012,
        y_min + 0.018,
        "G/C bias",
        color="#555555",
        fontsize=8.8,
        ha="right",
        va="bottom",
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False,
        fontsize=9.5,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontstyle("normal")

    figure.subplots_adjust(left=0.13, right=0.74, bottom=0.13, top=0.96)
    figure.savefig(
        output_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    figure.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(figure)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    sheet: object = 0,
    prefix: str = "PR2_Plot_Core_Peripheral",
    dpi: int = 300,
) -> dict[str, Path]:
    """Run the complete PR2 analysis and return the output paths."""
    if dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")
    if not prefix.strip():
        raise ValueError("--prefix cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{prefix}.png"
    output_pdf = output_dir / f"{prefix}.pdf"
    output_csv = output_dir / f"{prefix}_Data.csv"

    raw_data = read_input(input_path, sheet=sheet)
    result = validate_and_prepare(raw_data)
    result.to_csv(output_csv, index=False)
    make_pr2_plot(result, output_png, output_pdf, dpi=dpi)

    counts = result["Group"].value_counts()
    print(f"Rows: {len(result)}")
    print(f"Core: {int(counts.get('Core', 0))}")
    print(f"Peripheral: {int(counts.get('Peripheral', 0))}")
    print(f"Host: {int(counts.get('Host', 0))}")
    print(f"PNG: {output_png.resolve()}")
    print(f"PDF: {output_pdf.resolve()}")
    print(f"CSV: {output_csv.resolve()}")

    return {"png": output_png, "pdf": output_pdf, "csv": output_csv}


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_plot_style()
    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        sheet=args.sheet,
        prefix=args.prefix,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
