# PR2 bias analysis and visualization.


import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="PR2 plot analysis")
    parser.add_argument("--input", required=True, type=Path, help="Input Excel/CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("pr2_results"), help="Output directory")
    parser.add_argument("--prefix", default="PR2_Plot", help="Filename prefix")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")
    return parser.parse_args()

def process_data(input_path: Path) -> pd.DataFrame:

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"name": "Species", "order": "order"})


    df["GC_bias"] = df["G3s"] / (df["G3s"] + df["C3s"])
    df["AT_bias"] = df["A3s"] / (df["A3s"] + df["T3s"])


    order_str = df["order"].astype(str).str.lower()
    species_str = df["Species"].astype(str).str.lower()
    grouping = order_str if order_str.str.contains("core|peripheral").any() else species_str

    df["Group"] = np.select(
        [grouping.str.contains("core"), grouping.str.contains("peripheral")],
        ["Core", "Peripheral"],
        default="Host"
    )
    return df.dropna(subset=["GC_bias", "AT_bias"])

def draw_plot(df: pd.DataFrame, output_png: Path, dpi: int = 300):
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=dpi)


    ax.axvline(0.5, color="#6F747A", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="#6F747A", linestyle="--", linewidth=1.0)

    colors = {"Host": "#A8ADB4", "Peripheral": "#0072B5", "Core": "#BC3C29"}
    sizes = {"Host": 42, "Peripheral": 58, "Core": 58}


    for group in ("Host", "Peripheral", "Core"):
        sub = df[df["Group"] == group]
        if sub.empty:
            continue
        label = f"{group} (n={len(sub)})" if group != "Host" else "Host species"
        ax.scatter(
            sub["GC_bias"], sub["AT_bias"],
            s=sizes[group], color=colors[group],
            edgecolor="white", linewidth=0.7, alpha=0.8, label=label
        )


    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.4, 0.6)
    ax.set_aspect("equal")
    ax.grid(color="#E6E8EB", linewidth=0.8)
    ax.set_xlabel("G3s / (G3s + C3s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("A3s / (A3s + T3s)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    data = process_data(args.input)
    
    csv_out = args.output_dir / f"{args.prefix}_Data.csv"
    png_out = args.output_dir / f"{args.prefix}.png"
    
    data.to_csv(csv_out, index=False)
    draw_plot(data, png_out, dpi=args.dpi)
    print(f"saved {csv_out}，saved {png_out}")



# CLR-PCA analysis and visualization for Relative Synonymous Codon Usage (RSCU).

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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")

# Codons to exclude from CLR-PCA analysis (Stop codons + single-codon amino acids)
EXCLUDED_CODONS = {"UAA", "UAG", "UGA", "AUG", "UGG"}


GROUP_COLORS = {
    "MPXV Peripheral": "#0072B5",
    "MPXV Core": "#BC3C29",
    "Chiroptera": "#009E73",
    "Primates": "#4E79A7",
    "Rodentia": "#CC79A7",
    "Other": "#8C8C8C",
    "Virus": "#6A51A3",
}

GROUP_MARKERS = {
    "MPXV Peripheral": "D",
    "MPXV Core": "D",
    "Chiroptera": "o",
    "Primates": "o",
    "Rodentia": "o",
    "Other": "o",
    "Virus": "D",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform CLR transformation and PCA analysis on RSCU data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input Excel (.xlsx/.xls) or CSV (.csv) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save output files (default: outputs).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for generated PNG figures (default: 300).",
    )
    return parser.parse_args()


def setup_font_env() -> None:
    """Configure matplotlib fonts and publication-quality vector settings."""
    for font_name in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            path = fm.findfont(font_name, fallback_to_default=False)
            if path:
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [font_name]
                break
        except Exception:
            continue
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 0.9


def extract_codon_str(name: object) -> str:
    """Extract 3-letter codon string from column name."""
    name_str = str(name).strip()
    return name_str[-3:].upper() if len(name_str) >= 3 else name_str.upper()


def clr_transform(data: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Perform Centered Log-Ratio (CLR) transformation."""
    data_safe = np.asarray(data, dtype=float) + epsilon
    log_data = np.log(data_safe)
    return log_data - np.mean(log_data, axis=1, keepdims=True)


def classify_group(species: object, order: object) -> str:
    """Determine entity group classification based on species and order names."""
    species_text = str(species).lower()
    order_text = str(order).lower()
    if "core" in species_text:
        return "MPXV Core"
    if "peripheral" in species_text:
        return "MPXV Peripheral"
    if "virus" in order_text or "virus" in species_text:
        return "Virus"
    if "chiroptera" in order_text:
        return "Chiroptera"
    if "primates" in order_text:
        return "Primates"
    if "rodentia" in order_text:
        return "Rodentia"
    return "Other"


def pca_from_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PCA scores, loadings, and explained variance using SVD."""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    scores = u * singular_values
    denom = max(x.shape[0] - 1, 1)
    eigenvalues = singular_values**2 / denom
    explained = eigenvalues / eigenvalues.sum() * 100
    loadings = vt.T
    return scores, loadings, explained


def add_cov_ellipse(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, alpha: float = 0.07
) -> None:
    """Add a 95% confidence covariance ellipse to the plot."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = np.sqrt(5.991)  # Chi-squared critical value for 95% confidence
    width, height = 2 * scale * np.sqrt(vals)

    ax.add_patch(
        Ellipse(
            (np.mean(x), np.mean(y)),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            lw=1.2,
            alpha=alpha,
            zorder=1,
        )
    )
    ax.add_patch(
        Ellipse(
            (np.mean(x), np.mean(y)),
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            lw=1.0,
            alpha=0.55,
            zorder=2,
        )
    )


def plot_pca_panel(
    ax: plt.Axes,
    row_out: pd.DataFrame,
    scores: np.ndarray,
    explained: np.ndarray,
    show_y_label: bool = True,
    panel_label: str | None = None,
) -> None:
    """Plot a single PCA panel with scatter points and confidence ellipses."""
    # Draw confidence ellipses for major host groups
    for group in ["Chiroptera", "Primates", "Rodentia"]:
        mask = row_out["Group"].eq(group).to_numpy()
        if mask.sum() >= 3:
            add_cov_ellipse(ax, scores[mask, 0], scores[mask, 1], GROUP_COLORS[group])

    # Plot sample points
    for group in [
        "Chiroptera",
        "Primates",
        "Rodentia",
        "Other",
        "Virus",
        "MPXV Peripheral",
        "MPXV Core",
    ]:
        mask = row_out["Group"].eq(group).to_numpy()
        if not mask.any():
            continue
        is_mpxv = group.startswith("MPXV")
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=120 if is_mpxv else 52,
            marker=GROUP_MARKERS[group],
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=1.1 if is_mpxv else 0.65,
            alpha=0.95 if is_mpxv else 0.86,
            zorder=5 if is_mpxv else 3,
            label=group,
        )

    # Grid & Axes styling
    ax.axhline(0, color="#AEB4BA", lw=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.axvline(0, color="#AEB4BA", lw=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.grid(color="#E6E8EB", linewidth=0.75, zorder=0)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}% variance)" if show_y_label else "",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=8.5, width=0.8, length=4, color="#333333")

    if panel_label:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def build_legend_handles(row_out: pd.DataFrame) -> list[Line2D]:
    """Create legend handles based on present groups."""
    handles = []
    counts = row_out["Group"].value_counts().to_dict()
    for group in ["MPXV Peripheral", "MPXV Core", "Chiroptera", "Primates", "Rodentia"]:
        if group not in counts:
            continue
        label = f"{group} Genes" if group.startswith("MPXV") else f"{group} (n={counts[group]})"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=GROUP_MARKERS[group],
                linestyle="",
                markerfacecolor=GROUP_COLORS[group],
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8 if group.startswith("MPXV") else 7,
                label=label,
            )
        )
    return handles


def read_data(input_path: Path) -> pd.DataFrame:
    """Read dataset from Excel or CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported file format '{suffix}'. Use .xlsx, .xls, or .csv.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_font_env()

    # Define output file paths
    out_png = args.output_dir / "CLR_PCA_RSCU_MedicalStyle.png"
    out_pdf = args.output_dir / "CLR_PCA_RSCU_MedicalStyle.pdf"
    out_comb_png = args.output_dir / "CLR_PCA_RSCU_All_and_Top10Closest.png"
    out_comb_pdf = args.output_dir / "CLR_PCA_RSCU_All_and_Top10Closest.pdf"
    out_row_csv = args.output_dir / "CLR_PCA_RSCU_Row_Coordinates.csv"
    out_loadings_csv = args.output_dir / "CLR_PCA_RSCU_Codon_Loadings.csv"
    out_top10_csv = args.output_dir / "CLR_PCA_RSCU_Top10Closest_Row_Coordinates.csv"

    # 1. Load Data
    df = read_data(args.input)
    df.columns = [str(c).strip() for c in df.columns]
    species_col = "Species" if "Species" in df.columns else df.columns[0]
    order_col = "order" if "order" in df.columns else df.columns[1]

    # 2. Filter codons & perform CLR transformation
    codon_cols = [c for c in df.columns[2:] if extract_codon_str(c) not in EXCLUDED_CODONS]
    rscu = df[codon_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    clr = clr_transform(rscu)
    scores, loadings, explained = pca_from_matrix(clr)

    species = df[species_col].astype(str).str.strip()
    orders = df[order_col].astype(str).str.strip()
    groups = [classify_group(s, o) for s, o in zip(species, orders)]
    codon_labels = [extract_codon_str(c) for c in codon_cols]

    # 3. Save Coordinates & Loadings
    row_out = pd.DataFrame({
        "Species": species,
        "Order": orders,
        "Group": groups,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "PC3": scores[:, 2] if scores.shape[1] > 2 else np.nan,
    })
    loading_out = pd.DataFrame({
        "Codon": codon_labels,
        "PC1_loading": loadings[:, 0],
        "PC2_loading": loadings[:, 1],
        "PC3_loading": loadings[:, 2] if loadings.shape[1] > 2 else np.nan,
    })
    row_out.to_csv(out_row_csv, index=False)
    loading_out.to_csv(out_loadings_csv, index=False)

    # 4. Plot Single PCA Panel
    fig, ax = plt.subplots(figsize=(7.4, 6.7), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_pca_panel(ax, row_out, scores, explained)
    handles = build_legend_handles(row_out)

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Arial")
        text.set_fontstyle("normal")

    fig.subplots_adjust(left=0.12, right=0.72, bottom=0.13, top=0.96)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # 5. Distance Calculation: Find Top 10 Closest Hosts to MPXV
    virus_mask = row_out["Group"].str.startswith("MPXV").to_numpy()
    host_indices = np.where(~virus_mask)[0]
    virus_indices = np.where(virus_mask)[0]

    distances = np.linalg.norm(clr[host_indices, None, :] - clr[virus_indices, :], axis=2)
    mean_distance = distances.mean(axis=1)
    top_host_indices = host_indices[np.argsort(mean_distance)[:10]]
    top_indices = np.r_[virus_indices, top_host_indices]

    top_clr = clr[top_indices]
    top_scores, _, top_explained = pca_from_matrix(top_clr)
    top_row_out = row_out.iloc[top_indices].copy().reset_index(drop=True)
    top_row_out["Mean_CLR_distance_to_MPXV"] = np.r_[
        np.full(len(virus_indices), np.nan), np.sort(mean_distance)[:10]
    ]
    top_row_out["PC1_top10"] = top_scores[:, 0]
    top_row_out["PC2_top10"] = top_scores[:, 1]
    top_row_out["PC3_top10"] = top_scores[:, 2] if top_scores.shape[1] > 2 else np.nan
    top_row_out.to_csv(out_top10_csv, index=False)

    # 6. Plot Combined Panel (All samples vs. Top 10 Closest)
    fig2, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), dpi=args.dpi)
    fig2.patch.set_facecolor("white")
    plot_pca_panel(
        axes[0],
        row_out,
        scores,
        explained,
        show_y_label=True,
        panel_label="All samples",
    )
    plot_pca_panel(
        axes[1],
        top_row_out,
        top_scores,
        top_explained,
        show_y_label=True,
        panel_label="Top 10 closest species",
    )
    handles2 = build_legend_handles(row_out)
    legend2 = fig2.legend(
        handles=handles2,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.94),
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
    )
    for text in legend2.get_texts():
        text.set_fontfamily("Arial")
        text.set_fontstyle("normal")

    fig2.subplots_adjust(left=0.075, right=0.80, bottom=0.14, top=0.94, wspace=0.32)
    fig2.savefig(out_comb_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    fig2.savefig(out_comb_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig2)

    # Console Summary
    print("=" * 50)
    print("CLR-PCA Analysis Complete!")
    print(f"Total Rows Processed: {len(row_out)}")
    print(f"Codons Analyzed: {len(codon_cols)}")
    print(f"Explained Variance: PC1={explained[0]:.2f}%, PC2={explained[1]:.2f}%")
    print(f"Results saved to directory: {args.output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
