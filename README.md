#PR2 bias analysis and visualization.


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
