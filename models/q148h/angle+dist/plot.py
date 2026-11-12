import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
RUN1_DIR = Path("./run1")
RUN2_DIR = Path("./run2")
OUTPUT_DIR = Path("./plots_comparison")


def load_md_data(filepath):
    """Reads the two-column MD data file, skipping the header line.

    Returns the values (second column).
    """
    try:
        # np.loadtxt automatically ignores lines starting with '#'
        data = np.loadtxt(filepath)
        # Extract the second column (index 1) which contains the angles/distances
        return data[:, 1]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def plot_distributions():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all .dat files in both directories
    run1_files = {f.name for f in RUN1_DIR.glob("*.dat")}
    run2_files = {f.name for f in RUN2_DIR.glob("*.dat")}

    # Find the files present in both directories
    common_files = sorted(list(run1_files.intersection(run2_files)))

    if not common_files:
        print("No matching .dat files found between run1 and run2.")
        return

    print(f"Found {len(common_files)} matching files. Generating plots...")

    for filename in common_files:
        file1 = RUN1_DIR / filename
        file2 = RUN2_DIR / filename

        # Load data from both runs
        data1 = load_md_data(file1)
        data2 = load_md_data(file2)

        if data1 is None or data2 is None:
            continue

        # Initialize the plot
        plt.figure(figsize=(8, 5))

        # Determine bin sizing dynamically based on combined data range
        combined = np.concatenate([data1, data2])
        bins = np.linspace(np.min(combined), np.max(combined), 50)

        # Plot Run 1 (Semi-transparent Blue)
        plt.hist(
            data1,
            bins=bins,
            alpha=0.5,
            label="Run 1",
            color="#1f77b4",
            edgecolor="black",
            density=True,
        )

        # Plot Run 2 (Semi-transparent Orange)
        plt.hist(
            data2,
            bins=bins,
            alpha=0.5,
            label="Run 2",
            color="#ff7f0e",
            edgecolor="black",
            density=True,
        )

        # Labels & Aesthetics
        # Dynamically label X-axis based on whether it is an angle or distance
        if "angle" in filename:
            plt.xlabel("Angle (Degrees)", fontsize=12)
        elif "dist" in filename:
            plt.xlabel("Distance (Å)", fontsize=12)
        else:
            plt.xlabel("Value", fontsize=12)

        plt.ylabel("Probability Density", fontsize=12)
        plt.title(f"Distribution Comparison:\n{filename}", fontsize=13, weight="bold")
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Save the plot
        output_path = OUTPUT_DIR / f"compare_{Path(filename).stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f" Saved: {output_path.name}")


if __name__ == "__main__":
    plot_distributions()
