import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
# 1. Dipole RMSF
# --------------------
dipole_results = {}
for fname in sorted(glob.glob("dipole_res*.dat")):
    res_id = int(fname.split("res")[1].split(".")[0])
    data = np.loadtxt(fname, comments="#")
    if data.ndim == 1:
        magnitudes = np.array([data[-1]])
    else:
        magnitudes = data[:, -1]
    mean = np.mean(magnitudes)
    rmsf = np.sqrt(np.mean((magnitudes - mean)**2))
    dipole_results[res_id] = rmsf

dip_df = pd.DataFrame.from_dict(dipole_results, orient="index", columns=["Dipole_RMSF"])
dip_df.index.name = "Residue"

# --------------------
# 2. Cartesian RMSF
# --------------------
cart_data = []
for f in sorted(glob.glob("fluct.*.apf")):
    df = pd.read_csv(f, delim_whitespace=True, comment="#", names=["Residue", "RMSF"])
    cart_data.append(df)
cart_df = pd.concat(cart_data).sort_values("Residue").reset_index(drop=True)
cart_df = cart_df.groupby("Residue", as_index=False)["RMSF"].mean()
cart_df.rename(columns={"RMSF": "Cartesian_RMSF"}, inplace=True)

# --------------------
# 3. Combine both
# --------------------
combined = pd.merge(cart_df, dip_df, left_on="Residue", right_index=True, how="outer").sort_values("Residue")

# --------------------
# 4. Define duplex sections
# --------------------
duplex_mapping = [
    ((1,218),(219,436),"HIV-IN"),
    ((437,499),(500,562),"Lobe-BS"),
    ((563,609),(610,656),"Lobe-CT"),
    ((660,667),(699,706),"DNA-IN"),
    ((674,684),(714,723),"DNA-VL")
]

avg_blocks = []
for (a_start,a_end),(b_start,b_end),label in duplex_mapping:
    a_range = list(range(a_start,a_end+1))
    b_range = list(range(b_start,b_end+1))
    n = min(len(a_range), len(b_range))
    
    df_a = combined[combined["Residue"].isin(a_range[:n])].reset_index(drop=True)
    df_b = combined[combined["Residue"].isin(b_range[:n])].reset_index(drop=True)
    
    block = pd.DataFrame({
        "ResidueA": df_a["Residue"],
        "ResidueB": df_b["Residue"],
        "Cartesian_A": df_a["Cartesian_RMSF"],
        "Cartesian_B": df_b["Cartesian_RMSF"],
        "Dipole_A": df_a["Dipole_RMSF"],
        "Dipole_B": df_b["Dipole_RMSF"],
        "Section": label
    })
    block["Cartesian_Mean"] = block[["Cartesian_A","Cartesian_B"]].mean(axis=1)
    block["Dipole_Mean"] = block[["Dipole_A","Dipole_B"]].mean(axis=1)
    avg_blocks.append(block)

avg_df = pd.concat(avg_blocks, ignore_index=True)
avg_df.to_csv("averaged_duplex_rmsf.csv", index=False)

# --------------------
# 5. Helper function: pseudo-box plot per residue (linear scale)
# --------------------
def plot_duplex_rmsf_linear(avg_df, y_colA, y_colB, y_mean, ylabel, title):
    plt.figure(figsize=(14,6))
    positions = avg_df["ResidueA"].to_numpy()
    
    # Raw (non-log) values
    yA = np.clip(avg_df[y_colA], 1e-8, None)
    yB = np.clip(avg_df[y_colB], 1e-8, None)
    yM = np.clip(avg_df[y_mean], 1e-8, None)
    
    # Create pseudo-box data
    box_data = [[a, b] for a, b in zip(yA, yB)]
    
    plt.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightgray', color='black', linewidth=0.8),
                medianprops=dict(color='black', linewidth=0.8),
                whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8))
    
    plt.scatter(positions, yM, color='red', s=10, label='Mean (A+B)/2')
    
    plt.xlabel("Residue (lower copy numbering)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    
    # Label sections
    for start, end, label in [(1,218,"HIV-IN"),(437,499,"Lobe-BS"),
                              (563,609,"Lobe-CT"),(660,667,"DNA-IN"),
                              (674,684,"DNA-VL")]:
        plt.axvspan(start, end, color='gray', alpha=0.1)
        plt.text((start+end)/2, plt.ylim()[1]*0.95, label, ha='center', fontsize=10, alpha=0.7)
    
    step = 50
    xticks = positions[::step]
    plt.xticks(xticks, [str(x) for x in xticks])
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------
# 6. Plot Cartesian RMSF (linear scale)
# --------------------
plot_duplex_rmsf_linear(avg_df, "Cartesian_A","Cartesian_B","Cartesian_Mean",
                        "Cartesian RMSF (Å)","Per-residue Cartesian RMSF (Duplex Copies)")

# --------------------
# 7. Plot Dipole RMSF (linear scale)
# --------------------
plot_duplex_rmsf_linear(avg_df, "Dipole_A","Dipole_B","Dipole_Mean",
                        "Dipole RMSF (Debye)","Per-residue Dipole RMSF (Duplex Copies)")

# --------------------
# 8. Scatter plot (log2-scaled, collapsed ranges only)
# --------------------
# --------------------
# 8. Scatter plot (log2-scaled, restricted to key ranges)
# --------------------
# Define collapsed ranges
collapsed_ranges = [(61,190), (437,499), (563,609), (660,667), (674,684)]

def in_collapsed_ranges(residue):
    return any(start <= residue <= end for start, end in collapsed_ranges)

# Filter to only residues in collapsed ranges
valid = avg_df[avg_df["ResidueA"].apply(in_collapsed_ranges)].dropna(
    subset=["Cartesian_Mean","Dipole_Mean"]
).copy()

# Apply log2 transform safely (clip small/irrational values)
valid["Cartesian_Mean_log2"] = np.log2(np.clip(valid["Cartesian_Mean"], 1e-8, None))
valid["Dipole_Mean_log2"]   = np.log2(np.clip(valid["Dipole_Mean"], 1e-8, None))

# Only keep residues where BOTH are > 0 on the log2 scale
high_flex = valid[
    (valid["Cartesian_Mean_log2"] > 0) & 
    (valid["Dipole_Mean_log2"] > 0)
]

plt.figure(figsize=(7,7))
plt.scatter(
    high_flex["Cartesian_Mean_log2"],
    high_flex["Dipole_Mean_log2"],
    s=60, color='steelblue', edgecolor='black', alpha=0.8,
)

# Label residues
for _, r in high_flex.iterrows():
    plt.text(
        r["Cartesian_Mean_log2"] + 0.05,
        r["Dipole_Mean_log2"] + 0.05,
        str(int(r["ResidueA"])),
        fontsize=9, color="black", alpha=0.8
    )

plt.axhline(0, color='green', lw=1, ls='--', alpha=0.7)
plt.axvline(0, color='red', lw=1, ls='--', alpha=0.7)

plt.xlabel("log₂(Cartesian RMSF + 1e−8)", fontsize=12)
plt.ylabel("log₂(Dipole RMSF + 1e−8)", fontsize=12)
plt.title("log₂ Dipole vs Cartesian RMSF (collapsed ranges, both > 0)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------
# 9. Section counts (for residues with log₂(RMSF) > 0)
# --------------------

# Define thresholds
cart_thresh = 0  # log₂(RMSF) > 0 means RMSF > 1
dip_thresh = 0

# Categorize residues based on which RMSF is above threshold
high_cart_only = high_flex[(high_flex["Cartesian_Mean_log2"] > cart_thresh) &
                           (high_flex["Dipole_Mean_log2"] <= dip_thresh)].copy()
high_dip_only = high_flex[(high_flex["Dipole_Mean_log2"] > dip_thresh) &
                          (high_flex["Cartesian_Mean_log2"] <= cart_thresh)].copy()
high_both = high_flex[(high_flex["Dipole_Mean_log2"] > dip_thresh) &
                      (high_flex["Cartesian_Mean_log2"] > cart_thresh)].copy()

# Assign category labels
high_cart_only["Category"] = "High Cartesian only"
high_dip_only["Category"]  = "High Dipole only"
high_both["Category"]      = "High both"

# Combine
all_high = pd.concat([high_cart_only, high_dip_only, high_both], ignore_index=True)

# Compute counts per structural section
section_counts = all_high.groupby(["Section", "Category"]).size().unstack(fill_value=0)

# Ensure all expected columns exist (even if no residues fall in that category)
for col in ["High Cartesian only", "High Dipole only", "High both"]:
    if col not in section_counts.columns:
        section_counts[col] = 0

# Sort columns for consistent color order
section_counts = section_counts[["High Cartesian only", "High Dipole only", "High both"]]

# Plot grouped bar chart
plt.figure(figsize=(10,4))
width = 0.25
x = np.arange(len(section_counts))

plt.bar(x - width, section_counts["High Cartesian only"], width, label="High Cartesian only", color="red")
plt.bar(x,         section_counts["High Dipole only"],   width, label="High Dipole only",   color="green")
plt.bar(x + width, section_counts["High both"],          width, label="High both",          color="blue")

plt.xticks(x, section_counts.index, rotation=45)
plt.ylabel("Residue count (log₂-scaled RMSF > 0)")
plt.title("Residue counts per section (RMSF > 1 regions)")
plt.legend()
plt.tight_layout()
plt.show()