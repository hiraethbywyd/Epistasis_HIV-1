#!/usr/env python3
"""
Contact Δ analysis for HIV-IN mutants — collapsed single-protomer representation variant.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors

plt.ioff()

# ----------------- PARAMETERS & MAPPINGS -----------------
MIN_THRESH = 0.1
OUTPUT_DIR = "epistasis_contact_analysis_out"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Coordinates for sub-dividing IN domains based on their specific start/end
IN_MAPPINGS = {
    (1, 218):    {"IN-NTD": (1, 51),    "IN-CCD": (61, 208)},
    (219, 436): {"IN-NTD": (219, 269), "IN-CCD": (279, 426)},
    (329, 546): {"IN-NTD": (329, 379), "IN-CCD": (389, 536)}
}

# Domain Definitions (The Layouts)
SYSTEM_domains = {
    "wt": [
        ((1, 218), (219, 436), "IN"),
        ((437, 499), (500, 562), "CTD I"),
        ((563, 609), (610, 656), "CTD II"),
        ((657, 734), "DNA"),
    ],
    "e138k-q148k": [
        ((1, 218), (219, 436), "IN"),
        ((437, 561), (500, 562), "CTD I"),
        ((563, 609), (610, 656), "CTD II"),
        ((657, 734), "DNA"),
    ],
    "g140s-s147g": [
        ((1, 218), (219, 436), "IN"),
        ((437, 561), (500, 562), "CTD I"),
        ((563, 609), (610, 656), "CTD II"),
        ((657, 734), "DNA"),
    ],
    "default": [
        ((1, 218), (329, 546), "IN"),
        ((219, 281), (672, 734), "CTD I"),
        ((282, 328), (547, 593), "CTD II"),
        ((594, 671), "DNA"),
    ]
}

# Mapping System Keys to the correct Layouts above
SYSTEM_LAYOUT_MAP = {
    "wt": "wt",
    "e138k-n155h": "wt",
    "y143c-s230r": "wt",
    "y143r-n155h": "wt",
    "e138k-q148k": "e138k-q148k",
    "g140s-s147g": "g140s-s147g"
}

# Display names and groupings for plotting
DOMAIN_DISPLAY_NAMES = {
    "IN": "Catalytic\nNTD-CCD",
    "IN-NTD": "Catalytic\nNTD",
    "IN-CCD": "Catalytic\nCCD",
    "CTD I": "CTD I",
    "CTD II": "CTD II",
    "DNA": "DNA",
}

DOMAIN_GROUPS_INTRA = ["IN", "CTD I", "CTD II", "DNA"]

# Mutant class sets (lowercase)
ANTAGONISTIC = {"g140s-n155h", "e138k-n155h", "q148h-n155h", "y143r-n155h", "g140s-s147g"}
SYNERGISTIC  = {"e138k-q148k", "e138k-s147g", "g140a-q148k", "g140s-q148h", "y143c-s230r"}

# ---------------------------
# Column key text (defines R#P# mapping per system)
# ---------------------------
COLUMN_KEY_TEXT = """
wt- R1P1, R1P2, R2P1, R2P2
G140S-Q148H- R1P1, R1P2, R2P1, R2P2
E138K-N155H- R1P1, R1P2, R2P2
E138K-Q148K- R1P1, R2P1, R2P2
E138K-S147G- R1P1, R2P1
G140A-Q148K- R1P1, R2P1
G140S-N155H- R1P1, R1P2, R2P1, R2P2
G140S-S147G- R1P1, R1P2, R2P1, R2P2
Q148H-N155H- R1P2, R2P1, R2P2
G140A- R2P1, R1P2, R2P2
G140S- R1P1, R1P2, R2P1, R2P2
Q148H- R1P1, R2P1
E138K- R1P1, R2P1, R2P2
N155H- R1P1, R1P2, R2P1, R2P2
Q148K- R1P1, R1P2, R2P1, R2P2
S147G- R1P1, R1P2, R2P1, R2P2
Y143C- R1P1, R1P2, R2P1, R2P2
S230R- R1P1, R1P2, R2P1, R2P2
Y143R- R1P1, R1P2, R2P1, R2P2
"""

# Parse COLUMN_KEY_TEXT into a dictionary
COLUMN_KEY_MAPPING = {}
for line in COLUMN_KEY_TEXT.strip().split('\n'):
    if not line.strip():
        continue
    parts = line.split('-')
    sys_name = parts[0].strip().lower()
    reps = [x.strip() for x in parts[1].split(',')]
    COLUMN_KEY_MAPPING[sys_name] = reps

def get_paired_domains(sys_name):
    layout_key = SYSTEM_LAYOUT_MAP.get(sys_name.lower(), "default")
    return SYSTEM_domains.get(layout_key, SYSTEM_domains["default"])

def get_active_reps(sys_name):
    return COLUMN_KEY_MAPPING.get(sys_name.lower(), ["R1P1", "R1P2", "R2P1", "R2P2"])

# ----------------- UTILS (COLLAPSED COORDINATES) -----------------
def get_collapsed_domain_info(paired_domains):
    domain_ranges = {}
    boundaries = [0]
    centers = []
    labels = []
    current_pos = 0
    
    for item in paired_domains:
        if len(item) == 3:
            A, B, name = item
        elif len(item) == 2:
            A, name = item
        else:
            continue
        
        span = (A[1] - A[0] + 1)
        start_idx = current_pos + 1
        current_pos += span
        end_idx = current_pos
        
        domain_ranges[name] = (start_idx, end_idx)
        boundaries.append(current_pos)
        
        display_name = DOMAIN_DISPLAY_NAMES.get(name, name)
        labels.append(display_name)
        centers.append(boundaries[-2] + span / 2.0)
        
    total_len = current_pos
    return domain_ranges, total_len, boundaries, centers, labels

def load_contact_file_as_canonical_map(path, paired_domains, active_reps):
    df = pd.read_csv(path, sep=r"\s+", names=["i","j","v1","v2"], engine="python", comment="#")
    df = df.dropna(how='any').reset_index(drop=True)

    if len(df) > 0:
        first_row = df.iloc[0]
        if (first_row[['i','j','v1','v2']] == 0).all():
            df = df.iloc[1:].reset_index(drop=True)

    df[['i','j']] = np.sort(df[['i','j']], axis=1)

    contact_map = {}
    domain_ranges, total_len, _, _, _ = get_collapsed_domain_info(paired_domains)

    for item in paired_domains:
        if len(item) == 3:
            A, B, name = item
        elif len(item) == 2:
            A, name = item
            B = A
        else:
            continue

        a1, a2 = A
        b1, b2 = B
        col_start, _ = domain_ranges[name]

        base_map = {}

        for r in df.itertuples(index=False):
            if a1 <= r.i <= a2 and a1 <= r.j <= a2:
                base_i = int(r.i - a1)
                base_j = int(r.j - a1)
                if base_i > base_j:
                    base_i, base_j = base_j, base_i
                key = (base_i, base_j)
                base_map.setdefault(key, {})
                base_map[key]['R1P1'] = r.v1
                base_map[key]['R2P1'] = r.v2

            if b1 <= r.i <= b2 and b1 <= r.j <= b2:
                base_i = int(r.i - b1)
                base_j = int(r.j - b1)
                if base_i > base_j:
                    base_i, base_j = base_j, base_i
                key = (base_i, base_j)
                base_map.setdefault(key, {})
                base_map[key]['R1P2'] = r.v1
                base_map[key]['R2P2'] = r.v2

        for k, rep_dict in base_map.items():
            vals = []
            valid = True
            for rep in active_reps:
                if rep not in rep_dict or np.isnan(rep_dict[rep]) or rep_dict[rep] == 0:
                    valid = False
                    break
                vals.append(rep_dict[rep])

            if not valid:
                continue

            col_i = col_start + int(k[0])
            col_j = col_start + int(k[1])

            contact_map[(col_i, col_j)] = {
                'vals': np.array(vals, dtype=float),
                'domain': name
            }

    return contact_map, total_len, domain_ranges

# ----------------- MAIN -----------------
FOLDERS_FILE = "folders.txt"
with open(FOLDERS_FILE) as fh:
    systems_raw = [ln.strip() for ln in fh if ln.strip()]
singles = [s for s in systems_raw if "-" not in s and s != "wt"]
doubles = [s for s in systems_raw if "-" in s]
wt_folder = "wt"

domain_names = []
for item in SYSTEM_domains["wt"]:
    if len(item) == 3:
        domain_names.append(item[2])
    elif len(item) == 2:
        domain_names.append(item[1])

# load WT
wt_path = os.path.join(wt_folder, f"{wt_folder}_combined_contact.dat")
if not os.path.exists(wt_path):
    raise FileNotFoundError(f"WT contact file missing: {wt_path}")

print("Loading WT contact canonical map...")
wt_active_reps = get_active_reps(wt_folder)
wt_paired = get_paired_domains(wt_folder)
wt_map, wt_total_len, wt_ranges = load_contact_file_as_canonical_map(wt_path, wt_paired, wt_active_reps)

# load singles
single_maps = {}
single_lengths = {}
for sm in singles:
    path = os.path.join(sm, f"{sm}_combined_contact.dat")
    if not os.path.exists(path):
        print(f"Missing {path}, skipping {sm}")
        continue
    print(f"Loading SM: {sm}")
    sm_reps = get_active_reps(sm)
    sm_paired = get_paired_domains(sm)
    sm_map, sm_len, _ = load_contact_file_as_canonical_map(path, sm_paired, sm_reps)
    single_maps[sm] = sm_map
    single_lengths[sm] = sm_len

def get_mean_val(contact_map, key):
    if key in contact_map and len(contact_map[key]['vals']) > 0:
        return float(np.mean(contact_map[key]['vals']))
    return None

agg_dd_gain = {domain: {"ANTAGONISTIC": 0, "SYNERGISTIC": 0} for domain in domain_names}
agg_dd_loss = {domain: {"ANTAGONISTIC": 0, "SYNERGISTIC": 0} for domain in domain_names}

# Process DMs
for dm in doubles:
    print(f"\nProcessing DM: {dm}")
    sm1, sm2 = dm.split('-')
    if sm1 not in single_maps or sm2 not in single_maps:
        print(f"Missing single maps for {dm}: {sm1} or {sm2} not found. Skipping DM.")
        continue

    dm_path = os.path.join(dm, f"{dm}_combined_contact.dat")
    if not os.path.exists(dm_path):
        print(f"Missing DM contact file {dm_path}, skipping")
        continue

    dm_reps = get_active_reps(dm)
    dm_paired = get_paired_domains(dm)
    
    dm_map, dm_total_len, _ = load_contact_file_as_canonical_map(dm_path, dm_paired, dm_reps)

    all_keys = set(wt_map.keys()) | set(dm_map.keys()) | set(single_maps[sm1].keys()) | set(single_maps[sm2].keys())

    delta_dm_map = {}
    delta_sm1_map = {}
    delta_sm2_map = {}

    for key in all_keys:
        domain = None
        for m in (dm_map, single_maps[sm1], single_maps[sm2], wt_map):
            if key in m:
                domain = m[key]['domain']
                break

        wt_mean = get_mean_val(wt_map, key)
        dm_mean = get_mean_val(dm_map, key)
        sm1_mean = get_mean_val(single_maps[sm1], key)
        sm2_mean = get_mean_val(single_maps[sm2], key)

        if wt_mean is not None:
            if dm_mean is not None:
                d_dm = dm_mean - wt_mean
                delta_dm_map[key] = {'mean_delta': d_dm, 'domain': domain}
            if sm1_mean is not None:
                d_sm1 = sm1_mean - wt_mean
                delta_sm1_map[key] = {'mean_delta': d_sm1, 'domain': domain}
            if sm2_mean is not None:
                d_sm2 = sm2_mean - wt_mean
                delta_sm2_map[key] = {'mean_delta': d_sm2, 'domain': domain}

    dm_mean_mat = np.zeros((dm_total_len, dm_total_len))
    sm1_mean_mat = np.zeros((dm_total_len, dm_total_len))
    sm2_mean_mat = np.zeros((dm_total_len, dm_total_len))

    for k, v in delta_dm_map.items():
        i, j = k
        if i-1 < dm_total_len and j-1 < dm_total_len:
            dm_mean_mat[i-1, j-1] = v['mean_delta']
            dm_mean_mat[j-1, i-1] = v['mean_delta']
    for k, v in delta_sm1_map.items():
        i, j = k
        if i-1 < dm_total_len and j-1 < dm_total_len:
            sm1_mean_mat[i-1, j-1] = v['mean_delta']
            sm1_mean_mat[j-1, i-1] = v['mean_delta']
    for k, v in delta_sm2_map.items():
        i, j = k
        if i-1 < dm_total_len and j-1 < dm_total_len:
            sm2_mean_mat[i-1, j-1] = v['mean_delta']
            sm2_mean_mat[j-1, i-1] = v['mean_delta']

# ---- plotting: white background, flipped colormap, and enlarged/smoothed marks ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.15})
    fig.patch.set_facecolor('white')

    panels = [
        (sm1_mean_mat, sm1, delta_sm1_map),
        (sm2_mean_mat, sm2, delta_sm2_map),
        (dm_mean_mat, dm, delta_dm_map)
    ]

    matrices = [p[0] for p in panels]
    all_vals = np.concatenate([m.flatten() for m in matrices])
    
    # Filter out zeros and NaNs to find the true distribution of changes
    valid_vals = all_vals[(all_vals != 0) & (~np.isnan(all_vals))]
    
    if valid_vals.size > 0:
        p_low, p_high = np.percentile(valid_vals, 5), np.percentile(valid_vals, 95)
        limit = max(abs(p_low), abs(p_high))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = -MIN_THRESH * 2, MIN_THRESH * 2

    if vmin == vmax or vmin == 0 or vmax == 0:
        vmin, vmax = -MIN_THRESH * 2, MIN_THRESH * 2

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Flipped colormap: Red = Loss (negative), Gray = Zero, Blue = Gain (positive)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "flipped_high_contrast", 
        [(0.0, '#FF2200'), (0.5, '#B0B0B0'), (1.0, '#0044FF')]
    )

    _, _, boundaries, centers, labels = get_collapsed_domain_info(dm_paired)

    im = None
    for idx in range(3):
        mat, name, dmap = panels[idx]
        hax = axes[idx]
        hax.set_facecolor('white')
        
        # interpolation='bicubic' expands and softens the points so they look larger and more prominent
        im = hax.imshow(mat, origin='lower', cmap=cmap, norm=norm, interpolation='bicubic')
        hax.set_title(f"Δ vs WT: {name.upper()}", fontsize=12, color='black')
        
        hax.set_xticks(centers)
        hax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
        hax.set_xlabel("Domain", color='black')

        if idx == 0:
            hax.set_yticks(centers)
            hax.set_yticklabels(labels, fontsize=9, rotation=45, va='bottom')
            hax.set_ylabel("Domain", color='black')
        else:
            hax.set_yticks([])

        for b in boundaries[1:-1]:
            hax.axhline(b - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            hax.axvline(b - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        hax.tick_params(axis='both', direction='in', colors='black')

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.20)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        fig.colorbar(im, cax=cbar_ax, label='Δ Contact Value (delta of means)')

    heatmap_file = os.path.join(OUTPUT_DIR, f"{dm}_delta_heatmaps.png")
    plt.savefig(heatmap_file, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved Δ heatmaps: {heatmap_file}")
        
    # ---- Dump Expanded CSVs (Double + Single Mutants) ----
    contact_rows = []
    union_keys = set(delta_dm_map.keys()) | set(delta_sm1_map.keys()) | set(delta_sm2_map.keys())
    for key in sorted(union_keys):
        i, j = key
        domain = None
        for m in (delta_dm_map, delta_sm1_map, delta_sm2_map):
            if key in m and 'domain' in m[key]:
                domain = m[key]['domain']
                break

        row = {
            'i': i, 
            'j': j, 
            'domain': domain,
            'mean_delta': delta_dm_map.get(key, {}).get('mean_delta', np.nan),
            f'mean_delta_{sm1}': delta_sm1_map.get(key, {}).get('mean_delta', np.nan),
            f'mean_delta_{sm2}': delta_sm2_map.get(key, {}).get('mean_delta', np.nan)
        }
        contact_rows.append(row)

    df_contacts = pd.DataFrame(contact_rows)
    delta_contacts_file = os.path.join(OUTPUT_DIR, f"{dm}_delta_contacts.csv")
    df_contacts.to_csv(delta_contacts_file, index=False)
    print(f"Saved expanded per-contact CSV: {delta_contacts_file}")

    delta_res_sum = pd.DataFrame({
        'res': np.arange(1, dm_total_len + 1),
        'delta_sum': dm_mean_mat.sum(axis=0)
    })
    delta_res_file = os.path.join(OUTPUT_DIR, f"{dm}_delta_residues.csv")
    delta_res_sum.to_csv(delta_res_file, index=False)
    print(f"Saved per-residue CSV: {delta_res_file}")

    # ---- ΔΔ calculation per domain ----
    def gained_contacts_from_delta_map(dmap):
        s = set()
        for k, meta in dmap.items():
            if meta['mean_delta'] > MIN_THRESH:
                s.add((k, meta['domain']))
        return s

    def lost_contacts_from_delta_map(dmap):
        s = set()
        for k, meta in dmap.items():
            if meta['mean_delta'] < -MIN_THRESH:
                s.add((k, meta['domain']))
        return s

    dm_gained = gained_contacts_from_delta_map(delta_dm_map)
    sm1_gained = gained_contacts_from_delta_map(delta_sm1_map)
    sm2_gained = gained_contacts_from_delta_map(delta_sm2_map)

    dm_lost = lost_contacts_from_delta_map(delta_dm_map)
    sm1_lost = lost_contacts_from_delta_map(delta_sm1_map)
    sm2_lost = lost_contacts_from_delta_map(delta_sm2_map)

    per_domain_dd_gain = {d: 0 for d in domain_names}
    per_domain_dd_loss = {d: 0 for d in domain_names}

    for domain in domain_names:
        dm_set = set([k for (k, dom) in dm_gained if dom == domain])
        sm_union = set([k for (k, dom) in sm1_gained if dom == domain]) | set([k for (k, dom) in sm2_gained if dom == domain])
        dd_gain = len(dm_set) - len(sm_union)
        per_domain_dd_gain[domain] = dd_gain

        dm_set_loss = set([k for (k, dom) in dm_lost if dom == domain])
        sm_union_loss = set([k for (k, dom) in sm1_lost if dom == domain]) | set([k for (k, dom) in sm2_lost if dom == domain])
        dd_loss = len(dm_set_loss) - len(sm_union_loss)
        per_domain_dd_loss[domain] = dd_loss

        if dm.lower() in ANTAGONISTIC:
            agg_dd_gain[domain]["ANTAGONISTIC"] += dd_gain
            agg_dd_loss[domain]["ANTAGONISTIC"] += dd_loss
        if dm.lower() in SYNERGISTIC:
            agg_dd_gain[domain]["SYNERGISTIC"] += dd_gain
            agg_dd_loss[domain]["SYNERGISTIC"] += dd_loss

    df_dom = pd.DataFrame({
        'domain': list(per_domain_dd_gain.keys()),
        'dd_gain': [per_domain_dd_gain[d] for d in per_domain_dd_gain],
        'dd_loss': [per_domain_dd_loss[d] for d in per_domain_dd_loss]
    })
    dm_dom_file = os.path.join(OUTPUT_DIR, f"{dm}_per_domain_dd.csv")
    df_dom.to_csv(dm_dom_file, index=False)
    print(f"Saved per-domain ΔΔ counts: {dm_dom_file}")

print("\nAll DMs processed successfully.")