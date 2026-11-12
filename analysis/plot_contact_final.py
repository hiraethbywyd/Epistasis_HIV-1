#!/usr/bin/env python3
"""
Protomer-aware epistatic contact analysis (double-mutant cycle).

Main features:
- Explicit residue-residue matching (contacts keyed by (i,j))
- Explicit replicate matching by R# (from COLUMN_KEY_TEXT)
- Protomer-aware domain classification (per-system layouts)
- Inter-domain contacts included (e.g. "IN|CTD I")
- Strict consensus across replicates for each protomer
- Annotated and robust to differing column orders across systems
- Updated to use flexible IN subdomain mappings per protomer range
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.ioff()

# ---------------------------
# User parameters
# ---------------------------
epi_cutoff = 0.1         
output_dir = "epistasis_consensus_analysis_out"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Domain definitions with IN subdomain mappings
# ---------------------------

# Coordinates for sub-dividing IN domains based on their specific start/end
IN_MAPPINGS = {
    (1, 218):   {"IN-NTD": (1, 51),    "IN-CCD": (61, 208)},
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
ANTAGONISTIC = {"g140s-n155h","e138k-n155h","q148h-n155h","y143r-n155h","g140s-s147g"}
SYNERGISTIC  = {"e138k-q148k","e138k-s147g","g140a-q148k","g140s-q148h","y143c-s230r"}

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
Y143C-S230R- R1P1, R1P2, R2P1
Y143R-N155H- R1P1, R1P2, R2P1, R2P2
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

def parse_column_key(text):
    """
    Parse COLUMN_KEY_TEXT into:
      system_key (lowercase) -> list of (replicate_int, protomer_int)

    Correctly handles system names containing hyphens by splitting
    on the *last* hyphen in each line.
    """
    out = {}
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # Split on LAST hyphen only
        try:
            sys_name, col_text = ln.rsplit("-", 1)
        except ValueError:
            raise ValueError(f"Could not parse COLUMN_KEY_TEXT line: {ln}")

        sys_key = sys_name.strip().lower()
        entries = []

        for token in col_text.split(","):
            token = token.strip()
            m = re.fullmatch(r"R(\d+)P(\d+)", token, flags=re.I)
            if not m:
                raise ValueError(
                    f"Bad column token: '{token}' in line: '{ln}'"
                )
            entries.append((int(m.group(1)), int(m.group(2))))

        out[sys_key] = entries

    return out

COLUMN_MAP = parse_column_key(COLUMN_KEY_TEXT)

# ---------------------------
# Contact file loader
# ---------------------------
def load_contact_file_by_replicate(path, reps):
# reps looks like [(1, 2), (2, 2)] (Replicate, Protomer)
    contacts = defaultdict(dict)
    if not os.path.exists(path): return contacts
    with open(path) as fh:
        for ln in fh:
            toks = ln.strip().split()
            if len(toks) < 3: continue
            i, j = sorted(map(int, toks[:2]))
            values = [float(x) for x in toks[2:]]
            for (rep_num, prot_num), val in zip(reps, values):
                # KEY BY REP_NUM so we can match R1 in mutant to R1 in WT
                contacts[(i, j)][rep_num] = val
    return contacts

    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split()
            if len(toks) < 3:
                continue
            try:
                i, j = sorted(map(int, toks[:2]))
            except ValueError:
                continue
            values = [float(x) for x in toks[2:]]
            # Map values to replicate numbers explicitly
            for (rep, prot), val in zip(reps, values):
                contacts[(i, j)][rep] = val
            # if len(values) != len(reps): you might log or warn for diagnostics
    return contacts

# Load all systems (maps: system -> {(i,j): {rep: val, ...}, ...})
all_maps = {
    sys_key: load_contact_file_by_replicate(f"{sys_key}/{sys_key}_combined_contacts.dat", reps)
    for sys_key, reps in COLUMN_MAP.items()
}
wt_map = all_maps.get("wt", {})    # may be empty if file not present

# ---------------------------
# Domain classification (protomer-aware). Includes inter-domain labels.
# ---------------------------
def get_system_layout(system_key):
    """
    Get the appropriate domain layout for a system.
    Uses SYSTEM_LAYOUT_MAP if available, otherwise falls back to 'default'.
    """
    layout_key = SYSTEM_LAYOUT_MAP.get(system_key, "default")
    return SYSTEM_domains.get(layout_key, SYSTEM_domains["default"])

def classify_contact(system_key, protomer, i, j):
    """
    Return a domain label for the contact (i,j) in the context of 'system_key' and 'protomer'.
    - If both residues map to the same IN subdomain -> return "IN-NTD" or "IN-CCD"
    - If both residues map to same domain -> return "IN", "CTD I", etc.
    - If residues map to different domains -> return "DOM1|DOM2" (sorted alphabetically)
    - If either residue doesn't map -> return None
    """
    domains = get_system_layout(system_key)
    idx = protomer - 1  # protomer 1 -> index 0, protomer 2 -> index 1

    def domain_of(res):
        for *ranges, name in domains:
            if idx < len(ranges) and ranges[idx][0] <= res <= ranges[idx][1]:
                return name, ranges[idx]  # return both domain name and the range
        return None, None

    di, range_i = domain_of(i)
    dj, range_j = domain_of(j)
    
    if di is None or dj is None:
        return None

    # IN subdomain specificity
    if di == "IN" and dj == "IN" and range_i == range_j:
        # Both residues in same IN range - check if they're in same subdomain
        if range_i in IN_MAPPINGS:
            subdomain_map = IN_MAPPINGS[range_i]
            # Check each subdomain
            for subdomain_name, (sub_start, sub_end) in subdomain_map.items():
                if sub_start <= i <= sub_end and sub_start <= j <= sub_end:
                    return subdomain_name
        # If we're here, both are in IN but not in same subdomain (or subdomain not defined)
        return "IN"

    if di == dj:
        return di

    # inter-domain → combined label sorted for consistent naming
    return "|".join(sorted((di, dj)))

def contact_scope(domain_label):
    """
    Returns 'intra' or 'inter' based on domain label.
    """
    return "inter" if "|" in domain_label else "intra"

def collect_inter_domains(agg_inter):
    """
    Return sorted list of unique inter-domain labels.
    """
    labels = {dom for (_, _, dom) in agg_inter}
    return sorted(labels)

# ---------------------------
# Plotting helpers (auto include inter-domain labels related to the base domains)
# ---------------------------
def domain_keys_for_plot(label):
    """Return list of domain keys used for aggregating counts when plotting."""
    return ["IN-NTD", "IN-CCD"] if label == "IN" else [label]

def plot_bar_raw(agg, fname, base_domains, show_legend=True):
    """
    Plot only the aggregated net epistatic changes for Synergistic and Antagonistic classes.
    - base_domains: list like DOMAIN_GROUPS_INTRA
    - This function will also include inter-domain labels that involve any base_domain.
    """
    # find inter-domain labels present in agg that involve any base domain
    present_labels = {label for (_, _, label) in agg.keys()}
    interlabels = sorted(
        lab for lab in present_labels
        if "|" in lab and any(bd in lab.split("|") for bd in base_domains)
    )

    # full domain list to plot = base domains + matched inter-domain labels
    domains = list(base_domains) + interlabels

    x = np.arange(len(domains))
    w = 0.25  # Wider bar width since we are only plotting 2 bars per domain group

    # Define styles exclusively for the Net bars
    net_styles = [
        ("Synergistic",  "mediumblue", -w/2, '//'),
        ("Antagonistic", "crimson",     w/2, '\\\\'),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot only the net (hollow, striped) bars
    for cls, color, off, hatch in net_styles:
        heights = []
        for d in domains:
            gain = sum(agg.get(("gain", cls, subdom), 0) for subdom in domain_keys_for_plot(d))
            loss = sum(agg.get(("loss", cls, subdom), 0) for subdom in domain_keys_for_plot(d))
            heights.append(gain + loss)  # Net effect (losses are already negative values)
            
        ax.bar(x + off, heights, w, facecolor="none", edgecolor=color,
               hatch=hatch, linewidth=1.5, alpha=0.9, label=f"{cls} Net", zorder=5)
               
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    # display name fallback: if not in display dict, show label with '|' replaced by ' | '
    ax.set_xticklabels([DOMAIN_DISPLAY_NAMES.get(d, d.replace("|", " | ")) for d in domains], fontsize=12)
    ax.set_ylabel("Net ε_Contacts", fontsize=14)
    ax.tick_params(axis="both", direction="in", labelsize=12)
    if show_legend:
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()

#def plot_bar_raw(agg, fname, base_domains, show_legend=True):
#    """
#    Plot aggregated counts in 'agg'.
#    - base_domains: list like DOMAIN_GROUPS_INTRA
#     - This function will also include inter-domain labels that involve any base_domain.
#     """
#     # find inter-domain labels present in agg that involve any base domain
#     present_labels = {label for (_,_,label) in agg.keys()}
#     interlabels = sorted(
#         lab for lab in present_labels
#         if "|" in lab and any(bd in lab.split("|") for bd in base_domains)
#     )
# 
#     # full domain list to plot = base domains + matched inter-domain labels
#     domains = list(base_domains) + interlabels
# 
#     x = np.arange(len(domains))
#     w = 0.12
# 
#     styles = [
#         ("gain", "Synergistic",  "mediumblue", 1.0, -2*w),
#         ("gain", "Antagonistic", "crimson",     1.0, -w),
#         ("loss", "Synergistic",  "mediumblue", 0.5, 0),
#         ("loss", "Antagonistic", "crimson",     0.5, w),
#     ]
#     net_styles = [
#         ("net", "Synergistic",  "mediumblue", 0.5, 2*w, '//'),
#         ("net", "Antagonistic", "crimson",     0.5, 3*w, '\\\\'),
#     ]
# 
#     fig, ax = plt.subplots(figsize=(8, 6))
# 
#     # gain/loss bars (raw sum version)
#     for gl, cls, color, alpha, off in styles:
#         heights = []
#         for d in domains:
#             count = sum(agg.get((gl, cls, subdom), 0) for subdom in domain_keys_for_plot(d))
#             heights.append(count)  # use raw sum directly
#         ax.bar(x + off, heights, w, color=color, alpha=alpha, label=f"{cls} {gl.capitalize()}")
# 
#     # net (hollow, striped)
#     for gl, cls, color, alpha, off, hatch in net_styles:
#         heights = []
#         for d in domains:
#             gain = sum(agg.get(("gain", cls, subdom), 0) for subdom in domain_keys_for_plot(d))
#             loss = sum(agg.get(("loss", cls, subdom), 0) for subdom in domain_keys_for_plot(d))
#             heights.append(gain + loss)  # net effect; losses are negative
#         # plot the net bars
#         ax.bar(x + off, heights, w, facecolor="none", edgecolor=color,
#                hatch=hatch, linewidth=1.5, alpha=alpha, label=f"{cls} Net", zorder=5)
#                
#     ax.axhline(0, color="black", lw=0.8)
#     ax.set_xticks(x)
#     # display name fallback: if not in display dict, show label with '|' replaced by ' | '
#     ax.set_xticklabels([DOMAIN_DISPLAY_NAMES.get(d, d.replace("|", " | ")) for d in domains], fontsize=12)
#     ax.set_ylabel("ε_Contacts", fontsize=14)
#     ax.tick_params(axis="both", direction="in", labelsize=12)
#     if show_legend:
#         ax.legend(fontsize=10)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, fname), dpi=300)
#     plt.close()

# ---------------------------
# Main epistasis analysis (raw ε sum instead of binarization)
# ---------------------------
agg_intra = defaultdict(float)  # sum of raw epsilons
agg_inter = defaultdict(float)  # sum of raw epsilons

for dm_key in list(all_maps.keys()):
    if "-" not in dm_key:
        continue

    dm_key_l = dm_key.lower()
    cls = ("Antagonistic" if dm_key_l in ANTAGONISTIC else
           "Synergistic" if dm_key_l in SYNERGISTIC else None)
    if cls is None:
        continue

    sm1, sm2 = dm_key_l.split("-")
    
    # 1. Get the maps and the expected replicate counts for each component
    dm_map = all_maps.get(dm_key_l, {})
    s1_map = all_maps.get(sm1, {})
    s2_map = all_maps.get(sm2, {})
    
    # Count how many replicates are expected for each
    # This assumes the COLUMN_MAP entries define the 'replicate' IDs for that system
    expected_dm_reps = len(COLUMN_MAP.get(dm_key_l, []))
    expected_s1_reps = len(COLUMN_MAP.get(sm1, []))
    expected_s2_reps = len(COLUMN_MAP.get(sm2, []))
    expected_wt_reps = len(COLUMN_MAP.get("wt", []))

    reps = COLUMN_MAP[dm_key_l]
    prot_to_reps = defaultdict(list)
    for rep, prot in reps:
        prot_to_reps[prot].append(rep)

    contacts = set(dm_map) | set(s1_map) | set(s2_map) | set(wt_map)

    for prot, rep_list in prot_to_reps.items():
        for (i, j) in contacts:
            
            # 2. INTERNAL CONSENSUS CHECK
            # We check if the contact exists in EVERY replicate of the specific system.
            # If not, we set that system's contribution to 0.0 for this contact.
            
            has_dm_consensus = len(dm_map.get((i, j), {})) == expected_dm_reps
            has_s1_consensus = len(s1_map.get((i, j), {})) == expected_s1_reps
            has_s2_consensus = len(s2_map.get((i, j), {})) == expected_s2_reps
            has_wt_consensus = len(wt_map.get((i, j), {})) == expected_wt_reps

            dm_deltas, s1_deltas, s2_deltas = [], [], []

            for rep in rep_list:
                # 3. Apply the consensus: if system lacks consensus, value is 0.0
                wt_v = wt_map.get((i, j), {}).get(rep, 0.0) if has_wt_consensus else 0.0
                dm_v = dm_map.get((i, j), {}).get(rep, 0.0) if has_dm_consensus else 0.0
                s1_v = s1_map.get((i, j), {}).get(rep, 0.0) if has_s1_consensus else 0.0
                s2_v = s2_map.get((i, j), {}).get(rep, 0.0) if has_s2_consensus else 0.0

                dm_deltas.append(dm_v - wt_v)
                s1_deltas.append(s1_v - wt_v)
                s2_deltas.append(s2_v - wt_v)

            # 3. Average the differences across replicates
            mean_dm_d = np.mean(dm_deltas)
            mean_s1_d = np.mean(s1_deltas)
            mean_s2_d = np.mean(s2_deltas)

            # 4. APPLY CUTOFF INDEPENDENTLY
            # If the change isn't large enough, we treat it as 0.0 (no significant change)
            sig_dm_d = mean_dm_d if abs(mean_dm_d) >= epi_cutoff else 0.0
            sig_s1_d = mean_s1_d if abs(mean_s1_d) >= epi_cutoff else 0.0
            sig_s2_d = mean_s2_d if abs(mean_s2_d) >= epi_cutoff else 0.0

            # 5. Calculate Epistasis using only SIGNIFICANT changes
            # eps = ΔV_DM - (ΔV_S1 + ΔV_S2)
            # Note: Your previous code used a 0.5 multiplier; 
            # ensure that aligns with your specific thermodynamic model.
            mean_eps = sig_dm_d - 0.5 * (sig_s1_d + sig_s2_d)

            # 6. Final Filter: Does the epistatic effect itself pass the cutoff?
            if abs(mean_eps) < 1e-5: # Or use a second epi_cutoff if desired
                continue

            dom = classify_contact(dm_key_l, prot, i, j)
            if not dom:
                continue

            scope = contact_scope(dom)
            # Now "loss" vs "gain" is determined by the final epistatic balance
            sign = "gain" if mean_eps > 0 else "loss"

            if scope == "intra":
                agg_intra[(sign, cls, dom)] += mean_eps
            else:
                agg_inter[(sign, cls, dom)] += mean_eps

# ---------------------------
# Plot results (automatically includes inter-domain labels that are present)
# ---------------------------
# Intra-domain plots
plot_bar_raw(
    agg_intra,
    "composite_global_intra.png",
    DOMAIN_GROUPS_INTRA
)

plot_bar_raw(
    agg_intra,
    "composite_IN_breakdown_intra.png",
    ["IN-NTD", "IN-CCD"],
    show_legend=False
)

# Inter-domain plots
INTER_DOMAINS = collect_inter_domains(agg_inter)

if INTER_DOMAINS:
    plot_bar_raw(
        agg_inter,
        "composite_global_inter.png",
        INTER_DOMAINS
    )

print(f"Analysis complete. Results saved to {output_dir}/")