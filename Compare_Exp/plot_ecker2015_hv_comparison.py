"""
Plot HV comparison between ParEGO and LLMBO-MO on Ecker2015 dataset.
Canonical HV is divided by 3 for normalization.
Red line: LLMBO-MO, Blue line: ParEGO with shaded confidence bands.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load ParEGO data
parego_report_path = Path("D:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/parego_ecker_5seeds_56evals_2026_05_11/report.json")
with open(parego_report_path, 'r') as f:
    parego_data = json.load(f)

# Load LLMBO-MO data
llmbo_report_path = Path("D:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/scalarization_Exp/experiment_records/ecker_llmbo_5seeds_50iter_fixed_2026_05_11/report_5seeds.json")
with open(llmbo_report_path, 'r') as f:
    llmbo_data = json.load(f)

# Extract seeds and canonical_hv values
parego_seeds = []
parego_hv = []
for record in parego_data['records']:
    parego_seeds.append(record['seed'])
    parego_hv.append(record['canonical_hv'] / 3.0)  # Divide by 3

llmbo_seeds = []
llmbo_hv = []
for record in llmbo_data['records']:
    llmbo_seeds.append(record['seed'])
    llmbo_hv.append(record['canonical_hv'] / 3.0)  # Divide by 3

# Convert to numpy arrays
parego_seeds = np.array(parego_seeds)
parego_hv = np.array(parego_hv)
llmbo_seeds = np.array(llmbo_seeds)
llmbo_hv = np.array(llmbo_hv)

# Sort by seeds for proper line plotting
parego_sort_idx = np.argsort(parego_seeds)
parego_seeds = parego_seeds[parego_sort_idx]
parego_hv = parego_hv[parego_sort_idx]

llmbo_sort_idx = np.argsort(llmbo_seeds)
llmbo_seeds = llmbo_seeds[llmbo_sort_idx]
llmbo_hv = llmbo_hv[llmbo_sort_idx]

# Calculate statistics for shaded regions
parego_mean = np.mean(parego_hv)
parego_std = np.std(parego_hv)
llmbo_mean = np.mean(llmbo_hv)
llmbo_std = np.std(llmbo_hv)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ParEGO (Blue)
ax.plot(parego_seeds, parego_hv, 'b-o', linewidth=2, markersize=8, label=f'ParEGO (mean={parego_mean:.3f})')
ax.axhline(y=parego_mean, color='b', linestyle='--', alpha=0.5)
ax.fill_between(parego_seeds, parego_mean - parego_std, parego_mean + parego_std, color='b', alpha=0.2)

# Plot LLMBO-MO (Red)
ax.plot(llmbo_seeds, llmbo_hv, 'r-s', linewidth=2, markersize=8, label=f'LLMBO-MO (mean={llmbo_mean:.3f})')
ax.axhline(y=llmbo_mean, color='r', linestyle='--', alpha=0.5)
ax.fill_between(llmbo_seeds, llmbo_mean - llmbo_std, llmbo_mean + llmbo_std, color='r', alpha=0.2)

# Customize plot
ax.set_xlabel('Seed', fontsize=12)
ax.set_ylabel('Canonical HV / 3', fontsize=12)
ax.set_title('HV Comparison: ParEGO vs LLMBO-MO on Ecker2015\n(Canonical HV divided by 3)', fontsize=14)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(parego_seeds)
ax.set_xticklabels([str(s) for s in parego_seeds])

# Add value annotations
for i, (seed, hv) in enumerate(zip(parego_seeds, parego_hv)):
    ax.annotate(f'{hv:.3f}', (seed, hv), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='b')

for i, (seed, hv) in enumerate(zip(llmbo_seeds, llmbo_hv)):
    ax.annotate(f'{hv:.3f}', (seed, hv), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='r')

plt.tight_layout()

# Save figure
output_dir = Path("D:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/Compare_Exp/images/Ecker2015_HV05-12")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "ecker2015_hv_comparison_parego_vs_llmbo.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF
output_path_pdf = output_dir / "ecker2015_hv_comparison_parego_vs_llmbo.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure saved to: {output_path_pdf}")

# Print summary statistics
print("\n" + "="*60)
print("Summary Statistics (Canonical HV / 3)")
print("="*60)
print(f"\nParEGO (5 seeds, 56 evals):")
print(f"  Mean: {parego_mean:.4f}")
print(f"  Std:  {parego_std:.4f}")
print(f"  Min:  {np.min(parego_hv):.4f}")
print(f"  Max:  {np.max(parego_hv):.4f}")

print(f"\nLLMBO-MO (5 seeds, 50 iters):")
print(f"  Mean: {llmbo_mean:.4f}")
print(f"  Std:  {llmbo_std:.4f}")
print(f"  Min:  {np.min(llmbo_hv):.4f}")
print(f"  Max:  {np.max(llmbo_hv):.4f}")

print(f"\nDifference (LLMBO-MO - ParEGO):")
print(f"  Mean difference: {llmbo_mean - parego_mean:.4f} ({((llmbo_mean - parego_mean)/parego_mean)*100:.1f}% improvement)")
print("="*60)

plt.show()
