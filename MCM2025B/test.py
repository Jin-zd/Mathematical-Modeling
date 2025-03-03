import numpy as np
import matplotlib.pyplot as plt

years = np.arange(1, 21)
tourism_intensive_area = [45.7, 45.2, 46.8, 47.2, 48.1, 48.2, 46.1, 49.8, 50.5, 51.9, 52.1, 52.9, 53.2, 53.8, 54.8, 55.7, 55.0, 55.7, 56.1, 56.3]
# tourism_intensive_area = [35.1, 34.2, 35.8, 40.2, 37.1, 36.2, 35.1, 38.8, 39.5, 45.9, 46.1, 46.4, 47.2, 47.8, 48.3, 48.7, 49.0, 50.7, 51.9, 52.5]
tourism_affected_area = [(100 - t) * 0.7 for t in tourism_intensive_area]
residential_area = [(100 - t) * 0.2 for t in tourism_intensive_area]
total_budget = np.array(tourism_intensive_area) + np.array(tourism_affected_area) + np.array(residential_area)

tourist_area_percent = np.array(tourism_intensive_area) / total_budget * 100
residential_area_percent = np.array(tourism_affected_area) / total_budget * 100
other_areas_percent = np.array(residential_area) / total_budget * 100

x = np.arange(len(years))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x, tourist_area_percent, width=0.5, label='tourism intensive area', color='#8c82fc')
ax.bar(x, residential_area_percent, width=0.5, bottom=tourist_area_percent, label='tourism affected area', color='#b693fe')
ax.bar(x, other_areas_percent, width=0.5, bottom=tourist_area_percent + residential_area_percent, label='residential area', color='#7effdb')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Budget Allocation by Year and Region (Percentage)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=10)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.tight_layout()

plt.savefig('./images/budget_allocation.png', dpi=500)
plt.close()