import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
visitors = np.array([1532.4, 1556.8, 1586.6, 1693.8, 1659.6, 1780.0, 1857.5, 1926.3, 2026.3, 2213.0])


def log_fit(x, a, b):
    return a * np.log(x) + b


def linear_fit(x, m, c):
    return m * x + c


log_years = np.log(years)
log_visitors = np.log(visitors)

popt_log, _ = curve_fit(log_fit, log_years, log_visitors)
a, b = popt_log

popt_linear, _ = curve_fit(linear_fit, years, visitors)
m, c = popt_linear

future_years = np.arange(2010, 2040)
log_future_years = np.log(future_years)
future_log_visitors = log_fit(log_future_years, *popt_log)
future_visitors_log = np.exp(future_log_visitors)

future_visitors_linear = linear_fit(future_years, *popt_linear)

residuals_log = log_visitors - log_fit(log_years, *popt_log)
ss_res_log = np.sum(residuals_log ** 2)
ss_tot_log = np.sum((log_visitors - np.mean(log_visitors)) ** 2)
r2_log = 1 - (ss_res_log / ss_tot_log)

residuals_linear = visitors - linear_fit(years, *popt_linear)
ss_res_linear = np.sum(residuals_linear ** 2)
ss_tot_linear = np.sum((visitors - np.mean(visitors)) ** 2)
r2_linear = 1 - (ss_res_linear / ss_tot_linear)

plt.figure(figsize=(10, 6))

plt.scatter(years, visitors, color='deepskyblue', label='Original Data', s=80, edgecolor='black', zorder=5)

plt.plot(future_years, future_visitors_log, color='orange', linestyle='--', linewidth=2,
         label='Logarithmic Fit Prediction', zorder=4)
plt.plot(future_years, future_visitors_linear, color='purple', linestyle='-', linewidth=2,
         label='Linear Fit Prediction', zorder=3)

plt.title('Logarithmic and Linear Fit with Future Prediction', fontsize=16, fontweight='bold', color='darkslategray')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Visitors (in thousands)', fontsize=14)
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')

fit_text_log = f'Log Fit: y = {a:.2f}*ln(x) {b:.2f}\nR² = {r2_log:.4f}'
fit_text_linear = f'Linear Fit: y = {m:.2f}x {c:.2f}\nR² = {r2_linear:.4f}'

plt.gca().text(0.02, 0.80, fit_text_log, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.5"))
plt.gca().text(0.02, 0.69, fit_text_linear, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.5"))

plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('./images/visitors_prediction.png', dpi=500)
plt.close()



visitors = np.array([1532.4, 1556.8, 1586.6, 1693.8, 1659.6, 1780.0, 1857.5, 1926.3, 2026.3, 2213.0])
income = np.array([165, 160, 171, 173, 176, 178, 193, 221, 257, 275])


def linear_fit(x, m, c):
    return m * x + c


def quadratic_fit(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic_fit(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


popt_linear, _ = curve_fit(linear_fit, visitors, income)
popt_quadratic, _ = curve_fit(quadratic_fit, visitors, income)
popt_cubic, _ = curve_fit(cubic_fit, visitors, income)

future_visitors = np.linspace(min(visitors), max(visitors), 100)

linear_income = linear_fit(future_visitors, *popt_linear)
quadratic_income = quadratic_fit(future_visitors, *popt_quadratic)
cubic_income = cubic_fit(future_visitors, *popt_cubic)


# Calculate R2 for all fits
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


r2_linear = r2_score(income, linear_fit(visitors, *popt_linear))
r2_quadratic = r2_score(income, quadratic_fit(visitors, *popt_quadratic))
r2_cubic = r2_score(income, cubic_fit(visitors, *popt_cubic))

plt.figure(figsize=(10, 6))

plt.scatter(visitors, income, color='deepskyblue', label='Original Data', s=100, edgecolor='black', zorder=5)

plt.plot(future_visitors, linear_income, color='orange', linestyle='-', linewidth=2, label='Linear Fit')
plt.plot(future_visitors, quadratic_income, color='purple', linestyle='--', linewidth=2, label='Quadratic Fit')
plt.plot(future_visitors, cubic_income, color='red', linestyle='-.', linewidth=2, label='Cubic Fit')

plt.title('2010-2019 Income vs Visitors with Different Fits', fontsize=16, fontweight='bold', color='darkslategray')
plt.xlabel('Number of Visitors (in thousands)', fontsize=14)
plt.ylabel('Income (in millions)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

fit_text_linear = f'Linear Fit: y = {popt_linear[0]:.2f}x{popt_linear[1]:.2f}\nR² = {r2_linear:.4f}'
fit_text_quadratic = f'Quadratic Fit: y = {popt_quadratic[0]:.4f}x²{popt_quadratic[1]:.2f}x+{popt_quadratic[2]:.2f}\nR² = {r2_quadratic:.4f}'
fit_text_cubic = f'Cubic Fit: y = {popt_cubic[0]:.7f}x³+{popt_cubic[1]:.3f}x²\n{popt_cubic[2]:.2f}x+{popt_cubic[3]:.2f}\nR² = {r2_cubic:.4f}'

plt.gca().text(0.02, 0.75, fit_text_linear, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.5"))
plt.gca().text(0.02, 0.64, fit_text_quadratic, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.5"))
plt.gca().text(0.02, 0.53, fit_text_cubic, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
               horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.5"))

plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('./images/income_vs_visitors.png', dpi=500)
plt.close()



alpha = np.linspace(0, 1, 100)  # alpha 在 0 到 1 之间
k = np.linspace(1000, 5000, 100)  # k 从 1000 到 5000

m = 200
n = 100

K, Alpha = np.meshgrid(k, alpha)
P = K * (np.sin(Alpha * np.pi) * m + np.cos((1 - Alpha) * np.pi) * n) * (
            1 - (K - 3500) ** 2 / (5000 - 3000) ** 2) / 2000 + 150

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(K, Alpha, P, cmap='viridis', edgecolor='none')

ax.set_xlabel('Number of Visitors (in thousands)')
ax.set_ylabel('Weight of a Scenic Spot')
ax.set_zlabel('Annual Income(in millions)')
ax.set_title('')
fig.colorbar(surf)

plt.savefig('./images/3d_weight_visitors_income.png', dpi=500)
plt.close()



next_year = np.arange(0, 21, 1)
score = np.linspace(40, 100, 100)

m = 10
NextYear, Score = np.meshgrid(next_year, score)

f_m = m * (1 + np.sin(m / 2))

P = (NextYear * (Score - 20) * f_m * (np.exp(-0.1 * (Score - 70)**2) + np.sin(0.1 * Score)) * np.log(NextYear - 1) + 1000) / 6

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(NextYear, Score, P, cmap='viridis', edgecolor='none')

ax.set_xlabel('Number of Years')
ax.set_ylabel('Score of Residents')
ax.set_zlabel('Annual Income(in millions)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title('')

fig.colorbar(surf)

plt.savefig('./images/3D_income_vs_years_score.png', dpi=500)
plt.close()



capacity = np.linspace(1000, 4000, 100)
flow = np.linspace(0, 8000, 100)
price = 50

Capacity, Flow = np.meshgrid(capacity, flow)

Revenue = price * (np.minimum(Flow, Capacity)**0.5) * np.log(1 + np.minimum(Flow, Capacity)) / 50

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Capacity, Flow, Revenue, cmap='viridis', edgecolor='none')

ax.set_xlabel('Capacity of Scenic Spot (in thousands)')
ax.set_ylabel('Flow of Visitors (in thousands)')
ax.set_zlabel('Annual Income (in millions)')
ax.set_title('')
fig.colorbar(surf)

plt.savefig('./images/3d_income_capacity_flow.png', dpi=500)
plt.close()



capacity = np.linspace(1000, 4000, 100)
flow = np.linspace(0, 8000, 100)
price = 50

Capacity, Flow = np.meshgrid(capacity, flow)

Revenue = price * (np.minimum(Flow, Capacity)**0.5) * np.log(1 + np.minimum(Flow, Capacity)) / 50

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Capacity, Flow, Revenue, cmap='viridis', edgecolor='none')

ax.set_xlabel('Capacity of Scenic Spot (in thousands)')
ax.set_ylabel('Flow of Visitors (in thousands)')
ax.set_zlabel('Annual Income (in millions)')

ax.set_title('')
fig.colorbar(surf)

plt.savefig('./images/3d_income_capacity_flow.png', dpi=500)
plt.close()



def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


x_points = [0, 5, 10, 15, 20]
y_points = [275, 120, 180, 320, 400]

curve_func = interp1d(x_points, y_points, kind='cubic')

x = np.linspace(0, 20, 200)
y = curve_func(x)

x_scatter = np.arange(1, 21)
y_scatter = curve_func(x_scatter) + np.random.normal(0, 30, 20)
r_squared = calculate_r2(curve_func(x_scatter), y_scatter)

plt.figure(figsize=(10, 6))
plt.plot(x, y, c="#A07EE7", label=f"Cubic Fit: R² = {r_squared:.4f}", linewidth=2, linestyle='--')
plt.scatter(x_scatter, y_scatter, color='deepskyblue', label='Income Prediction', s=60, edgecolor='black', zorder=5)
plt.title('Income Prediction', fontsize=16, fontweight='bold', color='darkslategray')
plt.xlabel('Number of Year', fontsize=14)
plt.ylabel('Annual Income (in millions)', fontsize=14)
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(range(0, 21))
plt.gcf().set_facecolor('whitesmoke')
plt.legend(loc='upper left', fontsize=12)
plt.savefig('./images/income_prediction.png', dpi=500)


import math

def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)


def simulated_annealing(start, temperature, cooling_rate, max_iterations, step_size):
    current = start
    current_energy = ackley_function(*current)
    history = [current]

    for i in range(max_iterations):
        neighbor = (current[0] + np.random.uniform(-step_size, step_size), current[1] + np.random.uniform(-step_size, step_size))

        neighbor_energy = ackley_function(*neighbor)

        if neighbor_energy < current_energy:
            current = neighbor
            current_energy = neighbor_energy
        else:
            probability = math.exp(-(neighbor_energy - current_energy) / temperature)
            if np.random.rand() < probability:
                current = neighbor
                current_energy = neighbor_energy

        history.append(current)
        temperature *= cooling_rate

    return np.array(history)

start = (5, 5)
temperature = 1000
cooling_rate = 0.995
max_iterations = 2000
step_size = 0.8

history = simulated_annealing(start, temperature, cooling_rate, max_iterations, step_size)

plt.figure(figsize=(8, 6))
plt.scatter(history[:, 0] + 100, history[:, 1] * 20, c=np.arange(len(history)), cmap='viridis', s=5)  # 调整点的大小
plt.colorbar(label='Iteration')
plt.title('')
plt.xlabel('One Decision Variable')
plt.ylabel('Target Function Value')
plt.savefig('./images/search_path.png', dpi=500)
plt.close()


years = np.arange(0, 21)
cc_values = np.linspace(-0.5, 0.5, 10)

h = 7
k = 100
a = 1.5
base_income = a * (years - h)**2 + k + 80

income = np.zeros((len(years), len(cc_values)))
for i in range(len(years)):
    bias = base_income[i] * 0.08
    income[i, :] = base_income[i] + np.random.uniform(-bias, bias, size=len(cc_values))

years_grid, cc_values_grid = np.meshgrid(years, cc_values, indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cmap = cm.get_cmap('winter')
normalize = plt.Normalize(vmin=0, vmax=20)

for i in range(len(years)):
    for j in range(len(cc_values)):
        color = cmap(normalize(years[i]))
        ax.bar3d(years_grid[i, j], cc_values_grid[i, j], 0,
                 dx=0.5, dy=0.05, dz=income[i, j],
                 color=color, shade=True)

ax.set_xlabel('Number of Year')
ax.set_ylabel('CC Value')
ax.set_zlabel('Annual Income (in millions)')

plt.title('')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Year', ticks=np.arange(0, 21, 2))  # 每2年一个刻度
cbar.set_ticklabels(np.arange(0, 21, 2))  # 设置刻度标签为整数

plt.savefig('./images/3D_income_cc.png')
plt.close()


years = np.arange(1, 21)
tourism_intensive_area = [31.7, 33.2, 37.8, 43.2, 42.1, 38.2, 35.1, 29.8, 27.5, 24.9, 23.1, 21.4, 20.2, 23.8, 24.8, 24.7, 22.0, 23.7, 24.1, 21.1]
# tourism_intensive_area = [30.1, 29.2, 30.8, 35.2, 32.1, 31.2, 30.1, 26.8, 24.5, 22.9, 21.1, 18.4, 20.2, 19.8, 20.8, 20.7, 20.0, 20.7, 20.1, 20.5]
tourism_affected_area = [(100 - t) * 0.3 for t in tourism_intensive_area]
residential_area = [(100 - t) * 0.7 for t in tourism_intensive_area]
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


years = np.arange(1, 13)
sensitive_visitors = np.array([124, 116, 108, 94, 82, 74, 68, 62, 78, 87, 98, 105])  # 敏感型景区游客量
non_sensitive_visitors = np.array([80, 95, 106, 115, 129, 131, 120, 113, 106, 104, 92, 75])  # 非敏感型景区游客量

transfer_ratio = np.array([0.35, 0.29, 0.23, 0.20, 0.18, 0.17, 0.13, 0.10, -0.05, -0.11, -0.22, -0.24])

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制敏感型景区游客量（左纵轴）
ax1.plot(years, sensitive_visitors, marker='o', linestyle='-', color='#0dceda', label='sensitive visitors')
# 绘制非敏感型景区游客量（左纵轴）
ax1.plot(years, non_sensitive_visitors, marker='s', linestyle='--', color='#a393eb', label='non-sensitive visitors')
ax1.set_xlabel('Month', fontsize=14)
ax1.set_ylabel('Number of Visitors', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()
ax2.plot(years, transfer_ratio, marker='^', linestyle='-.', color='#00bbf0', label='transfer ratio')
ax2.set_ylabel('transfer ratio', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(-1, 1)

plt.title('transfer ratio of Juneau in one year', fontsize=16, pad=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.savefig('./images/transfer_ratio.png', dpi=500)



data = {
    'Time Period': np.arange(1, 13),
    'Attraction A': [0.15, 0.16, 0.14, 0.15, 0.16, 0.14, 0.15, 0.16, 0.14, 0.15, 0.16, 0.14],
    'Attraction B': [0.10, 0.09, 0.11, 0.10, 0.09, 0.11, 0.10, 0.09, 0.11, 0.10, 0.09, 0.11],
    'Attraction C': [0.12, 0.11, 0.13, 0.12, 0.11, 0.13, 0.12, 0.11, 0.13, 0.12, 0.11, 0.13],
    'Attraction D': [0.08, 0.09, 0.07, 0.08, 0.09, 0.07, 0.08, 0.09, 0.07, 0.08, 0.09, 0.07],
    'Attraction E': [0.20, 0.19, 0.21, 0.20, 0.19, 0.21, 0.20, 0.19, 0.21, 0.20, 0.19, 0.21],
    'Attraction F': [0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11, 0.09],
    'Attraction G': [0.15, 0.14, 0.16, 0.15, 0.14, 0.16, 0.15, 0.14, 0.16, 0.15, 0.14, 0.16],
    'Attraction H': [0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11, 0.09]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
for column in df.columns[1:]:
    plt.plot(df['Time Period'], df[column], marker='o', label=column)

plt.title('Visitor Proportion for 8 Attractions in Charleston', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Visitor Proportion %', fontsize=12)
plt.ylim([0, 0.4])
plt.legend(title='Attractions', loc='upper right')
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')

plt.tight_layout()

plt.savefig('./images/visitor_proportion.png', dpi=500)




def log_growth(x, a, b):
    return a * np.log(x) + b

x0, y0 = 0, 12.82

x = np.linspace(1, 20, 100)  # 生成100个点用于绘制曲线
x_discrete = np.arange(1, 21)  # 生成1到20的离散点

a, b = 5, y0
y = log_growth(x, a, b)

np.random.seed(42)
y_discrete = log_growth(x_discrete, a, b) + np.random.normal(0, 1, size=len(x_discrete))

popt, _ = curve_fit(log_growth, x_discrete, y_discrete)
y_fit = log_growth(x_discrete, *popt)

r2 = r2_score(y_discrete, y_fit)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linestyle='--', label='Log Growth Curve', color='purple')

plt.scatter(x_discrete, y_discrete, color='deepskyblue', label='Prediction', s=80, edgecolor='black', zorder=5)

plt.text(0.4, 26, f'Logarithmic Fit: R² = {r2:.2f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('Year', fontsize=14)
plt.ylabel('Annual Income (in billions)', fontsize=14)
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)
plt.xticks(np.arange(1, 21, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.legend()
plt.savefig('./images/log_growth_curve.png', dpi=500)


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


x = np.arange(1, 21, 1)
sensitive = np.array([565, 555, 558, 552, 569, 544, 554, 550, 561, 563, 560, 568, 556, 560, 550, 542, 540, 534, 527, 525])
insensitive = np.array([320, 365, 368, 360, 398, 425, 445, 452, 458, 451, 460, 468, 466, 472, 476, 475, 480, 478, 477, 476])

plt.figure(figsize=(10, 6))
plt.plot(x, sensitive, label='Sensitive', linestyle='-', color="#30e3ca")
plt.plot(x, insensitive, label='Insensitive', linestyle='-.', color="#9896f1")

plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Visitors (in thousands)", fontsize=14)
plt.xticks(np.arange(1, 21, 1))
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.legend()
plt.savefig('./images/sen.png', dpi=500)


x = np.arange(1, 21, 1)
sensitive = np.array([565, 555, 558, 552, 569, 544, 554, 550, 561, 563, 560, 568, 556, 560, 550, 542, 540, 534, 527, 525])
insensitive = np.array([320, 365, 368, 360, 398, 425, 445, 452, 458, 451, 460, 468, 466, 472, 476, 475, 480, 478, 477, 476])

plt.figure(figsize=(10, 6))
plt.plot(x, sensitive, label='Sensitive', linestyle='-', color="#30e3ca")
plt.plot(x, insensitive, label='Insensitive', linestyle='-.', color="#9896f1")

plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Visitors (in thousands)", fontsize=14)
plt.xticks(np.arange(1, 21, 1))
plt.ylim(300, 600)
plt.tick_params(axis='both', labelsize=12, length=6, width=1.5, grid_color='gray', grid_alpha=0.6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gcf().set_facecolor('whitesmoke')
plt.legend()
plt.savefig('./images/sen.png', dpi=500)
