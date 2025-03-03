import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats


def calculate_r2(y_true, y_pred):
    # 计算总平方和 (Total Sum of Squares)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    # 计算残差平方和 (Residual Sum of Squares)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # 计算R²
    r2 = 1 - (ss_residual / ss_total)

    return r2


def generate_custom_curve_with_points():
    # Initial point and curve generation
    x_points = [0, 5, 10, 15, 20]
    y_points = [275, 180, 220, 450, 500]

    # Create interpolation function with cubic spline
    curve_func = interp1d(x_points, y_points, kind='cubic')

    # Generate smooth x values for the curve
    x_curve = np.linspace(0, 20, 200)
    y_curve = curve_func(x_curve)

    # Generate points at each integer x-coordinate
    x_scatter = np.arange(0, 21)
    y_scatter = curve_func(x_scatter) + np.random.normal(0, 40, 21)

    # Calculate R-squared
    r_squared = calculate_r2(curve_func(x_scatter), y_scatter)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_curve, y_curve, 'b-', label='Curve')
    plt.scatter(x_scatter, y_scatter, color='red', label='Points')
    plt.title(f'Curve with Points (R² = {r_squared:.4f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(range(0, 21))
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"R-squared: {r_squared}")

    return x_curve, y_curve, x_scatter, y_scatter, r_squared

# Generate and plot the curve with points
generate_custom_curve_with_points()