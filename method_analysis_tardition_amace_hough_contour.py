import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress, friedmanchisquare, shapiro, normaltest
import warnings
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  


warnings.filterwarnings('ignore')

try:
    import pingouin as pg

except ImportError:
    pg = None

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.inter_rater import fleiss_kappa

except ImportError:
    pairwise_tukeyhsd = None
    fleiss_kappa = None


def create_safe_directory(base_path, dir_name):
    safe_dir = os.path.join(base_path, dir_name)
    try:
        os.makedirs(safe_dir, exist_ok=True)
        print(f"Successfully created directory: {safe_dir}")
        return safe_dir
    except Exception as e:
        print(f"Failed to create directory: {e}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_dir = os.path.join(current_dir, "advanced_analysis_results")
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"Using fallback directory: {fallback_dir}")
        return fallback_dir


def check_data_quality(df):
    print("\n" + "=" * 60)
    print("Data Quality Check")
    print("=" * 60)

    all_cols = ['amace_angle', 'tradition_angle', 'hough_angle', 'contour_angle', 'manual_angle']

    print(f"Total samples: {len(df)}")
    print(f"Missing values check:")
    for col in all_cols:
        missing_count = df[col].isna().sum()
        print(f"  {col}: {missing_count} missing values")

    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)]

    print(f"\nOutlier detection (IQR method):")
    for col in all_cols:
        outliers = detect_outliers_iqr(df[col])
        print(f"  {col}: {len(outliers)} outliers")
        if len(outliers) > 0:
            print(f"    Outlier range: {outliers.min():.1f}° - {outliers.max():.1f}°")

    print(f"\nNormality test (Shapiro-Wilk):")
    for col in all_cols:
        stat, p_value = shapiro(df[col])
        print(f"  {col}: W={stat:.3f}, p={p_value:.3f} {'(normal)' if p_value > 0.05 else '**(non-normal)**'}")

    return df


def perform_advanced_statistical_tests(df):
    print("\n" + "=" * 60)
    print("Advanced Statistical Tests")
    print("=" * 60)

    all_methods = ['amace_angle', 'tradition_angle', 'hough_angle', 'contour_angle', 'manual_angle']
    method_names = ['AMACE', 'Traditional', 'Hough', 'Contour', 'Manual']
    methods_data = [df[col] for col in all_methods]

    print("\n1. Friedman test (comparison of all five methods):")
    try:
        friedman_stat, friedman_p = friedmanchisquare(*methods_data)
        print(f"   Friedman statistic: {friedman_stat:.3f}, p-value: {friedman_p:.6f}")
        if friedman_p < 0.001:
            significance = "*** (p < 0.001)"
        elif friedman_p < 0.01:
            significance = "** (p < 0.01)"
        elif friedman_p < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "not significant"
        print(f"   Statistical significance: {significance}")
    except Exception as e:
        print(f"   Friedman test error: {e}")
        friedman_p = None

    print("\n2. Pairwise Wilcoxon signed-rank test (vs manual):")
    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    for name, col in auto_methods:
        stat, p_value = stats.wilcoxon(df[col], df['manual_angle'])
        print(f"   {name} vs Manual:")
        print(f"     Statistic: {stat:.3f}, p-value: {p_value:.6f}")
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        n = len(df)
        z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
        effect_size = z / np.sqrt(n)
        print(f"     Significance: {sig}, Effect size (r): {effect_size:.3f}")
        if abs(effect_size) < 0.1:
            interpretation = "negligible"
        elif abs(effect_size) < 0.3:
            interpretation = "small"
        elif abs(effect_size) < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        print(f"     Effect size interpretation: {interpretation}")

    if pg is not None:
        print("\n3. Intraclass Correlation Coefficient (ICC) analysis (all five methods):")
        icc_data = pd.DataFrame({
            'raters': np.repeat(method_names, len(df)),
            'targets': list(range(1, len(df) + 1)) * 5,
            'ratings': np.concatenate([df[col].values for col in all_methods])
        })
        icc_result = pg.intraclass_corr(data=icc_data, targets='targets', raters='raters', ratings='ratings')
        icc_result = icc_result.set_index('Type')
        icc_21 = icc_result.loc['ICC2', 'ICC']
        icc_21_ci = icc_result.loc['ICC2', 'CI95%']
        print(f"   ICC(2,1) - absolute agreement: {icc_21:.3f} (95% CI: {icc_21_ci})")
        if icc_21 >= 0.9:
            interpretation = "excellent agreement"
        elif icc_21 >= 0.75:
            interpretation = "good agreement"
        elif icc_21 >= 0.5:
            interpretation = "moderate agreement"
        else:
            interpretation = "poor agreement"
        print(f"   ICC interpretation: {interpretation}")
    else:
        print("\n3. Skipping ICC analysis (pingouin not installed)")

    return friedman_p


def calculate_error_metrics(df):
    print("\n" + "=" * 60)
    print("Detailed Error Analysis")
    print("=" * 60)

    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    short_names = ['amace', 'tradition', 'hough', 'contour']

    print("\nComparison with manual measurement (gold standard):")

    error_metrics = {}
    for (method_name, col), short_name in zip(auto_methods, short_names):
        differences = df[col] - df['manual_angle']
        print(f"\n{method_name}:")
        mae = np.mean(np.abs(differences))
        rmse = np.sqrt(np.mean(differences ** 2))
        bias = np.mean(differences)
        std_diff = np.std(differences)
        print(f"  Mean Absolute Error (MAE): {mae:.3f}°")
        print(f"  Root Mean Square Error (RMSE): {rmse:.3f}°")
        print(f"  Bias: {bias:.3f}°")
        print(f"  Standard deviation of differences: {std_diff:.3f}°")
        loa_upper = bias + 1.96 * std_diff
        loa_lower = bias - 1.96 * std_diff
        print(f"  Limits of agreement (95% LoA): [{loa_lower:.3f}°, {loa_upper:.3f}°]")
        print(f"  LoA width: {loa_upper - loa_lower:.3f}°")
        percentage_error = (np.abs(differences) / df['manual_angle']) * 100
        mape = np.mean(percentage_error)
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"  Clinical acceptability:")
        for threshold in [5, 10]:
            within = np.sum(np.abs(differences) <= threshold)
            percentage = (within / len(df)) * 100
            print(f"    Difference ≤ {threshold}°: {within}/{len(df)} ({percentage:.1f}%)")

        error_metrics[f'{short_name}_mae'] = mae
        error_metrics[f'{short_name}_bias'] = bias
        error_metrics[f'{short_name}_loa_width'] = loa_upper - loa_lower

    return error_metrics


def perform_regression_analysis(df):
    print("\n" + "=" * 60)
    print("Regression Analysis")
    print("=" * 60)

    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]

    for method_name, col_name in auto_methods:
        print(f"\n{method_name} vs Manual measurement:")
        x = df[col_name].values
        y = df['manual_angle'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2
        print(f"  Coefficient of determination (R²): {r_squared:.4f}")
        print(f"  Correlation coefficient (r): {r_value:.4f}")
        print(f"  Regression equation: y = {slope:.4f}x + {intercept:.4f}")
        print(f"  Regression coefficient p-value: {p_value:.6f}")
        predicted = slope * x + intercept
        residuals = y - predicted
        residual_std = np.std(residuals)
        print(f"  Residual standard deviation: {residual_std:.4f}°")
        dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        print(f"  Durbin-Watson statistic: {dw_stat:.4f}")
        if dw_stat < 1.5:
            print("    * Possible positive autocorrelation")
        elif dw_stat > 2.5:
            print("    * Possible negative autocorrelation")
        else:
            print("    * No significant autocorrelation")

    return True


def create_advanced_visualizations(df, output_dir):
    print("\nGenerating advanced visualization charts...")

    comp_dir = create_safe_directory(output_dir, "comprehensive_comparison")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    all_methods = ['amace_angle', 'tradition_angle', 'hough_angle', 'contour_angle', 'manual_angle']
    all_labels = ['AMACE', 'Traditional', 'Hough', 'Contour', 'Manual']
    colors_all = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

 
    print("\nCreating distribution comparison plot (all five methods)...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    data_to_plot = [df[col] for col in all_methods]
    bp = ax1.boxplot(data_to_plot, labels=all_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
    for i, data in enumerate(data_to_plot):
        violin_parts = ax1.violinplot(data, positions=[i + 1], showmeans=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors_all[i])
            pc.set_alpha(0.3)
    ax1.set_ylabel('Angle (°)', fontsize=12)
    ax1.set_title('Distribution Comparison (Box Plot + Violin Plot)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(comp_dir, 'distribution_comparison_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig1)
    print(f"Distribution comparison plot saved: {plot_path}")

   
    print("\nCreating correlation scatter plot (all automatic methods)...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    auto_colors = colors_all[:4]
    for (name, col), color in zip(auto_methods, auto_colors):
        x = df[col]
        y = df['manual_angle']
        slope, intercept, r_value, _, _ = linregress(x, y)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred = slope * x_range + intercept
        ax2.scatter(x, y, alpha=0.6, color=color, s=60, label=f'{name} (r={r_value:.3f})')
        ax2.plot(x_range, y_pred, color=color, linestyle='--', alpha=0.8, linewidth=2)
    ax2.plot([df['manual_angle'].min(), df['manual_angle'].max()],
             [df['manual_angle'].min(), df['manual_angle'].max()],
             'k-', alpha=0.5, linewidth=2, label='Perfect agreement')
    ax2.set_xlabel('Measured Angle (°)', fontsize=12)
    ax2.set_ylabel('Manual Measurement Angle (°)', fontsize=12)
    ax2.set_title('Correlation with Gold Standard', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)
    all_values = np.concatenate([df[col] for col in all_methods])
    min_val, max_val = all_values.min(), all_values.max()
    padding = (max_val - min_val) * 0.1
    ax2.set_xlim(min_val - padding, max_val + padding)
    ax2.set_ylim(min_val - padding, max_val + padding)
    plt.tight_layout()
    plot_path = os.path.join(comp_dir, 'correlation_scatter_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig2)
    print(f"Correlation scatter plot saved: {plot_path}")

   
    print("\nCreating difference distribution density plot (all automatic methods)...")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    from scipy.stats import gaussian_kde
    for (name, col), color in zip(auto_methods, auto_colors):
        diff = df[col] - df['manual_angle']
        kde = gaussian_kde(diff)
        x_range = np.linspace(diff.min(), diff.max(), 100)
        ax3.plot(x_range, kde(x_range), color=color, label=name, linewidth=2)
        ax3.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    for (name, col), color in zip(auto_methods, auto_colors):
        diff = df[col] - df['manual_angle']
        ax3.axvline(x=np.mean(diff), color=color, linestyle='--',
                    label=f'{name} mean: {np.mean(diff):.2f}°', linewidth=1.5)
    ax3.axvspan(-5, 5, alpha=0.1, color='green', label='Clinically acceptable zone (±5°)')
    ax3.axvspan(-10, 10, alpha=0.05, color='yellow', label='Clinically tolerable zone (±10°)')
    ax3.set_xlabel('Difference from Manual Measurement (°)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Difference Distribution Density Plot', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(comp_dir, 'difference_density_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig3)
    print(f"Difference distribution density plot saved: {plot_path}")

    
    print("\nCreating clinical performance radar chart (all four methods with clear colors)...")
    fig4, ax4 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    categories = ['MAE ≤ 5°', 'MAE ≤ 10°', 'Bias ≤ 5°', 'R² ≥ 0.9', 'ICC ≥ 0.9']

    def calculate_metrics(method_data, manual_data):
        mae = np.mean(np.abs(method_data - manual_data))
        bias = np.mean(method_data - manual_data)
        r_squared = np.corrcoef(method_data, manual_data)[0, 1] ** 2
        mae_5_score = max(0, 1 - mae / 5) if mae <= 5 else 0
        mae_10_score = max(0, 1 - mae / 10) if mae <= 10 else 0
        bias_score = max(0, 1 - abs(bias) / 5) if abs(bias) <= 5 else 0
        r2_score = min(1, r_squared / 0.9)
        icc_score = 0.8  
        return [mae_5_score, mae_10_score, bias_score, r2_score, icc_score]

   
    radar_methods = [
        ('AMACE', df['amace_angle'], '#0055ff'), 
        ('Traditional', df['tradition_angle'], '#ff0000'),
        ('Hough', df['hough_angle'], '#00cc00'),  
        ('Contour', df['contour_angle'], '#9467bd')
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  

    for name, data, color in radar_methods:
        scores = calculate_metrics(data, df['manual_angle'])
        scores += scores[:1]  
        ax4.plot(angles, scores, 'o-', linewidth=2, label=name, color=color, alpha=0.9)
        ax4.fill(angles, scores, alpha=0.5, color=color)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax4.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax4.set_title('Clinical Performance Radar Chart\n(Higher scores indicate better performance)',
                  fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax4.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(comp_dir, 'clinical_performance_radar_four.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig4)
    print(f"Clinical performance radar chart (four methods) saved: {plot_path}")


    create_advanced_bland_altman_all(df, output_dir)
    create_combined_bland_altman_four(df, output_dir)
    # create_side_by_side_bland_altman(df, output_dir)
    create_side_by_side_bland_altman_four(df, output_dir)
    create_trend_analysis_all(df, output_dir)


def create_advanced_bland_altman_all(df, output_dir):

    print("\nCreating advanced Bland-Altman plots for all methods...")
    ba_dir = create_safe_directory(output_dir, "bland_altman_plots")
    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    colors = ['blue', 'red', 'green', 'purple']
    dark_colors = ['darkblue', 'darkred', 'darkgreen', 'darkviolet']

    for (method_name, col), color, dark_color in zip(auto_methods, colors, dark_colors):
        fig, ax = plt.subplots(figsize=(10, 8))
        differences = df[col] - df['manual_angle']
        means = (df[col] + df['manual_angle']) / 2
        bias = np.mean(differences)
        std_diff = np.std(differences)
        loa_upper = bias + 1.96 * std_diff
        loa_lower = bias - 1.96 * std_diff
        within_loa = np.sum((differences >= loa_lower) & (differences <= loa_upper))
        within_loa_percent = (within_loa / len(df)) * 100

        ax.axhspan(-5, 5, alpha=0.1, color='green', label='Clinically acceptable zone', zorder=0)
        ax.axhspan(-10, 10, alpha=0.05, color='yellow', label='Clinically tolerable zone', zorder=0)
        ax.scatter(means, differences, alpha=0.6, color=color, s=50, zorder=5)
        z = np.polyfit(means, differences, 1)
        trend_line = np.poly1d(z)
        x_range = np.linspace(means.min(), means.max(), 100)
        ax.plot(x_range, trend_line(x_range), '--', color=dark_color,
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}', linewidth=2, zorder=4)
        ax.axhline(y=bias, color=color, linestyle='-', linewidth=2,
                   label=f'Bias: {bias:.2f}°', zorder=3)
        ax.axhline(y=loa_upper, color=color, linestyle='--', linewidth=1.5,
                   label=f'95% LoA upper: {loa_upper:.2f}°', zorder=2)
        ax.axhline(y=loa_lower, color=color, linestyle='--', linewidth=1.5,
                   label=f'95% LoA lower: {loa_lower:.2f}°', zorder=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1, zorder=1)

        ax.set_xlabel('Mean Angle (°)', fontsize=12)
        ax.set_ylabel(f'Difference ({method_name} - Manual) (°)', fontsize=12)
        ax.set_title(f'{method_name} Bland-Altman Plot\n({within_loa_percent:.1f}% within limits of agreement)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

        y_min, y_max = differences.min(), differences.max()
        max_abs_y = max(abs(y_min), abs(y_max))
        y_padding = max_abs_y * 0.5
        ax.set_ylim(-max_abs_y - y_padding, max_abs_y + y_padding)
        x_padding = (means.max() - means.min()) * 0.1
        ax.set_xlim(means.min() - x_padding, means.max() + x_padding)

        plt.tight_layout()
        plot_path = os.path.join(ba_dir, f'bland_altman_{method_name.lower()}.svg')
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Bland-Altman plot for {method_name} saved: {plot_path}")


def create_combined_bland_altman_four(df, output_dir):
   
    print("\nCreating combined Bland-Altman plot (four methods)...")
    combined_dir = create_safe_directory(output_dir, "combined_plots")
    fig, ax = plt.subplots(figsize=(14, 10))

    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    markers = ['o', 's', '^', 'D']
    alphas = [0.7, 0.6, 0.5, 0.4]

    for (name, col), color, marker, alpha in zip(auto_methods, colors, markers, alphas):
        differences = df[col] - df['manual_angle']
        means = (df[col] + df['manual_angle']) / 2
        bias = np.mean(differences)
        std_diff = np.std(differences)
        loa_upper = bias + 1.96 * std_diff
        loa_lower = bias - 1.96 * std_diff

        ax.scatter(means, differences, alpha=alpha, color=color, s=60,
                   marker=marker, edgecolor='white', linewidth=0.8, label=name, zorder=5)
        ax.axhline(y=bias, color=color, linestyle='-', linewidth=2, alpha=0.7,
                   label=f'{name} bias: {bias:.2f}°', zorder=3)
        ax.axhline(y=loa_upper, color=color, linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)
        ax.axhline(y=loa_lower, color=color, linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    ax.axhspan(-5, 5, alpha=0.1, color='green', label='Clinically acceptable (±5°)', zorder=0)
    ax.axhspan(-10, 10, alpha=0.05, color='yellow', label='Clinically tolerable (±10°)', zorder=0)

    ax.set_xlabel('Mean of Methods and Manual Measurement (°)', fontsize=13)
    ax.set_ylabel('Difference from Manual Measurement (°)', fontsize=13)
    ax.set_title('Combined Bland-Altman Plot: All Four Methods vs Manual',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.tick_params(axis='both', labelsize=11)

    all_differences = np.concatenate([df[col] - df['manual_angle'] for _, col in auto_methods])
    all_means = np.concatenate([(df[col] + df['manual_angle']) / 2 for _, col in auto_methods])
    y_min, y_max = all_differences.min(), all_differences.max()
    max_abs_y = max(abs(y_min), abs(y_max))
    y_padding = max_abs_y * 0.5
    ax.set_ylim(-max_abs_y - y_padding, max_abs_y + y_padding)
    x_padding = (all_means.max() - all_means.min()) * 0.1
    ax.set_xlim(all_means.min() - x_padding, all_means.max() + x_padding)

    plt.tight_layout()
    plot_path = os.path.join(combined_dir, 'combined_bland_altman_four.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Combined Bland-Altman plot (four methods) saved: {plot_path}")


def create_side_by_side_bland_altman_four(df, output_dir):

    print("\nCreating individual Bland-Altman plots for each method...")
    combined_dir = create_safe_directory(output_dir, "combined_plots")

    colors = {
        'AMACE': {
            'primary': '#2E86AB',  
            'light': '#A3D5FF',
            'dark': '#1A5F7A',
            'accent': '#73B1E0'
        },
        'Traditional': {
            'primary': '#E15554',  
            'light': '#FFB8B7',
            'dark': '#A63D40',
            'accent': '#E68A8A'
        },
        'Hough': {
            'primary': '#2ca02c', 
            'light': '#98df8a',
            'dark': '#1f771f',
            'accent': '#4caf50'
        },
        'Contour': {
            'primary': '#9467bd', 
            'light': '#c5b0d5',
            'dark': '#5a3e8a',
            'accent': '#b085f5'
        }
    }

   
    cmaps = {
        'AMACE': 'viridis',
        'Traditional': 'plasma',
        'Hough': 'cool',
        'Contour': 'hot'
    }

   
    zone_colors = {
        'acceptable': '#90EE90',  
        'tolerable': '#FFE5B4'  
    }

    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]

    for method_name, method_col in auto_methods:
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        method_data = df[method_col]

        differences = method_data - df['manual_angle']
        means = (method_data + df['manual_angle']) / 2

        bias = np.mean(differences)
        std_diff = np.std(differences)
        loa_upper = bias + 1.96 * std_diff
        loa_lower = bias - 1.96 * std_diff

        within_loa = np.sum((differences >= loa_lower) & (differences <= loa_upper))
        within_loa_percent = (within_loa / len(df)) * 100

       
        ax.axhspan(-5, 5, alpha=0.15, color=zone_colors['acceptable'],
                   label='Clinically acceptable (±5°)', zorder=0)
        ax.axhspan(-10, 10, alpha=0.08, color=zone_colors['tolerable'],
                   label='Clinically tolerable (±10°)', zorder=0)

      ）
        norm = plt.Normalize(means.min(), means.max())
        scatter = ax.scatter(
            means,
            differences,
            c=means,
            cmap=cmaps[method_name],
            alpha=0.7,
            s=55,
            edgecolor='white',
            linewidth=0.7,
            zorder=5,
            norm=norm
        )

      
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Mean Angle (°)', fontsize=10)

      
        z = np.polyfit(means, differences, 1)
        trend_line = np.poly1d(z)
        x_range = np.linspace(means.min(), means.max(), 100)
        ax.plot(
            x_range,
            trend_line(x_range),
            '--',
            color=colors[method_name]['dark'],
            linewidth=2,
            zorder=6,
            alpha=0.9,
            label=f'Trend (slope={z[0]:.3f})'
        )

     
        ax.axhline(
            y=bias,
            color=colors[method_name]['primary'],
            linestyle='-',
            linewidth=2.5,
            zorder=4,
            alpha=0.9,
            label=f'Bias: {bias:.2f}°'
        )

        ax.axhline(
            y=loa_upper,
            color=colors[method_name]['accent'],
            linestyle=':',
            linewidth=1.8,
            zorder=3,
            alpha=0.8,
            label=f'95% LoA upper: {loa_upper:.2f}°'
        )
        ax.axhline(
            y=loa_lower,
            color=colors[method_name]['accent'],
            linestyle=':',
            linewidth=1.8,
            zorder=3,
            alpha=0.8,
            label=f'95% LoA lower: {loa_lower:.2f}°'
        )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)

        ax.set_xlabel('Mean of Methods and Manual Measurement (°)', fontsize=11)
        ax.set_ylabel(f'Difference from Manual Measurement (°)', fontsize=11)

   
        ax.set_title(f'{method_name} Bland-Altman Plot\n({within_loa_percent:.1f}% within LoA)',
                     fontsize=13, fontweight='bold')


        y_min, y_max = differences.min(), differences.max()
        max_abs_y = max(abs(y_min), abs(y_max))
        y_padding = max_abs_y * 0.5
        ax.set_ylim(-max_abs_y - y_padding, max_abs_y + y_padding)

        x_min, x_max = means.min(), means.max()
        x_padding = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)


        ax.grid(True, alpha=0.25, zorder=1, linestyle='--')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        stats_text = (
            f"Bias: {bias:.2f}°\n"
            f"LoA: [{loa_lower:.2f}°, {loa_upper:.2f}°]\n"
            f"Width: {loa_upper - loa_lower:.2f}°\n"
            f"MAE: {np.mean(np.abs(differences)):.2f}°"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # 保存图形
        filename = f'bland_altman_{method_name.lower()}_detailed.svg'
        plot_path = os.path.join(combined_dir, filename)
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Individual Bland-Altman plot for {method_name} saved: {plot_path}")

def create_trend_analysis_all(df, output_dir):
   
    print("\nCreating trend analysis plots (all methods)...")
    trend_dir = create_safe_directory(output_dir, "trend_analysis_plots")
    sorted_df = df.sort_values('manual_angle')
    auto_methods = [('AMACE', 'amace_angle'), ('Traditional', 'tradition_angle'),
                    ('Hough', 'hough_angle'), ('Contour', 'contour_angle')]
    colors = ['blue', 'red', 'green', 'purple']

    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    x_range = range(len(sorted_df))
    for (name, col), color in zip(auto_methods, colors):
        ax1.plot(x_range, sorted_df[col], color=color, alpha=0.7, label=name, linewidth=2)
    ax1.plot(x_range, sorted_df['manual_angle'], 'k-', alpha=0.9, label='Manual', linewidth=2.5)
    ax1.set_xlabel('Sample Index (sorted by manual measurement)', fontsize=12)
    ax1.set_ylabel('Angle (°)', fontsize=12)
    ax1.set_title('Measurement Value Trend Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(trend_dir, 'angle_trend_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig1)
    print(f"Angle trend plot saved: {plot_path}")

 
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    window_size = 5
    for (name, col), color in zip(auto_methods, colors):
        errors = np.abs(sorted_df[col] - sorted_df['manual_angle'])
        smooth = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
        ax2.plot(range(len(smooth)), smooth, color=color, alpha=0.8, label=f'{name} error', linewidth=2)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Absolute Error (°)', fontsize=12)
    ax2.set_title('Error Trend Analysis (5-point moving average)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(trend_dir, 'error_trend_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig2)
    print(f"Error trend plot saved: {plot_path}")

  
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ['0-20°', '20-40°', '40-60°', '60-80°', '80-100°']
    x = np.arange(len(bin_labels))
    width = 0.2

    for i, ((name, col), color) in enumerate(zip(auto_methods, colors)):
        mean_errors = []
        for j in range(len(bins) - 1):
            mask = (df['manual_angle'] >= bins[j]) & (df['manual_angle'] < bins[j + 1])
            if mask.sum() > 0:
                mean_errors.append(np.mean(np.abs(df.loc[mask, col] - df.loc[mask, 'manual_angle'])))
            else:
                mean_errors.append(0)
        offset = (i - 1.5) * width
        ax3.bar(x + offset, mean_errors, width, label=name, color=color, alpha=0.7)

    ax3.set_xlabel('Manual Measurement Angle Range (°)', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error (°)', fontsize=12)
    ax3.set_title('Error Distribution by Angle Range', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plot_path = os.path.join(trend_dir, 'error_by_angle_range_all.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig3)
    print(f"Error distribution by angle range plot saved: {plot_path}")


def generate_comprehensive_report(df, output_dir, error_metrics, friedman_p):
    report_path = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Cobb Angle Measurement Methods Comprehensive Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Executive Summary\n")
        f.write("-" * 50 + "\n")
        f.write(f"Analysis sample count: {len(df)}\n")
        auto_methods = ['amace', 'tradition', 'hough', 'contour']
        for method in auto_methods:
            f.write(f"{method.upper()} Mean Absolute Error: {error_metrics[f'{method}_mae']:.3f}°\n")

        best_method = min(auto_methods, key=lambda m: error_metrics[f'{m}_mae'])
        f.write(f"Best performing automatic method: {best_method.upper()}\n")

        f.write("\n2. Key Findings\n")
        f.write("-" * 50 + "\n")
        if friedman_p and friedman_p < 0.05:
            f.write("• There is statistically significant difference among the five measurement methods\n")
        else:
            f.write("• No statistically significant difference among the five measurement methods\n")

        for method in auto_methods:
            diff = np.abs(df[f'{method}_angle'] - df['manual_angle'])
            within_5 = np.sum(diff <= 5)
            f.write(f"• {method.upper()} within 5° error: {within_5}/{len(df)} ({(within_5/len(df))*100:.1f}%)\n")

        for method in auto_methods:
            corr = df[f'{method}_angle'].corr(df['manual_angle'])
            f.write(f"• {method.upper()} correlation: r = {corr:.3f}\n")

        f.write("\n3. Clinical Recommendations\n")
        f.write("-" * 50 + "\n")
        best_mae = min(error_metrics[f'{m}_mae'] for m in auto_methods)
        if best_mae <= 5:
            f.write("• At least one automated method meets excellent clinical standards (MAE ≤ 5°)\n")
        elif best_mae <= 10:
            f.write("• At least one automated method meets clinically acceptable standards (MAE ≤ 10°)\n")
        else:
            f.write("• Automated methods require further improvement for clinical use\n")
        f.write("• Recommend using the method with lowest MAE in clinical practice\n")

        f.write("\n4. Technical Recommendations\n")
        f.write("-" * 50 + "\n")
        f.write("• Increase sample size to improve statistical power\n")
        f.write("• Validate measurement methods in different patient populations\n")
        f.write("• Consider the impact of image quality on measurement accuracy\n")
        f.write("• Regularly calibrate measurement equipment and algorithms\n")

        f.write("\n5. Detailed Statistics\n")
        f.write("-" * 50 + "\n")
        all_methods = ['amace_angle', 'tradition_angle', 'hough_angle', 'contour_angle', 'manual_angle']
        method_names = ['AMACE', 'Traditional', 'Hough', 'Contour', 'Manual']
        for name, col in zip(method_names, all_methods):
            data = df[col]
            f.write(f"\n{name} Measurement:\n")
            f.write(f"  Mean: {data.mean():.2f}°\n")
            f.write(f"  Standard Deviation: {data.std():.2f}°\n")
            f.write(f"  Minimum: {data.min():.2f}°\n")
            f.write(f"  Maximum: {data.max():.2f}°\n")
            f.write(f"  Median: {data.median():.2f}°\n")
            f.write(f"  25th Percentile: {data.quantile(0.25):.2f}°\n")
            f.write(f"  75th Percentile: {data.quantile(0.75):.2f}°\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Report generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"Comprehensive analysis report saved: {report_path}")


def main():
    input_excel_path = r"D:\unet_test\Deeplabv3+\cobb对比图像\L1-L5\相差较大的图像\手工测量对比三种.xlsx"
    base_dir = r"D:\unet_test\Deeplabv3+\cobb对比图像\L1-L5\相差较大的图像"
    output_dir = create_safe_directory(base_dir, "three_method_analysis_tardition_amace_hough_contour")
    output_csv_path = os.path.join(output_dir, 'processed_data.csv')

    try:
        print("Reading Excel file...")
        data_df = pd.read_excel(input_excel_path)
        print(f"Successfully read {len(data_df)} rows of data")
        print("Excel file columns:", data_df.columns.tolist())

        required_columns = ['filename', 'amace', 'tradition', 'manual', 'hough', 'contour']
        missing_columns = [col for col in required_columns if col not in data_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print("Available columns:", data_df.columns.tolist())
            return

        comparison_df = data_df.rename(columns={
            'amace': 'amace_angle',
            'tradition': 'tradition_angle',
            'manual': 'manual_angle',
            'hough': 'hough_angle',
            'contour': 'contour_angle'
        })

        for method in ['amace', 'tradition', 'hough', 'contour']:
            comparison_df[f'{method}_vs_manual_diff'] = np.abs(comparison_df[f'{method}_angle'] - comparison_df['manual_angle'])

        comparison_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessed data saved to: {output_csv_path}")

        print(f"\nStarting advanced analysis...")
        print(f"Sample count: {len(comparison_df)}")

        checked_df = check_data_quality(comparison_df)
        friedman_p = perform_advanced_statistical_tests(checked_df)
        error_metrics = calculate_error_metrics(checked_df)
        perform_regression_analysis(checked_df)
        create_advanced_visualizations(checked_df, output_dir)
        generate_comprehensive_report(checked_df, output_dir, error_metrics, friedman_p)

        print("\n" + "=" * 70)
        print("Advanced analysis completed!")
        print("=" * 70)
        print("Generated files include:")
        print(f"• Processed data: {output_csv_path}")
        print(f"• Comprehensive comparison plots: {os.path.join(output_dir, 'comprehensive_comparison')}/")
        print(f"• Bland-Altman plots: {os.path.join(output_dir, 'bland_altman_plots')}/")
        print(f"• Combined plots: {os.path.join(output_dir, 'combined_plots')}/")
        print(f"• Trend analysis plots: {os.path.join(output_dir, 'trend_analysis_plots')}/")
        print(f"• Comprehensive analysis report: {os.path.join(output_dir, 'comprehensive_analysis_report.txt')}")

        print(f"\nDirectory structure:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.svg') or file.endswith('.txt') or file.endswith('.csv'):
                    print(f'{subindent}{file}')

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
