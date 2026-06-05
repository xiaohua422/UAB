import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.patches as mpatches


class SpineAnalysisReviewer:
    """
    Spine Curvature Statistical Report Reviewer Analysis Class
    """

    def __init__(self, input_path, output_dir):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 全局字体 Times New Roman
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        self.report_data = self.parse_report()
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def parse_report(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        data = {}
        total_match = re.search(r"Total samples: (\d+)", content)
        data['total_samples'] = int(total_match.group(1)) if total_match else 0

        segment_patterns = {
            'L1-L2': r"L1-L2 Angle Statistics.*?Mean: ([\d.]+).*?Std: ([\d.]+).*?Range: ([\d.]+)° - ([\d.]+)°",
            'L2-L3': r"L2-L3 Angle Statistics.*?Mean: ([\d.]+).*?Std: ([\d.]+).*?Range: ([\d.]+)° - ([\d.]+)°",
            'L3-L4': r"L3-L4 Angle Statistics.*?Mean: ([\d.]+).*?Std: ([\d.]+).*?Range: ([\d.]+)° - ([\d.]+)°",
            'L4-L5': r"L4-L5 Angle Statistics.*?Mean: ([\d.]+).*?Std: ([\d.]+).*?Range: ([\d.]+)° - ([\d.]+)°",
        }

        data['segments'] = {}
        for segment, pattern in segment_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                data['segments'][segment] = {
                    'mean': float(match.group(1)),
                    'std': float(match.group(2)),
                    'min': float(match.group(3)),
                    'max': float(match.group(4))
                }

        ep_match = re.search(
            r"Endplate Angle Statistics.*?Mean: ([\d.]+).*?Std: ([\d.]+).*?Range: ([\d.]+)° - ([\d.]+)°", content,
            re.DOTALL)
        if ep_match:
            data['endplate'] = {
                'mean': float(ep_match.group(1)),
                'std': float(ep_match.group(2)),
                'min': float(ep_match.group(3)),
                'max': float(ep_match.group(4))
            }

        dist_pattern = r"Max Segment Distribution:(.*?)(?=\n\n|\Z)"
        dist_match = re.search(dist_pattern, content, re.DOTALL)
        if dist_match:
            dist_lines = dist_match.group(1).strip().split('\n')
            data['max_segment_dist'] = {}
            for line in dist_lines:
                if 'samples' in line:
                    seg_match = re.search(r"(\S+):\s+(\d+)\s+samples\s+\(([\d.]+)%\)", line)
                    if seg_match:
                        data['max_segment_dist'][seg_match.group(1)] = {
                            'count': int(seg_match.group(2)),
                            'percentage': float(seg_match.group(3))
                        }

        success_match = re.search(r"Processing Success Rate: (\d+)/(\d+)\s+\(([\d.]+)%\)", content)
        if success_match:
            data['success_rate'] = {
                'success': int(success_match.group(1)),
                'total': int(success_match.group(2)),
                'percentage': float(success_match.group(3))
            }

        return data

    def perform_statistical_analysis(self):
        print("=" * 80)
        print("Spine Curvature Statistical Report - Reviewer Analysis")
        print("=" * 80)

        print("\n1. Data Quality Assessment")
        print("-" * 40)
        if 'success_rate' in self.report_data:
            sr = self.report_data['success_rate']
            print(f"✓ Processing Success Rate: {sr['percentage']}% ({sr['success']}/{sr['total']})")

        print("\n2. Descriptive Statistical Analysis")
        print("-" * 40)

        segments_data = []
        for seg_name, seg_stats in self.report_data['segments'].items():
            segments_data.append({
                'Segment': seg_name,
                'Mean (°)': seg_stats['mean'],
                'SD (°)': seg_stats['std'],
                'CV': seg_stats['std'] / seg_stats['mean'] if seg_stats['mean'] > 0 else 0,
                'Min (°)': seg_stats['min'],
                'Max (°)': seg_stats['max'],
                'Range (°)': seg_stats['max'] - seg_stats['min']
            })

        df_segments = pd.DataFrame(segments_data)
        print(df_segments.to_string(index=False))

        print("\n3. Lumbar Angle Trend Analysis")
        print("-" * 40)
        segment_order = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5']
        angles = [self.report_data['segments'][seg]['mean'] for seg in segment_order]
        from scipy.stats import pearsonr
        x = list(range(len(segment_order)))
        r, p_value = pearsonr(x, angles)
        print(f"  Correlation coefficient (r): {r:.3f}, p-value: {p_value:.4f}")

        print("\n4. Data Variability Analysis")
        print("-" * 40)
        cv_values = df_segments['CV']
        max_cv_idx = cv_values.idxmax()
        min_cv_idx = cv_values.idxmin()
        print(f"  Highest variability: {df_segments.loc[max_cv_idx, 'Segment']}")
        print(f"  Lowest variability: {df_segments.loc[min_cv_idx, 'Segment']}")

        return df_segments

    def create_segment_angle_distribution_chart(self, df_segments, plots_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(df_segments))
        width = 0.6
        norm = plt.Normalize(df_segments['Mean (°)'].min(), df_segments['Mean (°)'].max())
        cmap = plt.cm.viridis
        colors = [cmap(norm(val)) for val in df_segments['Mean (°)']]

        bars = ax.bar(x_pos, df_segments['Mean (°)'],
                      width=width, yerr=df_segments['SD (°)'], capsize=8,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5},
                      color=colors, edgecolor='black', linewidth=1, alpha=0.85)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_segments['Segment'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Angle (°)', fontsize=13, fontweight='bold')
        ax.set_title('Lumbar Segment Angle Distribution', fontsize=15, fontweight='bold', pad=15)

        for i, (bar, mean_val, std_val) in enumerate(zip(bars, df_segments['Mean (°)'], df_segments['SD (°)'])):
            height = bar.get_height()
            label_y = height + std_val + 1.0
            ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                    f'{mean_val:.1f}° ± {std_val:.1f}°',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

        overall_mean = df_segments['Mean (°)'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Overall Mean: {overall_mean:.1f}°')
        ax.legend(fontsize=10)
        plt.tight_layout(pad=2.0)

        base_filename = plots_dir / "segment_angle_distribution"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_angle_trend_chart(self, df_segments, plots_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df_segments['Segment'], df_segments['Mean (°)'],
                marker='o', markersize=10, linewidth=2.5,
                color='#2E86AB', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor='#2E86AB', zorder=5)

        ax.fill_between(df_segments['Segment'],
                        df_segments['Mean (°)'] - df_segments['SD (°)'],
                        df_segments['Mean (°)'] + df_segments['SD (°)'],
                        alpha=0.25, color='#2E86AB', label='Mean ± SD')

        ax.set_xlabel('Lumbar Segment', fontsize=13, fontweight='bold')
        ax.set_ylabel('Angle (°)', fontsize=13, fontweight='bold')
        ax.set_title('Lumbar Angle Trend Across Segments', fontsize=15, fontweight='bold', pad=15)

        for i, mean_val in enumerate(df_segments['Mean (°)']):
            ax.text(i, mean_val + 1.0, f'{mean_val:.1f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout(pad=2.0)
        base_filename = plots_dir / "lumbar_angle_trend"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_variability_heatmap(self, df_segments, plots_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})
        cv_matrix = df_segments[['Segment', 'CV']].set_index('Segment')
        im = ax1.imshow(cv_matrix.values.reshape(1, -1), cmap='YlOrRd', aspect='auto', vmin=0)
        ax1.set_xticks(range(len(cv_matrix)))
        ax1.set_xticklabels(cv_matrix.index, fontsize=11, fontweight='bold')
        ax1.set_title('Coefficient of Variation (CV)', fontsize=14, fontweight='bold')

        y_pos = np.arange(len(df_segments))
        colors = plt.cm.YlOrRd(df_segments['CV'] / df_segments['CV'].max())
        ax2.barh(y_pos, df_segments['CV'], color=colors, edgecolor='black', linewidth=1, height=0.6)
        ax2.set_yticklabels(df_segments['Segment'], fontsize=11, fontweight='bold')

        plt.tight_layout(pad=2.0)
        base_filename = plots_dir / "variability_heatmap"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_mean_std_scatter_plot(self, df_segments, plots_dir):
        fig, ax = plt.subplots(figsize=(12, 9))
        x = df_segments['Mean (°)']
        y = df_segments['SD (°)']
        cv_values = df_segments['CV']
        ranges = df_segments['Range (°)']
        scatter_size = 300 + ranges * 30
        norm = plt.Normalize(cv_values.min(), cv_values.max())
        ax.scatter(x, y, s=scatter_size, c=cv_values, cmap=plt.cm.viridis, alpha=0.85, edgecolors='black', linewidths=1.5)

        ax.set_xlabel('Mean Angle (°)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Standard Deviation (°)', fontsize=13, fontweight='bold')
        ax.set_title('Mean Angle vs Standard Deviation', fontsize=15, fontweight='bold')
        plt.tight_layout(pad=2.0)

        base_filename = plots_dir / "mean_std_scatter_plot"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_max_segment_distribution_chart(self, df_segments, plots_dir):
        if 'max_segment_dist' not in self.report_data:
            return
        dist_data = self.report_data['max_segment_dist']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        labels = list(dist_data.keys())
        sizes = [dist_data[label]['percentage'] for label in labels]
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=plt.cm.Set3(np.linspace(0,1,len(labels))))
        ax2.barh(range(len(labels)), sizes)
        plt.tight_layout(pad=2.0)

        base_filename = plots_dir / "max_segment_distribution"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_angle_range_comparison_chart(self, df_segments, plots_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
        ax1.barh(range(len(df_segments)), df_segments['Range (°)'])
        ax2.scatter(df_segments['Range (°)'], df_segments['CV'])
        plt.tight_layout(pad=2.0)
        base_filename = plots_dir / "angle_range_comparison"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def create_endplate_comparison_chart(self, df_segments, plots_dir):
        if 'endplate' not in self.report_data:
            return
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,7))
        plt.tight_layout(pad=2.0)
        base_filename = plots_dir / "endplate_comparison"
        plt.savefig(f"{base_filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {base_filename}.svg")

    def _calculate_label_positions(self, x, y):
        return [(xi + x.std()*0.25, yi + y.std()*0.25) for xi, yi in zip(x, y)]

    def create_comprehensive_visualizations(self, df_segments):
        print("\nGenerating SVG Visualizations...")
        plots_dir = self.output_dir / "analysis_plots"
        plots_dir.mkdir(exist_ok=True)

        self.create_segment_angle_distribution_chart(df_segments, plots_dir)
        self.create_angle_trend_chart(df_segments, plots_dir)
        self.create_variability_heatmap(df_segments, plots_dir)
        self.create_mean_std_scatter_plot(df_segments, plots_dir)
        self.create_max_segment_distribution_chart(df_segments, plots_dir)
        self.create_angle_range_comparison_chart(df_segments, plots_dir)
        self.create_endplate_comparison_chart(df_segments, plots_dir)

    def generate_report_summary(self, df_segments):
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({"status": "completed"}, f)

    def export_analysis_results(self, df_segments):
        df_segments.to_csv(self.output_dir / "results.csv", index=False)

    def run_full_analysis(self):
        df_segments = self.perform_statistical_analysis()
        self.create_comprehensive_visualizations(df_segments)
        self.generate_report_summary(df_segments)
        self.export_analysis_results(df_segments)
        print("\n✅ All figures exported as SVG only!")

def main():
    input_path = r"D:\unet_test\Deeplabv3+\Deeplabv3_plus_ours\deeplabv3-plus-pytorch-main-ours\every_amace_cobb_results\statistical_report.txt"
    output_dir = r"D:\unet_test\Deeplabv3+\Deeplabv3_plus_ours\deeplabv3-plus-pytorch-main-ours\every_amace_cobb_results\every_amace_cobb_analysis"
    analyzer = SpineAnalysisReviewer(input_path, output_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()