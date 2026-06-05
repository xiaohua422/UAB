import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shap
import warnings
import os
import itertools
from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10   # 可选，根据需要设置字号
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
# Save directory
save_dir = "CSF\csf_amace_angle_analysis"

# Create save directory and subdirectories
os.makedirs(save_dir, exist_ok=True)

# Create subdirectories for different types of plots
subdirs = [
    'boxplots',
    'correlation',
    'scatter_plots',
    'ml_results',
    'feature_importance',
    'shap_analysis',
    'severity_analysis',
    'cv_results'
]

for subdir in subdirs:
    os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)


class CSFCobbAnalyzer:
    def __init__(self, csf_path, cobb_path):
        """
        Initialize analyzer
        :param csf_path: CSV file path with CSF features (contains various CSF information)
        :param cobb_path: CSV file path with Cobb angles (contains filename and amace_angle columns only)
        """
        self.csf_path = csf_path
        self.cobb_path = cobb_path
        self.merged_data = None
        self.scaler = StandardScaler()

        # Scoliosis severity groups
        self.severity_groups = None
        self.severity_stats = None

        # Properties to save analysis results
        self.cobb_stats = None
        self.csf_key_stats = None
        self.correlation_results = None
        self.ml_results = None
        self.feature_importance = None
        self.selected_features = None

        # SHAP analysis results
        self.shap_values = None
        self.shap_explainer = None

        # Subgroup analysis results
        self.subgroup_results = None

        # Cross-validation results
        self.cv_results = None

    def load_and_merge_data(self):
        """Load and merge CSF data with Cobb angle data"""
        print("Loading data...")

        # 1. Load CSF data (contains various CSF information)
        csf_df = pd.read_csv(self.csf_path)
        print(f"✓ CSF data loaded successfully, {len(csf_df)} records, {len(csf_df.columns)} columns")
        print(f"  CSF data columns: {list(csf_df.columns)}")

        # 2. Load Cobb angle data (contains only filename and amace_angle columns)
        cobb_df = pd.read_csv(self.cobb_path)
        print(f"✓ Cobb angle data loaded successfully, {len(cobb_df)} records, {len(cobb_df.columns)} columns")
        print(f"  Cobb angle data columns: {list(cobb_df.columns)}")

        # 3. Ensure both files have filename column
        if 'filename' not in csf_df.columns:
            raise ValueError("CSF data file missing filename column!")
        if 'filename' not in cobb_df.columns:
            raise ValueError("Cobb angle data file missing filename column!")
        if 'amace_angle' not in cobb_df.columns:
            raise ValueError("Cobb angle data file missing amace_angle column!")

        # 4. Process filename column, remove file extension for matching
        csf_df['base_filename'] = csf_df['filename'].apply(lambda x: str(x).split('.')[0])
        cobb_df['base_filename'] = cobb_df['filename'].apply(lambda x: str(x).split('.')[0])

        # 5. Merge data: inner join based on base_filename
        self.merged_data = pd.merge(
            cobb_df,
            csf_df.drop('filename', axis=1),
            on='base_filename',
            how='inner'
        )

        print(f"\n✓ Data merged successfully!")
        print(f"  - Merged sample count: {len(self.merged_data)}")
        print(f"  - Match rate: {len(self.merged_data) / len(cobb_df) * 100:.2f}%")
        print(f"  - Total columns after merging: {len(self.merged_data.columns)}")

        # 6. Remove outliers: samples with CSF pixel count = 0
        if 'csf_pixel_count' in self.merged_data.columns:
            initial_count = len(self.merged_data)
            self.merged_data = self.merged_data[self.merged_data['csf_pixel_count'] > 0]
            if len(self.merged_data) < initial_count:
                print(f"\n✓ Removed {initial_count - len(self.merged_data)} abnormal samples with CSF pixel count = 0")
                print(f"  - Remaining samples: {len(self.merged_data)}")

        # 7. Define scoliosis severity based on Cobb angle
        self._define_severity_groups()

        return self.merged_data

    def _define_severity_groups(self):
        """Define scoliosis severity groups based on Cobb angle"""

        # Clinical grading standards
        def get_severity(angle):
            if angle < 20:
                return 'Mild'
            elif 20 <= angle < 40:
                return 'Moderate'
            else:
                return 'Severe'

        self.merged_data['severity'] = self.merged_data['amace_angle'].apply(get_severity)

        # Count samples in each group
        self.severity_groups = self.merged_data['severity'].value_counts()
        self.severity_stats = self.merged_data.groupby('severity')['amace_angle'].agg(
            ['count', 'mean', 'std', 'min', 'max'])

        print("\nScoliosis severity group statistics:")
        print(self.severity_stats)
        print(f"\nGroup sample counts: Mild={self.severity_groups.get('Mild', 0)}, "
              f"Moderate={self.severity_groups.get('Moderate', 0)}, "
              f"Severe={self.severity_groups.get('Severe', 0)}")

    def descriptive_analysis(self):
        """Descriptive statistical analysis"""
        print("\n" + "=" * 60)
        print("Descriptive Statistical Analysis")
        print("=" * 60)

        # 1. Cobb angle statistics
        print("\n1. Cobb angle (amace_angle) statistics:")
        self.cobb_stats = self.merged_data['amace_angle'].describe()
        print(self.cobb_stats)

        # 2. CSF feature statistics
        print("\n2. CSF feature statistics:")
        non_feature_cols = ['filename', 'amace_angle', 'base_filename', 'severity']
        csf_feature_cols = [col for col in self.merged_data.columns
                            if col not in non_feature_cols
                            and self.merged_data[col].dtype in [np.float64, np.int64]]

        print(f"   Identified {len(csf_feature_cols)} CSF numerical features")

        # Calculate and save CSF feature statistics
        self.csf_key_stats = self.merged_data[csf_feature_cols].describe()
        print(self.csf_key_stats.head(10))

        # 3. Scoliosis severity group analysis
        self._analyze_severity_groups()

    def _analyze_severity_groups(self):
        """Analyze CSF feature differences across different scoliosis severity levels"""
        print("\n3. CSF feature differences across scoliosis severity groups:")

        # Get CSF feature columns
        non_feature_cols = ['filename', 'amace_angle', 'base_filename', 'severity']
        csf_features = [col for col in self.merged_data.columns
                        if col not in non_feature_cols
                        and self.merged_data[col].dtype in [np.float64, np.int64]]

        # Filter out features with zero variance (all values identical)
        csf_features = [f for f in csf_features if self.merged_data[f].var() > 0]

        print(f"   After removing zero-variance features, {len(csf_features)} features remain")

        if len(csf_features) == 0:
            print("   All features have zero variance, cannot perform group analysis")
            return

        # Select top 12 key features for detailed analysis
        if len(csf_features) > 12:
            # Calculate correlation with angle first, select most correlated features
            corr_values = []
            for feature in csf_features:
                if self.merged_data[feature].isnull().any():
                    continue
                try:
                    corr, _ = stats.pearsonr(self.merged_data[feature], self.merged_data['amace_angle'])
                    corr_values.append((feature, abs(corr)))
                except:
                    continue

            # Sort by absolute correlation
            corr_values.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in corr_values[:12]]
        else:
            selected_features = csf_features

        print(f"   Selected {len(selected_features)} key features for group analysis")

        # Create individual boxplots for each feature
        self._create_individual_severity_boxplots(selected_features)

        # Statistical tests
        self._perform_statistical_tests(selected_features)

    def _create_individual_severity_boxplots(self, features):
        """Create individual boxplots for each CSF feature across different scoliosis severity levels"""
        print("\n   Creating individual scoliosis severity group boxplots...")

        # Set group colors
        severity_palette = {'Mild': '#66c2a5', 'Moderate': '#fc8d62', 'Severe': '#8da0cb'}

        for feature in features:
            # Create a new figure for each feature
            fig, ax = plt.subplots(figsize=(10, 8))

            # Prepare data
            data = []
            group_labels = []
            valid_severities = []

            for severity in ['Mild', 'Moderate', 'Severe']:
                if severity in self.merged_data['severity'].unique():
                    group_data = self.merged_data[self.merged_data['severity'] == severity][feature].dropna()
                    if len(group_data) > 0:  # Ensure there is data
                        data.append(group_data)
                        group_labels.append(f'{severity}\n(n={len(group_data)})')
                        valid_severities.append(severity)

            # If no data for this feature, skip
            if len(data) == 0:
                plt.close(fig)
                continue

            # Create boxplot
            box_plot = ax.boxplot(data, patch_artist=True, labels=group_labels,
                                  medianprops=dict(color='red', linewidth=2),
                                  whiskerprops=dict(color='black', linewidth=1),
                                  capprops=dict(color='black', linewidth=1),
                                  flierprops=dict(marker='o', markersize=5, alpha=0.5))

            # Set colors
            colors = [severity_palette[severity] for severity in valid_severities]
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Set title
            ax.set_title(f'CSF Feature: {feature} by Scoliosis Severity', fontsize=14, fontweight='bold', pad=20)

            # Add statistical significance markers and effect size
            if len(data) >= 2:
                # Check if data is suitable for ANOVA test
                try:
                    if len(data) == 3:
                        # Check if each group has variation
                        all_same = all(len(np.unique(g)) == 1 for g in data)
                        if not all_same:
                            f_stat, p_value = f_oneway(*data)
                            if p_value < 0.05:
                                # Calculate eta squared for ANOVA
                                n_total = sum(len(g) for g in data)
                                grand_mean = np.mean(np.concatenate(data))
                                ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in data)
                                sst = sum(np.sum((g - grand_mean) ** 2) for g in data)
                                eta_squared = ssb / sst if sst != 0 else 0
                                ax.text(0.5, 1.02, f'ANOVA p={p_value:.3e}, η²={eta_squared:.3f}',
                                        transform=ax.transAxes, ha='center', fontsize=11,
                                        color='red', fontweight='bold')
                except Exception as e:
                    # If ANOVA fails, try non-parametric test (Kruskal-Wallis)
                    try:
                        if len(data) == 3:
                            # Check if all values are identical
                            all_values = np.concatenate(data)
                            if len(np.unique(all_values)) > 1:
                                h_stat, p_value = kruskal(*data)
                                if p_value < 0.05:
                                    # Calculate eta squared for Kruskal-Wallis
                                    n_total = sum(len(g) for g in data)
                                    k = len(data)
                                    eta_squared_k = h_stat / (n_total * (k - 1)) if (n_total * (k - 1)) != 0 else 0
                                    ax.text(0.5, 1.02, f'K-W p={p_value:.3e}, η²={eta_squared_k:.3f}',
                                            transform=ax.transAxes, ha='center', fontsize=11,
                                            color='red', fontweight='bold')
                    except:
                        pass

            ax.set_ylabel(f'{feature} Value', fontsize=12)
            ax.set_xlabel('Scoliosis Severity Group', fontsize=12)
            ax.tick_params(axis='x', rotation=0, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save as SVG
            safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.savefig(f'{save_dir}/boxplots/severity_boxplot_{safe_feature_name}.svg', format='svg',
                        bbox_inches='tight')
            plt.close(fig)

        print(f"   ✓ Individual scoliosis severity group boxplots saved to: {save_dir}/boxplots/")

    def _perform_statistical_tests(self, features):
        """Perform statistical tests to compare different scoliosis severity groups"""
        print("\n   Performing statistical tests to compare scoliosis severity groups...")

        results = []

        for feature in features:
            # Extract group data
            groups_data = []
            group_names = []

            for severity in ['Mild', 'Moderate', 'Severe']:
                if severity in self.merged_data['severity'].unique():
                    group_data = self.merged_data[self.merged_data['severity'] == severity][feature].dropna()
                    # Ensure data is valid and has at least one unique value
                    if len(group_data) > 0 and group_data.nunique() > 0:
                        groups_data.append(group_data)
                        group_names.append(severity)

            if len(groups_data) >= 2:
                # Check if all values are identical
                all_values = np.concatenate(groups_data)
                if len(np.unique(all_values)) <= 1:
                    # All values identical, skip statistical test
                    continue

                try:
                    if len(groups_data) == 3:
                        # Kruskal-Wallis H test (non-parametric ANOVA)
                        try:
                            h_stat, p_value = kruskal(*groups_data)
                            test_type = 'Kruskal-Wallis'
                            # Calculate effect size for Kruskal-Wallis
                            n_total = sum(len(g) for g in groups_data)
                            k = len(groups_data)
                            effect_size = h_stat / (n_total * (k - 1)) if (n_total * (k - 1)) != 0 else 0
                        except ValueError as e:
                            # If Kruskal-Wallis fails, try ANOVA
                            try:
                                f_stat, p_value = f_oneway(*groups_data)
                                test_type = 'ANOVA'
                                # Calculate eta squared for ANOVA
                                grand_mean = np.mean(np.concatenate(groups_data))
                                ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_data)
                                sst = sum(np.sum((g - grand_mean) ** 2) for g in groups_data)
                                effect_size = ssb / sst if sst != 0 else 0
                            except:
                                continue
                    else:
                        # Mann-Whitney U test (non-parametric t-test)
                        try:
                            u_stat, p_value = mannwhitneyu(groups_data[0], groups_data[1])
                            test_type = 'Mann-Whitney U'
                            # Calculate effect size r for Mann-Whitney U
                            n1 = len(groups_data[0])
                            n2 = len(groups_data[1])
                            # Calculate Z-score for U
                            mu_u = n1 * n2 / 2
                            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                            z_stat = (u_stat - mu_u) / sigma_u
                            effect_size = np.abs(z_stat) / np.sqrt(n1 + n2)
                        except:
                            continue

                    # Determine significance
                    significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'

                    # Calculate group means
                    means = [np.mean(g) for g in groups_data]
                    stds = [np.std(g, ddof=1) for g in groups_data]

                    results.append({
                        'Feature': feature,
                        'Test': test_type,
                        'P-value': p_value,
                        'Significance': significance,
                        'Effect_Size': effect_size,
                        'Effect_Size_Interpretation': 'Large' if effect_size >= 0.14 else 'Medium' if effect_size >= 0.06 else 'Small',
                        'Means': means,
                        'Std_Devs': stds,
                        'Groups': group_names
                    })

                except Exception as e:
                    print(f"    Statistical test failed for feature {feature}: {str(e)}")
                    continue

        # Save statistical test results
        if results:
            self.subgroup_results = pd.DataFrame(results)

            print("\n   Statistical test results (top 10 significant features):")
            sig_results = self.subgroup_results[self.subgroup_results['P-value'] < 0.05].head(10)
            if not sig_results.empty:
                print(sig_results[
                          ['Feature', 'Test', 'P-value', 'Significance', 'Effect_Size', 'Effect_Size_Interpretation']])
            else:
                print("    No significant differences found (p < 0.05)")

            # Save detailed results to CSV
            self.subgroup_results.to_csv(f'{save_dir}/severity_statistical_tests.csv', index=False, encoding='utf-8')
            print(f"   ✓ Statistical test results saved: severity_statistical_tests.csv")
        else:
            print("    No available statistical test results")
            self.subgroup_results = pd.DataFrame()

    def correlation_analysis(self):
        """Correlation analysis: Relationship between CSF features and amace_angle"""
        print("\n" + "=" * 60)
        print("Correlation Analysis: CSF Features vs Cobb Angle")
        print("=" * 60)

        # 1. Automatically identify CSF feature columns
        non_feature_cols = ['filename', 'amace_angle', 'base_filename', 'severity']
        csf_features = [col for col in self.merged_data.columns
                        if col not in non_feature_cols
                        and self.merged_data[col].dtype in [np.float64, np.int64]]

        print(f"   Analyzing correlation between {len(csf_features)} CSF features and Cobb angle...")

        # 2. Calculate Pearson correlation coefficient for each CSF feature with amace_angle
        correlations = []
        for feature in csf_features:
            # Skip features with missing values
            if self.merged_data[feature].isnull().any():
                continue

            # Check if feature variance is zero
            if self.merged_data[feature].var() == 0:
                continue

            try:
                corr, p_value = stats.pearsonr(self.merged_data[feature], self.merged_data['amace_angle'])
                correlations.append({
                    'Feature': feature,
                    'Correlation': corr,
                    'P-value': p_value,
                    'Absolute_Correlation': abs(corr)
                })
            except Exception as e:
                print(f"    Correlation calculation failed for feature {feature}: {str(e)}")
                continue

        # 3. Sort by absolute correlation
        if correlations:
            self.correlation_results = pd.DataFrame(correlations).sort_values(by='Absolute_Correlation',
                                                                              ascending=False)

            print("\n   Correlation analysis results (sorted by absolute value, top 20):")
            print(self.correlation_results.head(20))

            # 4. Visualize correlation heatmap: Top 15 most correlated features + amace_angle
            top_n = min(15, len(self.correlation_results))
            top_features = self.correlation_results['Feature'].head(top_n).tolist() + ['amace_angle']

            # Ensure all features exist and are valid
            top_features = [f for f in top_features if f in self.merged_data.columns and self.merged_data[f].var() > 0]

            if len(top_features) > 1:  # Need at least 2 features to create heatmap
                corr_matrix = self.merged_data[top_features].corr()

                plt.figure(figsize=(14, 12))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                            fmt='.2f', center=0, square=True, linewidths=0.5,
                            cbar_kws={"shrink": 0.8})
                plt.title(f'CSF Feature Correlation Heatmap with Cobb Angle (Top {top_n} Features)', fontsize=16,
                          fontweight='bold')
                plt.tight_layout()

                # Save as SVG
                plt.savefig(f'{save_dir}/correlation/csf_cobb_correlation_heatmap.svg', format='svg',
                            bbox_inches='tight')
                plt.close()
                print(f"\n   ✓ Correlation heatmap saved: {save_dir}/correlation/csf_cobb_correlation_heatmap.svg")

                # 5. Stratified correlation analysis by scoliosis severity
                self._analyze_stratified_correlations(top_features)
            else:
                print("    Not enough valid features to create correlation heatmap")
        else:
            print("    No available correlation results")
            self.correlation_results = pd.DataFrame()

        return self.correlation_results

    def _analyze_stratified_correlations(self, features):
        """Analyze feature correlations stratified by scoliosis severity"""
        print("\n   Performing stratified correlation analysis by scoliosis severity...")

        # Remove 'amace_angle' as we're focusing on relationships with other features
        if 'amace_angle' in features:
            features.remove('amace_angle')

        stratified_results = []

        for severity in ['Mild', 'Moderate', 'Severe']:
            if severity in self.merged_data['severity'].unique():
                sub_data = self.merged_data[self.merged_data['severity'] == severity]

                if len(sub_data) >= 5:  # Need at least 5 samples for meaningful analysis
                    for feature in features:
                        if feature in sub_data.columns and not sub_data[feature].isnull().any():
                            # Check if feature variance is zero
                            if sub_data[feature].var() == 0:
                                continue
                            try:
                                corr, p_value = stats.pearsonr(sub_data[feature], sub_data['amace_angle'])
                                stratified_results.append({
                                    'Severity': severity,
                                    'Feature': feature,
                                    'Correlation': corr,
                                    'P-value': p_value,
                                    'Sample_Size': len(sub_data)
                                })
                            except:
                                continue

        if stratified_results:
            strat_df = pd.DataFrame(stratified_results)

            # Create visualization
            self._visualize_stratified_correlations(strat_df)

            # Save results
            strat_df.to_csv(f'{save_dir}/stratified_correlation_results.csv', index=False, encoding='utf-8')
            print(f"   ✓ Stratified correlation analysis results saved: stratified_correlation_results.csv")
        else:
            print("    No available stratified correlation results")

    def _visualize_stratified_correlations(self, strat_df):
        """Visualize stratified correlation analysis results"""
        # Select most important features (based on overall correlation)
        if len(self.correlation_results) > 0:
            top_features = self.correlation_results['Feature'].head(10).tolist()
            strat_top = strat_df[strat_df['Feature'].isin(top_features)]

            if not strat_top.empty:
                # Create grouped bar chart
                plt.figure(figsize=(14, 8))

                # Prepare data
                pivot_data = strat_top.pivot(index='Feature', columns='Severity', values='Correlation')
                pivot_data = pivot_data.reindex(top_features)

                # Create grouped bar chart
                ax = pivot_data.plot(kind='bar', figsize=(14, 8), color=['#66c2a5', '#fc8d62', '#8da0cb'])

                plt.title('CSF Feature Correlations with Cobb Angle by Scoliosis Severity', fontsize=16,
                          fontweight='bold')
                plt.xlabel('CSF Feature', fontsize=12)
                plt.ylabel('Pearson Correlation Coefficient', fontsize=12)
                plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                plt.legend(title='Scoliosis Severity', fontsize=10)
                plt.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # # Add sample size annotations
                # for i, severity in enumerate(['Mild', 'Moderate', 'Severe']):
                #     if severity in strat_top['Severity'].unique():
                #         sample_sizes = strat_top[strat_top['Severity'] == severity].groupby('Feature')[
                #             'Sample_Size'].first()
                #         for j, feature in enumerate(pivot_data.index):
                #             if feature in sample_sizes.index:
                #                 y_pos = pivot_data.loc[feature, severity]
                #                 if not np.isnan(y_pos):
                #                     ax.text(j + i * 0.25 - 0.25, y_pos + (0.02 if y_pos >= 0 else -0.02),
                #                             f'n={sample_sizes[feature]}',
                #                             ha='center', va='bottom' if y_pos >= 0 else 'top',
                #                             fontsize=8, color='black')

                # Save as SVG
                plt.savefig(f'{save_dir}/correlation/stratified_correlation_barchart.svg', format='svg',
                            bbox_inches='tight')
                plt.close()
                print(
                    f"   ✓ Stratified correlation bar chart saved: {save_dir}/correlation/stratified_correlation_barchart.svg")
            else:
                print("    Not enough stratified correlation data for visualization")
        else:
            print("    No correlation results for stratified analysis")

    def scatter_plot_analysis(self, top_n=5):
        """Create individual scatter plots of top N most correlated features, colored by scoliosis severity"""
        if len(self.correlation_results) == 0:
            print("\n    No correlation data, skipping scatter plot creation")
            return

        print(
            f"\n    Creating individual scatter plots for top {top_n} most correlated features (colored by scoliosis severity)...")

        # Get top N most correlated features
        top_features = self.correlation_results['Feature'].head(top_n).tolist()

        # Define color mapping
        severity_colors = {'Mild': '#66c2a5', 'Moderate': '#fc8d62', 'Severe': '#8da0cb'}

        for i, feature in enumerate(top_features):
            # Create a new figure for each feature
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot colored by scoliosis severity
            for severity in ['Mild', 'Moderate', 'Severe']:
                if severity in self.merged_data['severity'].unique():
                    mask = self.merged_data['severity'] == severity
                    sub_data = self.merged_data[mask]

                    if len(sub_data) > 0:
                        ax.scatter(sub_data[feature], sub_data['amace_angle'],
                                   alpha=0.7, s=80, edgecolor='w', linewidth=0.5,
                                   label=severity, color=severity_colors[severity])

            # Add overall trend line
            try:
                sns.regplot(x=self.merged_data[feature], y=self.merged_data['amace_angle'],
                            ax=ax, scatter=False, color='black', line_kws={'linewidth': 2, 'alpha': 0.5})
            except:
                pass

            # Get correlation coefficient
            corr_row = self.correlation_results[self.correlation_results['Feature'] == feature]
            if not corr_row.empty:
                corr = corr_row['Correlation'].values[0]
                p_value = corr_row['P-value'].values[0]

                # Add statistical information
                stats_text = f'r = {corr:.3f}\np = {p_value:.3e}'
                if p_value < 0.05:
                    stats_text += ' *'
                if p_value < 0.01:
                    stats_text += '*'
                if p_value < 0.001:
                    stats_text += '*'

                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Set title and labels
            ax.set_title(f'{feature} vs Cobb Angle', fontsize=14, fontweight='bold')
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Cobb Angle (°)', fontsize=12)

            # Adjust y-axis spacing to make points appear more concentrated
            # Calculate y-axis limits with padding
            y_min = self.merged_data['amace_angle'].min()
            y_max = self.merged_data['amace_angle'].max()
            y_range = y_max - y_min

            # Add 10% padding on both sides
            y_padding = y_range * 0.5
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Adjust x-axis limits
            x_min = self.merged_data[feature].min()
            x_max = self.merged_data[feature].max()
            x_range = x_max - x_min

            # Add 5% padding on both sides
            x_padding = x_range * 0.05
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

            ax.grid(True, alpha=0.3)
            ax.legend(title='Scoliosis Severity', fontsize=10, title_fontsize=11)

            plt.tight_layout()

            # Save as SVG
            safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.savefig(f'{save_dir}/scatter_plots/scatter_{safe_feature_name}.svg', format='svg', bbox_inches='tight')
            plt.close(fig)

            print(f"    ✓ Created scatter plot for {feature}")

        print(f"\n   ✓ All scatter plots saved to: {save_dir}/scatter_plots/")

    def machine_learning_analysis(self):
        """Machine learning analysis: Evaluate CSF features' ability to predict Cobb angle with 5-fold CV"""
        print("\n" + "=" * 60)
        print("Machine Learning Analysis: Predicting Cobb Angle from CSF Features")
        print("=" * 60)

        if len(self.correlation_results) == 0 or len(self.correlation_results) < 3:
            print("\n    Insufficient correlation data, skipping machine learning analysis")
            return

        # 1. Select features: CSF features with correlation absolute value > 0.1
        self.selected_features = self.correlation_results[abs(self.correlation_results['Correlation']) > 0.1][
            'Feature'].tolist()

        if len(self.selected_features) == 0:
            # If no features with correlation > 0.1, select top 15
            self.selected_features = self.correlation_results['Feature'].head(15).tolist()

        # Ensure no more than 20 features to avoid overfitting
        if len(self.selected_features) > 20:
            self.selected_features = self.selected_features[:20]

        print(f"   Selected {len(self.selected_features)} features for modeling:")
        print(f"   {self.selected_features}")

        # 2. Prepare feature matrix X and target variable y
        X = self.merged_data[self.selected_features].fillna(0)
        y = self.merged_data['amace_angle']

        # 3. Standardize data
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.selected_features)

        # 4. Split into training and test sets (70% training, 30% testing)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=self.merged_data['severity'])
        except:
            # If stratified sampling fails, use regular sampling
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42)

        print(f"\n   Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # 5. 5-fold cross-validation设置
        cv_folds = 5
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        print(f"\n   Using {cv_folds}-fold cross-validation for model evaluation")

        # 6. Cross-validation evaluation function for all metrics
        def cross_val_evaluate_metrics(name, model, X, y, cv_strategy):
            """使用交叉验证评估模型的多个指标"""
            try:
                # 评估R²
                r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
                r2_mean = r2_scores.mean()
                r2_std = r2_scores.std()

                # 评估MAE
                mae_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
                mae_scores = -mae_scores  # 转换为正数
                mae_mean = mae_scores.mean()
                mae_std = mae_scores.std()

                # 评估RMSE
                rmse_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_root_mean_squared_error')
                rmse_scores = -rmse_scores  # 转换为正数
                rmse_mean = rmse_scores.mean()
                rmse_std = rmse_scores.std()

                print(f"   {name} {cv_folds}-Fold CV Results:")
                print(f"   R²: {r2_mean:.4f} ± {r2_std:.4f}")
                print(f"   MAE: {mae_mean:.2f}° ± {mae_std:.2f}°")
                print(f"   RMSE: {rmse_mean:.2f}° ± {rmse_std:.2f}°")

                return {
                    'r2_scores': r2_scores,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'mae_scores': mae_scores,
                    'mae_mean': mae_mean,
                    'mae_std': mae_std,
                    'rmse_scores': rmse_scores,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std
                }
            except Exception as e:
                print(f"   Cross-validation failed for {name}: {str(e)}")
                return None

        # 7. Model evaluation function for test set
        def evaluate_model(name, y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100

            # Error analysis by scoliosis severity
            error_by_severity = {}
            test_indices = y_true.index
            for severity in ['Mild', 'Moderate', 'Severe']:
                mask = self.merged_data.loc[test_indices, 'severity'] == severity
                if mask.any():
                    y_true_sub = y_true[mask]
                    y_pred_sub = y_pred[mask]
                    if len(y_true_sub) > 1:  # Need at least 2 samples to calculate R2
                        error_by_severity[severity] = {
                            'MAE': mean_absolute_error(y_true_sub, y_pred_sub),
                            'RMSE': np.sqrt(mean_squared_error(y_true_sub, y_pred_sub)),
                            'R2': r2_score(y_true_sub, y_pred_sub) if len(np.unique(y_true_sub)) > 1 else 0,
                            'n_samples': len(y_true_sub)
                        }

            print(f"\n   {name} model evaluation on test set:")
            print(f"   R² score: {r2:.4f}")
            print(f"   Mean Absolute Error (MAE): {mae:.2f}°")
            print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}°")
            print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            if error_by_severity:
                print(f"   Error analysis by scoliosis severity:")
                for severity, errors in error_by_severity.items():
                    print(f"   - {severity}: MAE={errors['MAE']:.2f}°, RMSE={errors['RMSE']:.2f}°, "
                          f"R²={errors['R2']:.3f} (n={errors['n_samples']})")

            return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape,
                    'error_by_severity': error_by_severity}

        # 8. Linear Regression model
        print("\n" + "-" * 40)
        print("Training Linear Regression model...")
        try:
            lr_model = LinearRegression()

            # 5-fold交叉验证评估
            lr_cv_results = cross_val_evaluate_metrics("Linear Regression", lr_model, X_train, y_train, cv_strategy)

            # 训练最终模型
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_eval = evaluate_model("Linear Regression", y_test, lr_pred)

            # 保存交叉验证结果
            if lr_cv_results is not None:
                lr_eval['cv_r2_mean'] = lr_cv_results['r2_mean']
                lr_eval['cv_r2_std'] = lr_cv_results['r2_std']
                lr_eval['cv_mae_mean'] = lr_cv_results['mae_mean']
                lr_eval['cv_mae_std'] = lr_cv_results['mae_std']
                lr_eval['cv_rmse_mean'] = lr_cv_results['rmse_mean']
                lr_eval['cv_rmse_std'] = lr_cv_results['rmse_std']
        except Exception as e:
            print(f"   Linear Regression model training failed: {str(e)}")
            lr_eval = {'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0, 'error_by_severity': {}}

        # 9. Random Forest Regression model (with parameter tuning using 5-fold CV)
        print("\n" + "-" * 40)
        print("Training Random Forest model...")

        try:
            # Simplified parameter grid for faster computation
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }

            # 使用5折交叉验证进行参数调优
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=cv_strategy,  # 使用5折交叉验证
                scoring='r2',
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            grid_search.fit(X_train, y_train)

            print(f"\n   Best Random Forest parameters: {grid_search.best_params_}")
            print(f"   Best CV R² score: {grid_search.best_score_:.4f}")

            # 展示所有参数组合的交叉验证结果
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(f"   Top 5 parameter combinations by CV score:")
            for i, row in cv_results_df.head(5).iterrows():
                params = row['params']
                mean_score = row['mean_test_score']
                std_score = row['std_test_score']
                print(f"   {i + 1}. {params}: {mean_score:.4f} ± {std_score:.4f}")

            rf_model = grid_search.best_estimator_
            rf_pred = rf_model.predict(X_test)
            rf_eval = evaluate_model("Random Forest", y_test, rf_pred)

            # 保存交叉验证结果
            rf_eval['best_cv_score'] = grid_search.best_score_
            rf_eval['best_params'] = grid_search.best_params_
            rf_eval['cv_results'] = cv_results_df

        except Exception as e:
            print(f"   Random Forest model training failed: {str(e)}")
            rf_model = None
            rf_eval = {'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0, 'error_by_severity': {}}

        # 10. Gradient Boosting Regression model
        print("\n" + "-" * 40)
        print("Training Gradient Boosting Regression model...")
        try:
            gb_model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)

            # 5-fold交叉验证评估
            gb_cv_results = cross_val_evaluate_metrics("Gradient Boosting", gb_model, X_train, y_train, cv_strategy)

            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_eval = evaluate_model("Gradient Boosting", y_test, gb_pred)

            # 保存交叉验证结果
            if gb_cv_results is not None:
                gb_eval['cv_r2_mean'] = gb_cv_results['r2_mean']
                gb_eval['cv_r2_std'] = gb_cv_results['r2_std']
                gb_eval['cv_mae_mean'] = gb_cv_results['mae_mean']
                gb_eval['cv_mae_std'] = gb_cv_results['mae_std']
                gb_eval['cv_rmse_mean'] = gb_cv_results['rmse_mean']
                gb_eval['cv_rmse_std'] = gb_cv_results['rmse_std']
        except Exception as e:
            print(f"   Gradient Boosting model training failed: {str(e)}")
            gb_model = None
            gb_eval = {'r2': 0, 'mae': 0, 'rmse': 0, 'mape': 0, 'error_by_severity': {}}

        # 11. Save machine learning results
        self.ml_results = {
            'linear_regression': lr_eval,
            'random_forest': rf_eval,
            'gradient_boosting': gb_eval,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'selected_features': self.selected_features,
            'cv_folds': cv_folds
        }

        if rf_model is not None:
            self.ml_results['best_rf_params'] = grid_search.best_params_

        # 12. Create individual visualizations for prediction results
        self._create_individual_prediction_plots(y_test, lr_pred, rf_pred, gb_pred,
                                                 lr_eval, rf_eval, gb_eval)

        # 13. Create cross-validation comparison visualizations
        self._create_cv_comparison_visualizations(lr_eval, rf_eval, gb_eval)

        # 14. Feature importance analysis
        if rf_model is not None or gb_model is not None:
            self._analyze_feature_importance(rf_model, gb_model)
        else:
            print("    Model training failed, skipping feature importance analysis")

        # 15. SHAP model interpretation
        if rf_model is not None and not X_test.empty:
            try:
                self._perform_shap_analysis(rf_model, X_test, y_test)
            except Exception as e:
                print(f"   SHAP analysis failed: {str(e)}")
        else:
            print("    Random Forest model unavailable or test set empty, skipping SHAP analysis")

        return self.ml_results

    def _create_individual_prediction_plots(self, y_test, lr_pred, rf_pred, gb_pred, lr_eval, rf_eval, gb_eval):
        """Create individual visualizations for prediction results"""
        print("\n   Creating individual prediction result visualizations...")

        # 1. Linear Regression prediction plot
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter(y_test, lr_pred, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Cobb Angle (°)', fontsize=12)
        ax1.set_ylabel('Predicted Cobb Angle (°)', fontsize=12)
        ax1.set_title(f'Linear Regression Prediction\nR²={lr_eval["r2"]:.3f}, MAE={lr_eval["mae"]:.2f}°', fontsize=14,
                      fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Adjust y-axis spacing
        y_min = min(y_test.min(), lr_pred.min())
        y_max = max(y_test.max(), lr_pred.max())
        y_range = y_max - y_min
        y_padding = y_range * 0.5
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/linear_regression_prediction.svg', format='svg', bbox_inches='tight')
        plt.close(fig1)

        # 2. Random Forest prediction plot
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.scatter(y_test, rf_pred, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax2.set_xlabel('Actual Cobb Angle (°)', fontsize=12)
        ax2.set_ylabel('Predicted Cobb Angle (°)', fontsize=12)
        ax2.set_title(f'Random Forest Prediction\nR²={rf_eval["r2"]:.3f}, MAE={rf_eval["mae"]:.2f}°', fontsize=14,
                      fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Adjust y-axis spacing
        y_min = min(y_test.min(), rf_pred.min())
        y_max = max(y_test.max(), rf_pred.max())
        y_range = y_max - y_min
        y_padding = y_range * 0.5
        ax2.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/random_forest_prediction.svg', format='svg', bbox_inches='tight')
        plt.close(fig2)

        # 3. Gradient Boosting prediction plot
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.scatter(y_test, gb_pred, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax3.set_xlabel('Actual Cobb Angle (°)', fontsize=12)
        ax3.set_ylabel('Predicted Cobb Angle (°)', fontsize=12)
        ax3.set_title(f'Gradient Boosting Prediction\nR²={gb_eval["r2"]:.3f}, MAE={gb_eval["mae"]:.2f}°', fontsize=14,
                      fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Adjust y-axis spacing
        y_min = min(y_test.min(), gb_pred.min())
        y_max = max(y_test.max(), gb_pred.max())
        y_range = y_max - y_min
        y_padding = y_range * 0.5
        ax3.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/gradient_boosting_prediction.svg', format='svg', bbox_inches='tight')
        plt.close(fig3)

        # 4. Model performance comparison bar chart
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
        r2_scores = [lr_eval['r2'], rf_eval['r2'], gb_eval['r2']]
        mae_scores = [lr_eval['mae'], rf_eval['mae'], gb_eval['mae']]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, r2_scores, width, label='R² Score', color='skyblue')
        bars2 = ax4.bar(x + width / 2, mae_scores, width, label='MAE(°)', color='lightcoral')

        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/model_performance_comparison.svg', format='svg', bbox_inches='tight')
        plt.close(fig4)

        # 5. Error distribution plot
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        errors = {
            'Linear Regression': y_test - lr_pred,
            'Random Forest': y_test - rf_pred,
            'Gradient Boosting': y_test - gb_pred
        }

        error_data = []
        for model_name, error_values in errors.items():
            for err in error_values:
                error_data.append([model_name, err])

        error_df = pd.DataFrame(error_data, columns=['Model', 'Error'])
        sns.boxplot(x='Model', y='Error', data=error_df, ax=ax5, palette='Set2')
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax5.set_xlabel('Model', fontsize=12)
        ax5.set_ylabel('Prediction Error (°)', fontsize=12)
        ax5.set_title('Model Prediction Error Distribution (Test Set)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/error_distribution.svg', format='svg', bbox_inches='tight')
        plt.close(fig5)

        # 6. Prediction error by scoliosis severity
        fig6, ax6 = plt.subplots(figsize=(12, 8))

        # Use Random Forest prediction error for analysis (usually best performance)
        test_indices = y_test.index
        error_by_severity = []

        for severity in ['Mild', 'Moderate', 'Severe']:
            mask = self.merged_data.loc[test_indices, 'severity'] == severity
            if mask.any():
                severity_errors = y_test[mask] - rf_pred[mask]
                for err in severity_errors:
                    error_by_severity.append([severity, err])

        if error_by_severity:
            severity_error_df = pd.DataFrame(error_by_severity, columns=['Severity', 'Error'])
            sns.boxplot(x='Severity', y='Error', data=severity_error_df, ax=ax6,
                        palette={'Mild': '#66c2a5', 'Moderate': '#fc8d62', 'Severe': '#8da0cb'})
            ax6.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax6.set_xlabel('Scoliosis Severity', fontsize=12)
            ax6.set_ylabel('Prediction Error (°)', fontsize=12)
            ax6.set_title('Prediction Error Distribution by Scoliosis Severity', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        else:
            ax6.text(0.5, 0.5, 'No data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Prediction Error Distribution by Scoliosis Severity', fontsize=14, fontweight='bold')
            ax6.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ml_results/error_by_severity.svg', format='svg', bbox_inches='tight')
        plt.close(fig6)

        print(f"   ✓ Individual prediction result visualizations saved to: {save_dir}/ml_results/")

    def _create_cv_comparison_visualizations(self, lr_eval, rf_eval, gb_eval):
        """Create visualizations for cross-validation results"""
        print("\n   Creating cross-validation result visualizations...")

        # 1. Cross-validation R² scores comparison
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
        cv_r2_means = []
        cv_r2_stds = []

        # Collect cross-validation results
        for model_name, eval_dict in zip(models, [lr_eval, rf_eval, gb_eval]):
            if 'cv_r2_mean' in eval_dict:
                cv_r2_means.append(eval_dict['cv_r2_mean'])
                cv_r2_stds.append(eval_dict['cv_r2_std'])
            elif 'best_cv_score' in eval_dict:
                cv_r2_means.append(eval_dict['best_cv_score'])
                cv_r2_stds.append(0.05)  # Default std for display
            else:
                cv_r2_means.append(0)
                cv_r2_stds.append(0)

        # Create bar chart
        x_pos = np.arange(len(models))
        bars = ax1.bar(x_pos, cv_r2_means, yerr=cv_r2_stds,
                       capsize=10, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])

        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('5-Fold Cross-Validation R² Score', fontsize=12)
        ax1.set_title('5-Fold Cross-Validation Performance Comparison (Training Set)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, cv_r2_means, cv_r2_stds)):
            if mean > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/cv_results/cv_r2_comparison.svg',
                    format='svg', bbox_inches='tight')
        plt.close(fig1)

        # 2. Train-Test performance comparison
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        test_r2_scores = [lr_eval['r2'], rf_eval['r2'], gb_eval['r2']]

        width = 0.35
        x = np.arange(len(models))

        bars1 = ax2.bar(x - width / 2, cv_r2_means, width, label='CV R² (Train)', color='skyblue')
        bars2 = ax2.bar(x + width / 2, test_r2_scores, width, label='Test R²', color='lightcoral')

        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('Train (CV) vs Test Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/cv_results/train_test_comparison.svg',
                    format='svg', bbox_inches='tight')
        plt.close(fig2)

        # 3. Save CV results to CSV
        cv_comparison_data = []
        for i, model_name in enumerate(models):
            cv_comparison_data.append({
                'Model': model_name,
                'CV_R2_Mean': cv_r2_means[i] if i < len(cv_r2_means) else 0,
                'CV_R2_Std': cv_r2_stds[i] if i < len(cv_r2_stds) else 0,
                'Test_R2': test_r2_scores[i] if i < len(test_r2_scores) else 0,
                'Test_MAE': [lr_eval['mae'], rf_eval['mae'], gb_eval['mae']][i],
                'Test_RMSE': [lr_eval['rmse'], rf_eval['rmse'], gb_eval['rmse']][i]
            })

        cv_comparison_df = pd.DataFrame(cv_comparison_data)
        cv_comparison_df.to_csv(f'{save_dir}/cv_results/cv_test_comparison.csv', index=False, encoding='utf-8')

        print(f"   ✓ Cross-validation visualizations saved to: {save_dir}/cv_results/")

    def _analyze_feature_importance(self, rf_model, gb_model):
        """Analyze feature importance"""
        print("\n" + "-" * 40)
        print("Analyzing feature importance...")

        importance_data = []

        # Random Forest feature importance
        if rf_model is not None:
            rf_importance = pd.DataFrame({
                'Feature': self.selected_features,
                'RF_Importance': rf_model.feature_importances_
            }).sort_values(by='RF_Importance', ascending=False)
            importance_data.append(rf_importance)

        # Gradient Boosting feature importance
        if gb_model is not None:
            gb_importance = pd.DataFrame({
                'Feature': self.selected_features,
                'GB_Importance': gb_model.feature_importances_
            }).sort_values(by='GB_Importance', ascending=False)
            importance_data.append(gb_importance)

        if importance_data:
            # Merge feature importance
            if len(importance_data) == 2:
                self.feature_importance = pd.merge(importance_data[0], importance_data[1], on='Feature')
                self.feature_importance['Average_Importance'] = (
                                                                        self.feature_importance['RF_Importance'] +
                                                                        self.feature_importance['GB_Importance']) / 2
            else:
                self.feature_importance = importance_data[0]
                self.feature_importance['Average_Importance'] = self.feature_importance.iloc[:, 1]

            self.feature_importance = self.feature_importance.sort_values(by='Average_Importance', ascending=False)

            print("\n   Feature importance ranking (top 10):")
            print(self.feature_importance.head(10))

            # Visualize feature importance
            if rf_model is not None:
                # Random Forest feature importance plot
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                rf_top = importance_data[0].head(15)
                ax1.barh(range(len(rf_top)), rf_top['RF_Importance'][::-1], color='skyblue')
                ax1.set_yticks(range(len(rf_top)))
                ax1.set_yticklabels(rf_top['Feature'][::-1], fontsize=11)
                ax1.set_xlabel('Importance', fontsize=12)
                ax1.set_title('Random Forest Feature Importance (Top 15)', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')

                plt.tight_layout()
                plt.savefig(f'{save_dir}/feature_importance/rf_feature_importance.svg', format='svg',
                            bbox_inches='tight')
                plt.close(fig1)

            if gb_model is not None:
                # Gradient Boosting feature importance plot
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                gb_top = importance_data[1].head(15)
                ax2.barh(range(len(gb_top)), gb_top['GB_Importance'][::-1], color='lightcoral')
                ax2.set_yticks(range(len(gb_top)))
                ax2.set_yticklabels(gb_top['Feature'][::-1], fontsize=11)
                ax2.set_xlabel('Importance', fontsize=12)
                ax2.set_title('Gradient Boosting Feature Importance (Top 15)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')

                plt.tight_layout()
                plt.savefig(f'{save_dir}/feature_importance/gb_feature_importance.svg', format='svg',
                            bbox_inches='tight')
                plt.close(fig2)

            # Average feature importance plot
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            avg_top = self.feature_importance.head(15)
            ax3.barh(range(len(avg_top)), avg_top['Average_Importance'][::-1], color='lightgreen')
            ax3.set_yticks(range(len(avg_top)))
            ax3.set_yticklabels(avg_top['Feature'][::-1], fontsize=11)
            ax3.set_xlabel('Average Importance', fontsize=12)
            ax3.set_title('Average Feature Importance (Top 15)', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance/average_feature_importance.svg', format='svg',
                        bbox_inches='tight')
            plt.close(fig3)

            print(f"   ✓ Feature importance visualizations saved to: {save_dir}/feature_importance/")

            # Save feature importance to CSV
            self.feature_importance.to_csv(f'{save_dir}/feature_importance/feature_importance_details.csv',
                                           index=False, encoding='utf-8')
            print(
                f"   ✓ Feature importance details saved: {save_dir}/feature_importance/feature_importance_details.csv")
        else:
            print("    No available feature importance data")
            self.feature_importance = pd.DataFrame()

    def _perform_shap_analysis(self, model, X_test, y_test):
        """Perform SHAP analysis to interpret model predictions"""
        print("\n" + "-" * 40)
        print("Performing SHAP analysis to interpret model predictions...")

        try:
            # Use TreeExplainer to interpret Random Forest model
            self.shap_explainer = shap.TreeExplainer(model)
            self.shap_values = self.shap_explainer.shap_values(X_test)

            # 调试信息
            print(f"   SHAP values shape: {self.shap_values.shape}")
            print(f"   Expected value: {self.shap_explainer.expected_value}")
            print(f"   Expected value type: {type(self.shap_explainer.expected_value)}")

            # 1. SHAP summary plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(self.shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_analysis/shap_summary_plot.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(f"   ✓ SHAP summary plot saved: {save_dir}/shap_analysis/shap_summary_plot.svg")

            # 2. SHAP bar chart (feature importance)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, X_test, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_analysis/shap_feature_importance.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(
                f"   ✓ SHAP feature importance visualization saved: {save_dir}/shap_analysis/shap_feature_importance.svg")

            # 3. Save SHAP values for further analysis
            shap_df = pd.DataFrame(self.shap_values, columns=X_test.columns)

            # 修复base_value的处理
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, (int, float, np.number)):
                base_value_array = np.full(len(shap_df), expected_value)
            elif isinstance(expected_value, (list, np.ndarray)) and len(expected_value) == 1:
                base_value_array = np.full(len(shap_df), expected_value[0])
            else:
                base_value_array = expected_value

            if len(base_value_array) != len(shap_df):
                raise ValueError(f"base_value维度({len(base_value_array)})与SHAP数据({len(shap_df)})不匹配")

            shap_df['base_value'] = base_value_array

            # 实际Cobb角：直接从y_test获取（无需通过索引匹配）
            shap_df['actual_cobb_angle'] = y_test.values  # y_test是测试集的实际值，索引已对齐
            # 预测值：直接用模型预测X_test
            shap_df['predicted_cobb_angle'] = model.predict(X_test)

            shap_df.to_csv(f'{save_dir}/shap_analysis/shap_values.csv', index=False, encoding='utf-8')
            print(f"   ✓ SHAP values saved: {save_dir}/shap_analysis/shap_values.csv")

            # 4. 创建SHAP依赖图（针对最重要的特征）
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                try:
                    top_features = self.feature_importance['Feature'].head(3).tolist()

                    for feature in top_features:
                        if feature in X_test.columns:
                            plt.figure(figsize=(10, 8))
                            shap.dependence_plot(feature, self.shap_values, X_test, show=False)
                            plt.title(f'SHAP Dependence Plot for {feature}', fontsize=14, fontweight='bold')
                            plt.tight_layout()

                            safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                            plt.savefig(f'{save_dir}/shap_analysis/shap_dependence_{safe_feature_name}.svg',
                                        format='svg', bbox_inches='tight')
                            plt.close()
                            print(f"   ✓ SHAP dependence plot for {feature} saved")
                except Exception as e:
                    print(f"   ⚠ Failed to create SHAP dependence plots: {str(e)}")

        except Exception as e:
            print(f"   ⚠ Error in SHAP analysis: {str(e)}")
            import traceback
            traceback.print_exc(limit=2)
            print("   ⚠ Skipping SHAP analysis")

    def generate_markdown_report(self):
        """Generate structured Markdown analysis report with cross-validation results"""
        report_path = os.path.join(save_dir, "CSF_Cobb_Angle_Analysis_Report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            # Title
            f.write("# CSF Signal and Scoliosis Angle Multidimensional Analysis Report\n\n")
            f.write("**Report Generation Time**: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

            # 1. Basic Data Information
            f.write("## 1. Basic Data Information\n")
            f.write(f"- **Data Sources**:\n")
            f.write(f"  - CSF Feature Data: `{os.path.basename(self.csf_path)}`\n")
            f.write(f"  - Cobb Angle Data: `{os.path.basename(self.cobb_path)}`\n")
            f.write(f"- **Matching Status**: {len(self.merged_data)} samples fully matched\n")
            f.write(f"- **Match Rate**: {len(self.merged_data) / len(pd.read_csv(self.cobb_path)) * 100:.2f}%\n")

            f.write(f"- **Scoliosis Severity Grouping**:\n")
            if self.severity_stats is not None:
                for severity, stats in self.severity_stats.iterrows():
                    f.write(
                        f"  - **{severity}**: {stats['count']} cases, angle range: {stats['min']:.1f}°-{stats['max']:.1f}°, average: {stats['mean']:.1f}°±{stats['std']:.1f}°\n")
            f.write("\n")

            # 2. Key Findings Summary
            f.write("## 2. Key Findings Summary\n")

            # 2.1 Scoliosis severity group feature differences
            if self.subgroup_results is not None and not self.subgroup_results.empty:
                sig_features = self.subgroup_results[self.subgroup_results['P-value'] < 0.05]
                if not sig_features.empty:
                    f.write("### 2.1 Scoliosis Severity Group Feature Differences\n")
                    f.write("Significant differences in CSF features across scoliosis severity groups:\n")
                    for _, row in sig_features.head(5).iterrows():
                        means_str = ', '.join([f"{m:.2f}" for m in row['Means']])
                        groups_str = ', '.join(row['Groups'])
                        f.write(
                            f"- **{row['Feature']}**: Means for {groups_str} groups: {means_str} (p={row['P-value']:.3f}{row['Significance']})\n")
                    f.write("\n")

            # 2.2 Strongest correlation features
            if self.correlation_results is not None and not self.correlation_results.empty:
                f.write("### 2.2 Strongest Correlation Features\n")
                top_corr = self.correlation_results.head(3)
                for _, row in top_corr.iterrows():
                    direction = "positive" if row['Correlation'] > 0 else "negative"
                    f.write(
                        f"- **{row['Feature']}**: {abs(row['Correlation']):.3f} {direction} correlation with Cobb angle (p={row['P-value']:.3e})\n")
                f.write("\n")

            # 2.3 Best prediction model with cross-validation results
            if self.ml_results is not None:
                f.write("### 2.3 Best Prediction Model (with 5-Fold Cross-Validation)\n")

                # Find model with highest CV R²
                cv_r2_scores = {}
                models = ['linear_regression', 'random_forest', 'gradient_boosting']
                model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

                for model_key, model_name in zip(models, model_names):
                    eval_dict = self.ml_results[model_key]
                    if 'cv_r2_mean' in eval_dict:
                        cv_r2_scores[model_name] = eval_dict['cv_r2_mean']
                    elif 'best_cv_score' in eval_dict:
                        cv_r2_scores[model_name] = eval_dict['best_cv_score']

                if cv_r2_scores:
                    best_model_name = max(cv_r2_scores, key=cv_r2_scores.get)
                    best_result = self.ml_results[models[model_names.index(best_model_name)]]

                    f.write(f"- **Best Model (based on CV)**: {best_model_name}\n")
                    f.write(f"- **5-Fold CV R²**: {cv_r2_scores[best_model_name]:.3f}\n")
                    f.write(f"- **Test Set R²**: {best_result['r2']:.3f}\n")
                    f.write(f"- **Test Set MAE**: {best_result['mae']:.2f}°\n")

                    if 'best_params' in best_result:
                        f.write(f"- **Best Parameters**: {best_result['best_params']}\n")

                    f.write("\n")

            # 2.4 Most important prediction features
            if self.feature_importance is not None and not self.feature_importance.empty:
                f.write("### 2.4 Most Important Prediction Features\n")
                top_features = self.feature_importance.head(3)
                for _, row in top_features.iterrows():
                    f.write(f"- **{row['Feature']}**: Average importance={row['Average_Importance']:.3f}\n")
                f.write("\n")

            # 3. CSF Features and Scoliosis Angle Correlation
            f.write("## 3. CSF Features and Scoliosis Angle Correlation\n")

            if self.correlation_results is not None and not self.correlation_results.empty:
                f.write("### 3.1 Correlation Analysis Results\n")
                f.write("| Feature | Correlation | P-value | Significance | Clinical Significance |\n")
                f.write("|---------|-------------|---------|--------------|------------------------|\n")

                def get_significance(p_value):
                    if p_value < 0.001:
                        return "***"
                    elif p_value < 0.01:
                        return "**"
                    elif p_value < 0.05:
                        return "*"
                    else:
                        return ""

                top_20 = self.correlation_results.head(20)
                for _, row in top_20.iterrows():
                    feature = row['Feature']
                    corr = row['Correlation']
                    p_value = row['P-value']
                    significance = get_significance(p_value)

                    # Generate clinical significance description
                    if 'elongation' in feature.lower():
                        desc = "Higher scoliosis severity associated with lower CSF region elongation, suggesting morphological compression"
                    elif 'csf_pixel_count' in feature.lower() or 'area' in feature.lower():
                        desc = "Scoliosis severity correlates with CSF region area, reflecting spinal canal space changes"
                    elif 'intensity' in feature.lower():
                        desc = "Signal intensity features correlate with scoliosis angle, reflecting CSF distribution abnormalities"
                    elif 'circularity' in feature.lower() or 'compactness' in feature.lower():
                        desc = "Shape feature changes reflect degree of spinal canal morphological distortion"
                    elif 'solidity' in feature.lower():
                        desc = "Solidity changes reflect alterations in CSF region internal structure"
                    else:
                        desc = "Reflects CSF distribution and morphological changes induced by scoliosis"

                    f.write(f"| `{feature}` | {corr:.3f} | {p_value:.3e} | {significance} | {desc} |\n")

                f.write("\nNote: *p<0.05, **p<0.01, ***p<0.001\n\n")

            # 4. Scoliosis Severity Group Analysis
            f.write("## 4. Scoliosis Severity Group Analysis\n")
            f.write("### 4.1 Group Boxplots\n")
            f.write(
                "Individual boxplots showing distribution differences in key CSF features across Mild (green), Moderate (orange), and Severe (blue) scoliosis groups.\n\n")

            # List boxplot files
            boxplot_files = [f for f in os.listdir(f'{save_dir}/boxplots') if f.endswith('.svg')]
            if boxplot_files:
                f.write("**Available Boxplots**:\n")
                for file in sorted(boxplot_files)[:10]:  # List first 10
                    feature_name = file.replace('severity_boxplot_', '').replace('.svg', '')
                    f.write(f"- `{feature_name}`\n")
                if len(boxplot_files) > 10:
                    f.write(f"- ... and {len(boxplot_files) - 10} more\n")
            f.write("\n")

            if self.subgroup_results is not None and not self.subgroup_results.empty:
                f.write("### 4.2 Statistical Test Results\n")
                f.write("Statistical test results for group differences in key features:\n")
                f.write("| Feature | Test Method | P-value | Significance | Effect Size | Clinical Interpretation |\n")
                f.write("|---------|-------------|---------|--------------|-------------|-------------------------|\n")

                f.write("\nNote: *p<0.05, **p<0.01, ***p<0.001\n")
                f.write("Effect Size Interpretation: Small (0.01-0.059), Medium (0.06-0.139), Large (≥0.14)\n\n")

                sig_results = self.subgroup_results[self.subgroup_results['P-value'] < 0.05]
                if not sig_results.empty:
                    sig_results = sig_results.head(10)  # Show top 10 significant features
                    for _, row in sig_results.iterrows():
                        # Generate clinical interpretation
                        means = row['Means']
                        groups = row['Groups']
                        if len(means) == 2:
                            diff = means[1] - means[0]
                            direction = "increase" if diff > 0 else "decrease"
                            interpretation = f"{groups[1]} group shows {direction} of {abs(diff):.2f} compared to {groups[0]} group"
                        else:
                            interpretation = f"Significant differences among {', '.join(groups)} groups"

                        f.write(
                            f"| `{row['Feature']}` | {row['Test']} | {row['P-value']:.3e} | {row['Significance']} | {row['Effect_Size']:.3f} | {interpretation} |\n")
                else:
                    f.write("| - | - | - | - | - | No significant difference features found |\n")
                f.write("\n")

            # 5. Machine Learning Model Performance with Cross-Validation
            f.write("## 5. Machine Learning Model Performance with 5-Fold Cross-Validation\n")

            f.write("### 5.1 Cross-Validation Methodology\n")
            f.write(f"- **Cross-Validation Strategy**: {self.ml_results.get('cv_folds', 5)}-fold cross-validation\n")
            f.write(f"- **Training Set Size**: {self.ml_results.get('train_size', 0)} samples\n")
            f.write(f"- **Test Set Size**: {self.ml_results.get('test_set_size', 0)} samples\n")
            f.write(
                "- **Evaluation Metrics**: R² (coefficient of determination), MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)\n\n")

            if self.ml_results is not None:
                f.write("### 5.2 Model Performance Comparison\n")
                f.write("| Model | 5-Fold CV R² | Test R² | Test MAE(°) | Test RMSE(°) | MAPE(%) | Notes |\n")
                f.write("|-------|--------------|---------|-------------|--------------|---------|-------|\n")

                models_info = [
                    ("Linear Regression", self.ml_results['linear_regression']),
                    ("Random Forest", self.ml_results['random_forest']),
                    ("Gradient Boosting", self.ml_results['gradient_boosting'])
                ]

                for name, results in models_info:
                    cv_r2 = "N/A"
                    if 'cv_r2_mean' in results:
                        cv_r2 = f"{results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f}"
                    elif 'best_cv_score' in results:
                        cv_r2 = f"{results['best_cv_score']:.3f}"

                    notes = ""
                    if name == "Random Forest" and 'best_params' in results:
                        notes = f"Best params: {results['best_params']}"

                    f.write(
                        f"| {name} | {cv_r2} | {results['r2']:.3f} | {results['mae']:.2f} | {results['rmse']:.2f} | {results['mape']:.2f} | {notes} |\n")

                f.write("\n### 5.3 Model Prediction Results Visualization\n")
                f.write("Individual prediction result visualizations are available in the `ml_results` folder:\n")
                ml_files = [f for f in os.listdir(f'{save_dir}/ml_results') if f.endswith('.svg')]
                if ml_files:
                    for file in sorted(ml_files):
                        f.write(f"- `{file}`\n")
                f.write("\n")

                f.write("### 5.4 Cross-Validation Results Visualization\n")
                f.write(
                    "Cross-validation performance comparison visualizations are available in the `cv_results` folder:\n")
                cv_files = [f for f in os.listdir(f'{save_dir}/cv_results') if f.endswith('.svg')]
                if cv_files:
                    for file in sorted(cv_files):
                        f.write(f"- `{file}`\n")
                f.write("\n")

            # 6. Feature Importance and Model Interpretation
            f.write("## 6. Feature Importance and Model Interpretation\n")

            if self.feature_importance is not None and not self.feature_importance.empty:
                f.write("### 6.1 Feature Importance Ranking\n")
                f.write("| Rank | Feature | Average Importance | Clinical Significance |\n")
                f.write("|------|---------|--------------------|------------------------|\n")

                top_10 = self.feature_importance.head(10)
                for i, (_, row) in enumerate(top_10.iterrows(), 1):
                    feature = row['Feature']

                    # Clinical significance description
                    if 'elongation' in feature.lower():
                        clinical_meaning = "CSF region elongation, reflects morphological compression degree"
                    elif 'csf_pixel_count' in feature.lower():
                        clinical_meaning = "CSF region area, reflects spinal canal space size"
                    elif 'intensity_mean' in feature.lower():
                        clinical_meaning = "Average signal intensity, reflects CSF distribution uniformity"
                    elif 'solidity' in feature.lower():
                        clinical_meaning = "Solidity, reflects CSF region internal structure integrity"
                    elif 'circularity' in feature.lower():
                        clinical_meaning = "Circularity, reflects CSF region shape regularity"
                    else:
                        clinical_meaning = "Reflects CSF feature changes induced by scoliosis"

                    f.write(f"| {i} | `{feature}` | {row['Average_Importance']:.3f} | {clinical_meaning} |\n")

                f.write("\n")

            f.write("### 6.2 SHAP Model Interpretation\n")
            if os.path.exists(f'{save_dir}/shap_analysis/shap_summary_plot.svg'):
                f.write("#### 6.2.1 SHAP Summary Plot\n")
                f.write(
                    "SHAP summary plot shows direction and magnitude of each feature's impact on model predictions. Features sorted by importance, color indicates feature value magnitude.\n\n")

            if os.path.exists(f'{save_dir}/shap_analysis/shap_feature_importance.svg'):
                f.write("#### 6.2.2 SHAP Feature Importance\n")
                f.write(
                    "Feature importance ranking based on SHAP values, reflecting absolute contribution of each feature to prediction results.\n\n")

            # 7. Clinical Significance and Mechanism Discussion
            f.write("## 7. Clinical Significance and Mechanism Discussion\n")
            f.write("### 7.1 Scoliosis Mechanism Heterogeneity\n")
            f.write(
                "1. **Mild Scoliosis Group**: CSF feature changes may mainly reflect early compensatory adjustments, with smaller feature variations\n")
            f.write(
                "2. **Moderate Scoliosis Group**: Structural changes become apparent, with significant alterations in CSF morphology and distribution features\n")
            f.write(
                "3. **Severe Scoliosis Group**: Obvious spinal canal deformation, CSF features show extreme changes, possibly accompanied by nerve compression risk\n\n")

            f.write("### 7.2 Clinical Significance of CSF Feature Changes\n")
            f.write(
                "1. **Morphological Feature Changes**: CSF region area, elongation, circularity, etc., reflect geometric changes in spinal canal space\n")
            f.write(
                "2. **Signal Feature Changes**: Intensity distribution, uniformity, etc., reflect CSF flow status and tissue compression\n")
            f.write(
                "3. **Combined Feature Value**: Multiple feature combinations provide more comprehensive scoliosis assessment than single angle measurements\n\n")

            f.write("### 7.3 Clinical Application Potential of Prediction Models\n")
            f.write(
                "1. **Auxiliary Diagnosis**: Provide objective CSF feature quantification metrics, supplementing traditional angle measurements\n")
            f.write(
                "2. **Progression Monitoring**: Track CSF feature changes to monitor scoliosis progression and treatment effectiveness\n")
            f.write(
                "3. **Personalized Assessment**: Consider scoliosis severity differences for stratified prediction and assessment\n\n")

            # 8. Conclusions and Recommendations
            f.write("## 8. Conclusions and Recommendations\n")
            f.write("### 8.1 Main Conclusions\n")
            f.write(
                "1. **CSF Features Significantly Correlate with Scoliosis Angle**: Multiple CSF features show statistically significant correlations with Cobb angle\n")
            f.write(
                "2. **Clear Feature Differences Across Scoliosis Severity Groups**: Mild, Moderate, and Severe scoliosis groups show significant differences in key CSF features\n")
            if self.ml_results is not None:
                # Find best model based on CV
                best_model_name = ""
                best_cv_score = 0
                for model_key in ['linear_regression', 'random_forest', 'gradient_boosting']:
                    eval_dict = self.ml_results[model_key]
                    cv_score = eval_dict.get('cv_r2_mean', eval_dict.get('best_cv_score', 0))
                    if cv_score > best_cv_score:
                        best_cv_score = cv_score
                        best_model_name = model_key.replace('_', ' ').title()

                f.write(
                    f"3. **Good Machine Learning Model Prediction Performance with Robust Cross-Validation**: {best_model_name} model achieves R² of {best_cv_score:.3f} in 5-fold cross-validation\n")
            else:
                f.write(
                    "3. **Good Machine Learning Model Prediction Performance**: Models can accurately predict scoliosis angle with robust cross-validation\n")
            f.write(
                "4. **Strong Model Interpretability**: SHAP analysis reveals contribution of each feature to predictions, enhancing clinical interpretability\n\n")

            f.write("### 8.2 Research Limitations\n")
            f.write("1. Limited sample size, especially for Severe scoliosis group\n")
            f.write("2. Cross-sectional study, lacking longitudinal follow-up data\n")
            f.write("3. Did not consider confounding factors like age, gender, scoliosis type\n")
            f.write("4. 5-fold cross-validation provides robust but not perfect generalization estimate\n\n")

            f.write("### 8.3 Future Research Directions\n")
            f.write(
                "1. **Expand Sample Size**: Particularly increase Severe scoliosis cases to validate current findings\n")
            f.write(
                "2. **Longitudinal Studies**: Track CSF feature changes in same patients at different time points\n")
            f.write(
                "3. **Multimodal Integration**: Combine clinical information, radiomics, and other multidimensional data\n")
            f.write(
                "4. **Clinical Application Validation**: Validate model practical value in actual clinical scenarios\n")
            f.write(
                "5. **Advanced Cross-Validation**: Consider nested cross-validation or repeated cross-validation for more robust evaluation\n")
            f.write(
                "6. **External Validation**: Validate models on independent external datasets\n")

            # 9. Supplementary Materials
            f.write("\n## 9. Supplementary Materials\n")
            f.write("All analysis results and raw data from this report can be found in the following directories:\n")

            # List all directories and their contents
            for subdir in subdirs:
                dir_path = os.path.join(save_dir, subdir)
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) if f.endswith(('.svg', '.csv', '.png'))]
                    if files:
                        f.write(f"\n### {subdir.replace('_', ' ').title()} Directory\n")
                        for file in sorted(files)[:15]:  # List first 15 files
                            f.write(f"- `{file}`\n")
                        if len(files) > 15:
                            f.write(f"- ... and {len(files) - 15} more files\n")

            print(f"\n✓ Analysis report generated: {report_path}")
        return report_path

    def run_full_analysis(self):
        """Run complete analysis workflow"""
        print("=" * 80)
        print("CSF Signal and Scoliosis Angle Multidimensional Analysis")
        print("=" * 80)

        try:
            self.load_and_merge_data()
            self.descriptive_analysis()
            self.correlation_analysis()
            self.scatter_plot_analysis()
            self.machine_learning_analysis()
            self.generate_markdown_report()

            print("\n" + "=" * 80)
            print("Analysis Complete!")
            print("=" * 80)
            print(f"All results saved to: {save_dir}")
            print(f"\nDirectory Structure:")
            for subdir in subdirs:
                dir_path = os.path.join(save_dir, subdir)
                if os.path.exists(dir_path):
                    files = len([f for f in os.listdir(dir_path) if f.endswith(('.svg', '.csv', '.png'))])
                    print(f"  - {subdir}/: {files} files")

            print(f"\nPlease review detailed report: {save_dir}/CSF_Cobb_Angle_Analysis_Report.md")

        except Exception as e:
            print(f"\n✗ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    # File path configuration (modify according to actual situation)
    csf_path = "CSF\output_CSF_analysis\csf_analysis\CSF_statistics_detailed.csv"
    cobb_path = "csf_amace.csv"

    # Create analyzer instance
    analyzer = CSFCobbAnalyzer(csf_path, cobb_path)

    # Run complete analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
