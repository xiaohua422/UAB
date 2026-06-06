import os
import cv2
import numpy as np
import pandas as pd
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from skimage import measure, morphology
from skimage.morphology import skeletonize
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csf_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CSFStatistics:
    filename: str
    csf_pixel_count: int
    total_pixels: int
    csf_percentage: float
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    intensity_median: float
    intensity_q25: float
    intensity_q75: float
    intensity_iqr: float
    intensity_skewness: float
    intensity_kurtosis: float
    intensity_entropy: float
    contour_count: int
    largest_contour_area: float
    compactness: float
    circularity: float
    convexity: float
    elongation: float
    orientation_angle: float
    bounding_box_x: int
    bounding_box_y: int
    bounding_box_width: int
    bounding_box_height: int
    aspect_ratio: float
    centroid_x: float
    centroid_y: float
    signal_mean: float
    signal_std: float
    signal_skewness: float
    signal_kurtosis: float
    signal_entropy: float
    solidity: float
    eccentricity: float
    equivalent_diameter: float
    perimeter: float
    area_ratio: float
    extent: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_dataframe_row(self) -> Dict:
        data = self.to_dict()
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 4)
        return data


class CSFAnalyzer:
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 save_dir: str,
                 csf_label: int = 12,
                 pixel_to_mm: Optional[float] = None,
                 config: Optional[Dict] = None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.save_dir = Path(save_dir)
        self.csf_label = csf_label
        self.pixel_to_mm = pixel_to_mm

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            'smooth_kernel_size': 9,
            'smooth_sigma': 3,
            'savgol_window': 15,
            'savgol_polyorder': 3,
            'min_contour_area': 10,
            'visualization_dpi': 150,
            'heatmap_colormap': 'jet',
            'supported_image_exts': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
            'report_figsize': (15, 12),
            'curve_figsize': (10, 6),
            'output_format': 'svg'
        }

        if config:
            self.config.update(config)

        self.results: List[CSFStatistics] = []
        self.signal_profiles: Dict[str, np.ndarray] = {}

        logger.info(f"CSF Analyzer initialized: img_dir={img_dir}, mask_dir={mask_dir}")
        logger.info(f"Save directory: {save_dir}")

    def find_image_file(self, base_name: str) -> Optional[Path]:
        for ext in self.config['supported_image_exts']:
            img_path = self.img_dir / (base_name + ext)
            if img_path.exists():
                return img_path
        return None

    def calculate_intensity_statistics(self, gray_img: np.ndarray, csf_mask: np.ndarray) -> Dict[str, float]:
        csf_intensities = gray_img[csf_mask == 255]

        if len(csf_intensities) == 0:
            return {
                'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0,
                'q25': 0, 'q75': 0, 'iqr': 0, 'skewness': 0,
                'kurtosis': 0, 'entropy': 0
            }

        intensity_min = np.min(csf_intensities)
        intensity_max = np.max(csf_intensities)
        intensity_mean = np.mean(csf_intensities)
        intensity_std = np.std(csf_intensities)
        intensity_median = np.median(csf_intensities)

        intensity_q25, intensity_q75 = np.percentile(csf_intensities, [25, 75])
        intensity_iqr = intensity_q75 - intensity_q25

        intensity_skewness = stats.skew(csf_intensities)
        intensity_kurtosis = stats.kurtosis(csf_intensities)

        hist, _ = np.histogram(csf_intensities, bins=32, range=(0, 255))
        hist_norm = hist / hist.sum()
        intensity_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

        return {
            'min': float(intensity_min),
            'max': float(intensity_max),
            'mean': float(intensity_mean),
            'std': float(intensity_std),
            'median': float(intensity_median),
            'q25': float(intensity_q25),
            'q75': float(intensity_q75),
            'iqr': float(intensity_iqr),
            'skewness': float(intensity_skewness),
            'kurtosis': float(intensity_kurtosis),
            'entropy': float(intensity_entropy)
        }

    def calculate_morphological_features(self, csf_mask: np.ndarray, contour: np.ndarray) -> Dict[str, float]:
        features = {}

        if contour is None or len(contour) < 3:
            return {
                'area': 0, 'perimeter': 0, 'compactness': 0, 'circularity': 0,
                'convexity': 0, 'elongation': 0, 'orientation': 0,
                'solidity': 0, 'eccentricity': 0, 'equivalent_diameter': 0,
                'extent': 0
            }

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), angle = rect
        elongation = max(width, height) / min(width, height) if min(width, height) > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (ma, MA), _ = ellipse
            eccentricity = np.sqrt(1 - (ma ** 2) / (MA ** 2)) if MA > 0 else 0
        else:
            eccentricity = 0

        equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'circularity': float(circularity),
            'convexity': float(convexity),
            'elongation': float(elongation),
            'orientation': float(angle),
            'solidity': float(solidity),
            'eccentricity': float(eccentricity),
            'equivalent_diameter': float(equivalent_diameter),
            'extent': float(extent)
        }

    def extract_signal_profile(self, gray_img: np.ndarray, csf_mask: np.ndarray, bounding_box: Tuple) -> np.ndarray:
        x, y, w, h = bounding_box
        csf_crop_mask = csf_mask[y:y + h, x:x + w]
        gray_crop = gray_img[y:y + h, x:x + w]

        signal_distribution = []
        for row in range(h):
            row_pixels = gray_crop[row, :][csf_crop_mask[row, :] == 255]
            if len(row_pixels) > 0:
                signal_distribution.append(np.mean(row_pixels))
            else:
                signal_distribution.append(0)

        signal_array = np.array(signal_distribution)

        if len(signal_array) >= self.config['savgol_window']:
            try:
                signal_smoothed = savgol_filter(
                    signal_array,
                    self.config['savgol_window'],
                    self.config['savgol_polyorder']
                )
                return signal_smoothed
            except:
                return signal_array
        else:
            return signal_array

    def calculate_signal_features(self, signal: np.ndarray) -> Dict[str, float]:
        if len(signal) == 0 or np.all(signal == 0):
            return {
                'mean': 0, 'std': 0, 'skewness': 0,
                'kurtosis': 0, 'entropy': 0
            }

        signal_nonzero = signal[signal > 0]
        if len(signal_nonzero) == 0:
            return {
                'mean': 0, 'std': 0, 'skewness': 0,
                'kurtosis': 0, 'entropy': 0
            }

        signal_mean = np.mean(signal_nonzero)
        signal_std = np.std(signal_nonzero)
        signal_skewness = stats.skew(signal_nonzero)
        signal_kurtosis = stats.kurtosis(signal_nonzero)

        hist, _ = np.histogram(signal_nonzero, bins=32)
        hist_norm = hist / hist.sum()
        signal_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

        return {
            'mean': float(signal_mean),
            'std': float(signal_std),
            'skewness': float(signal_skewness),
            'kurtosis': float(signal_kurtosis),
            'entropy': float(signal_entropy)
        }

    def generate_comprehensive_visualization(self, img: np.ndarray, mask: np.ndarray, csf_mask: np.ndarray,
                                             statistics: CSFStatistics, signal_profile: np.ndarray, base_name: str):
        sample_dir = self.save_dir / base_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_fmt = self.config['output_format']
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(csf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original MRI Image')
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(sample_dir / f"01_original_image.{output_fmt}", format=output_fmt)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        overlay = img.copy()
        overlay[csf_mask == 255] = [0, 0, 255]
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title('CSF Region Overlay')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(sample_dir / f"02_csf_overlay.{output_fmt}", format=output_fmt)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        ax3.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        ax3.set_title('CSF Boundary Contours')
        ax3.axis('off')
        plt.tight_layout()
        plt.savefig(sample_dir / f"03_csf_contours.{output_fmt}", format=output_fmt)
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        csf_intensities = gray_img[csf_mask == 255]
        if len(csf_intensities) > 0:
            ax4.hist(csf_intensities, bins=32, alpha=0.7, color='blue', edgecolor='black', density=True)
            ax4.axvline(statistics.intensity_mean, color='red', linestyle='--', label=f'Mean: {statistics.intensity_mean:.1f}')
            ax4.axvline(statistics.intensity_median, color='green', linestyle='--', label=f'Median: {statistics.intensity_median:.1f}')
            ax4.legend()
        ax4.set_title('CSF Intensity Distribution')
        ax4.set_xlabel('Intensity')
        ax4.set_ylabel('Density')
        plt.tight_layout()
        plt.savefig(sample_dir / f"04_intensity_histogram.{output_fmt}", format=output_fmt)
        plt.close(fig4)

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        if len(signal_profile) > 0:
            ax5.plot(signal_profile, 'b-', linewidth=2, label='Original Signal')
            if len(signal_profile) >= self.config['savgol_window']:
                signal_smooth = savgol_filter(signal_profile, self.config['savgol_window'], self.config['savgol_polyorder'])
                ax5.plot(signal_smooth, 'r-', linewidth=1.5, alpha=0.7, label='Smoothed Signal')
            ax5.legend()
            ax5.set_title('CSF Signal Distribution')
            ax5.set_xlabel('Position (pixels)')
            ax5.set_ylabel('Average Intensity')
            ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sample_dir / f"05_signal_curve.{output_fmt}", format=output_fmt)
        plt.close(fig5)

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        if len(signal_profile) > 0:
            heatmap_data = np.tile(signal_profile, (20, 1)).T
            im = ax6.imshow(heatmap_data, cmap=self.config['heatmap_colormap'], aspect='auto', interpolation='bilinear')
            ax6.set_title('Signal Heatmap')
            ax6.set_xlabel('Width')
            ax6.set_ylabel('Position')
            plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(sample_dir / f"06_signal_heatmap.{output_fmt}", format=output_fmt)
        plt.close(fig6)

        fig7, ax7 = plt.subplots(figsize=(6, 6))
        if contours and len(contours) > 0:
            contour_areas = []
            contour_centroids_x = []
            contour_centroids_y = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.config['min_contour_area']:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        contour_areas.append(area)
                        contour_centroids_x.append(cx)
                        contour_centroids_y.append(cy)
            if contour_areas:
                scatter = ax7.scatter(contour_centroids_x, contour_centroids_y,
                                      s=np.array(contour_areas) / 10, c=contour_areas, cmap='viridis', alpha=0.6)
                ax7.set_title('Contour Distribution')
                ax7.set_xlabel('X Coordinate')
                ax7.set_ylabel('Y Coordinate')
                ax7.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax7, label='Area')
        plt.tight_layout()
        plt.savefig(sample_dir / f"07_contour_distribution.{output_fmt}", format=output_fmt)
        plt.close(fig7)

        fig8, ax8 = plt.subplots(figsize=(4, 6))
        if len(signal_profile) > 0 and np.any(signal_profile > 0):
            signal_nonzero = signal_profile[signal_profile > 0]
            bp = ax8.boxplot(signal_nonzero, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['medians'][0].set_color('red')
            ax8.set_title('Signal Intensity Box Plot')
            ax8.set_ylabel('Intensity Value')
            ax8.set_xticks([1])
            ax8.set_xticklabels(['CSF Signal'])
            ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sample_dir / f"08_signal_boxplot.{output_fmt}", format=output_fmt)
        plt.close(fig8)

        logger.info(f"Comprehensive visualizations saved to {sample_dir}")

    def generate_curve_comparison(self, signal_profile: np.ndarray, base_name: str) -> str:
        if len(signal_profile) == 0:
            return ""

        sample_dir = self.save_dir / base_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_fmt = self.config['output_format']

        plt.figure(figsize=self.config['curve_figsize'])
        plt.plot(signal_profile, 'b-', linewidth=1.5, label='Original Signal', alpha=0.7)

        if len(signal_profile) > 3:
            kernel_size = min(len(signal_profile), self.config['smooth_kernel_size'])
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size >= 3:
                gaussian_smoothed = cv2.GaussianBlur(signal_profile.reshape(-1, 1), (kernel_size, kernel_size),
                                                     self.config['smooth_sigma']).flatten()
                plt.plot(gaussian_smoothed, 'g-', linewidth=2, label='Gaussian Smoothing', alpha=0.8)

        if len(signal_profile) >= self.config['savgol_window']:
            try:
                sg_smoothed = savgol_filter(signal_profile, self.config['savgol_window'], self.config['savgol_polyorder'])
                plt.plot(sg_smoothed, 'r-', linewidth=2, label='Savitzky-Golay Filter', alpha=0.8)
            except:
                pass

        if len(signal_profile) > 5:
            window_size = min(7, len(signal_profile))
            moving_avg = np.convolve(signal_profile, np.ones(window_size) / window_size, mode='same')
            plt.plot(moving_avg, 'm-', linewidth=1.5, label=f'{window_size}-point Moving Average', alpha=0.7)

        plt.xlabel('Position (pixels)')
        plt.ylabel('Average Intensity')
        plt.title(f'CSF Signal Distribution - {base_name}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')

        if np.any(signal_profile > 0):
            signal_nonzero = signal_profile[signal_profile > 0]
            stats_text = f'Mean: {np.mean(signal_nonzero):.1f}\nStd: {np.std(signal_nonzero):.1f}\nLength: {len(signal_profile)} points'
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=8,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

        curve_path = sample_dir / f"signal_curve_comparison.{output_fmt}"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight', format=output_fmt)
        plt.close()
        logger.info(f"Curve comparison saved: {curve_path}")
        return str(curve_path)

    def process_single_image(self, mask_file: str) -> Optional[CSFStatistics]:
        if not mask_file.endswith('.png'):
            return None

        mask_path = self.mask_dir / mask_file
        base_name = Path(mask_file).stem

        img_path = self.find_image_file(base_name)
        if not img_path:
            logger.warning(f"Original image not found, skipping: {mask_file}")
            return None

        logger.info(f"Processing file: {mask_file} -> Original: {img_path.name}")

        try:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                logger.error(f"Cannot read original image: {img_path}")
                return None
            if mask is None:
                logger.error(f"Cannot read mask: {mask_path}")
                return None

            height, width = mask.shape[:2]
            total_pixels = height * width
            csf_mask = np.where(mask == self.csf_label, 255, 0).astype(np.uint8)
            csf_pixel_count = np.sum(csf_mask == 255)

            if csf_pixel_count == 0:
                logger.info(f"{mask_file} has no CSF region detected")
                return CSFStatistics(
                    filename=mask_file, csf_pixel_count=0, total_pixels=total_pixels, csf_percentage=0.0,
                    intensity_min=0, intensity_max=0, intensity_mean=0.0, intensity_std=0.0, intensity_median=0.0,
                    intensity_q25=0.0, intensity_q75=0.0, intensity_iqr=0.0, intensity_skewness=0.0, intensity_kurtosis=0.0,
                    intensity_entropy=0.0, contour_count=0, largest_contour_area=0.0, compactness=0.0, circularity=0.0,
                    convexity=0.0, elongation=0.0, orientation_angle=0.0, bounding_box_x=0, bounding_box_y=0,
                    bounding_box_width=0, bounding_box_height=0, aspect_ratio=0.0, centroid_x=0.0, centroid_y=0.0,
                    signal_mean=0.0, signal_std=0.0, signal_skewness=0.0, signal_kurtosis=0.0, signal_entropy=0.0,
                    solidity=0.0, eccentricity=0.0, equivalent_diameter=0.0, perimeter=0.0, area_ratio=0.0, extent=0.0
                )

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            intensity_stats = self.calculate_intensity_statistics(gray_img, csf_mask)
            contours, _ = cv2.findContours(csf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)

            largest_contour = None
            largest_contour_area = 0.0
            if contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) >= self.config['min_contour_area']]
                contour_count = len(valid_contours)
                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    largest_contour_area = cv2.contourArea(largest_contour)

            bounding_box = (0, 0, 0, 0)
            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                bounding_box = (x, y, w, h)
                aspect_ratio = w / h if h > 0 else 0.0
            else:
                x, y, w, h = 0, 0, 0, 0
                aspect_ratio = 0.0

            M = cv2.moments(csf_mask)
            centroid_x = M['m10'] / M['m00'] if M['m00'] != 0 else 0.0
            centroid_y = M['m01'] / M['m00'] if M['m00'] != 0 else 0.0

            signal_profile = self.extract_signal_profile(gray_img, csf_mask, bounding_box)
            self.signal_profiles[base_name] = signal_profile
            signal_features = self.calculate_signal_features(signal_profile)
            morph_features = self.calculate_morphological_features(csf_mask, largest_contour)
            area_ratio = largest_contour_area / csf_pixel_count if csf_pixel_count > 0 else 0.0

            stats = CSFStatistics(
                filename=mask_file,
                csf_pixel_count=csf_pixel_count,
                total_pixels=total_pixels,
                csf_percentage=(csf_pixel_count / total_pixels) * 100,
                intensity_min=intensity_stats['min'],
                intensity_max=intensity_stats['max'],
                intensity_mean=intensity_stats['mean'],
                intensity_std=intensity_stats['std'],
                intensity_median=intensity_stats['median'],
                intensity_q25=intensity_stats['q25'],
                intensity_q75=intensity_stats['q75'],
                intensity_iqr=intensity_stats['iqr'],
                intensity_skewness=intensity_stats['skewness'],
                intensity_kurtosis=intensity_stats['kurtosis'],
                intensity_entropy=intensity_stats['entropy'],
                contour_count=contour_count,
                largest_contour_area=largest_contour_area,
                compactness=morph_features['compactness'],
                circularity=morph_features['circularity'],
                convexity=morph_features['convexity'],
                elongation=morph_features['elongation'],
                orientation_angle=morph_features['orientation'],
                bounding_box_x=x,
                bounding_box_y=y,
                bounding_box_width=w,
                bounding_box_height=h,
                aspect_ratio=aspect_ratio,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                signal_mean=signal_features['mean'],
                signal_std=signal_features['std'],
                signal_skewness=signal_features['skewness'],
                signal_kurtosis=signal_features['kurtosis'],
                signal_entropy=signal_features['entropy'],
                solidity=morph_features['solidity'],
                eccentricity=morph_features['eccentricity'],
                equivalent_diameter=morph_features['equivalent_diameter'],
                perimeter=morph_features['perimeter'],
                area_ratio=area_ratio,
                extent=morph_features['extent']
            )

            if csf_pixel_count > 0:
                result_img = img.copy()
                if largest_contour is not None:
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.circle(result_img, (int(centroid_x), int(centroid_y)), 4, (0, 255, 0), -1)
                output_path = self.save_dir / mask_file
                cv2.imwrite(str(output_path), result_img)

                self.generate_curve_comparison(signal_profile, base_name)
                self.generate_comprehensive_visualization(img, mask, csf_mask, stats, signal_profile, base_name)

            logger.info(f"Processing complete: {mask_file}, CSF pixel count: {csf_pixel_count}")
            return stats

        except Exception as e:
            logger.error(f"Error processing file {mask_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def analyze_batch(self) -> pd.DataFrame:
        logger.info("Starting batch CSF analysis...")
        mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith('.png')]
        logger.info(f"Found {len(mask_files)} mask files")

        for mask_file in tqdm(mask_files, desc="Processing CSF analysis"):
            stats = self.process_single_image(mask_file)
            if stats:
                self.results.append(stats)

        self.save_results()
        logger.info(f"Batch processing complete! Processed {len(self.results)} files")
        return self.results_to_dataframe()

    def save_results(self):
        csv_path = self.save_dir / "CSF_statistics_detailed.csv"
        df = self.results_to_dataframe()
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        excel_path = self.save_dir / "CSF_statistics_detailed.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detailed Statistics', index=False)
            summary_df = self.generate_summary_statistics()
            summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            corr_df = df.select_dtypes(include=[np.number]).corr()
            corr_df.to_excel(writer, sheet_name='Feature Correlation')

        signal_path = self.save_dir / "signal_profiles.json"
        signal_data = {k: v.tolist() for k, v in self.signal_profiles.items()}
        with open(signal_path, 'w') as f:
            json.dump(signal_data, f)

        config_path = self.save_dir / "analysis_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Results saved: {csv_path}")
        logger.info(f"Detailed report saved: {excel_path}")

    def results_to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = [stats.to_dataframe_row() for stats in self.results]
        return pd.DataFrame(rows)

    def generate_summary_statistics(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        df = self.results_to_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary_data = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_data.append({
                    'Feature': col,
                    'Count': len(col_data),
                    'Mean': round(col_data.mean(), 4),
                    'Std': round(col_data.std(), 4),
                    'Min': round(col_data.min(), 4),
                    '25%': round(col_data.quantile(0.25), 4),
                    'Median': round(col_data.median(), 4),
                    '75%': round(col_data.quantile(0.75), 4),
                    'Max': round(col_data.max(), 4),
                    'Missing': df[col].isna().sum()
                })
        return pd.DataFrame(summary_data)

    def generate_batch_report(self):
        if not self.results:
            logger.warning("No analysis results, cannot generate batch report")
            return

        df = self.results_to_dataframe()
        output_fmt = self.config['output_format']

        if 'csf_pixel_count' in df.columns:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.hist(df['csf_pixel_count'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(df['csf_pixel_count'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df["csf_pixel_count"].mean():.0f}')
            ax1.set_xlabel('CSF Pixel Count')
            ax1.set_ylabel('Frequency')
            ax1.set_title('CSF Area Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / f"batch_csf_area_distribution.{output_fmt}", format=output_fmt)
            plt.close(fig1)

        if 'csf_percentage' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.hist(df['csf_percentage'], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(df['csf_percentage'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df["csf_percentage"].mean():.2f}%')
            ax2.set_xlabel('CSF Percentage (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('CSF Percentage Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / f"batch_csf_percentage_distribution.{output_fmt}", format=output_fmt)
            plt.close(fig2)

        if 'csf_pixel_count' in df.columns and 'intensity_mean' in df.columns:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            scatter = ax3.scatter(df['csf_pixel_count'], df['intensity_mean'],
                                  c=df.get('csf_percentage', 50), cmap='viridis', alpha=0.6, s=50, edgecolor='k', linewidth=0.5)
            ax3.set_xlabel('CSF Pixel Count')
            ax3.set_ylabel('Intensity Mean')
            ax3.set_title('Area vs Intensity')
            ax3.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
            cbar.set_label('CSF Percentage (%)')
            plt.tight_layout()
            plt.savefig(self.save_dir / f"batch_area_vs_intensity.{output_fmt}", format=output_fmt)
            plt.close(fig3)

        morph_features = ['compactness', 'circularity', 'elongation', 'solidity']
        available_features = [f for f in morph_features if f in df.columns]
        if len(available_features) >= 1:
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            box_data = [df[f].dropna() for f in available_features]
            bp = ax4.boxplot(box_data, patch_artist=True, labels=available_features)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(available_features)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            for median in bp['medians']:
                median.set_color('red')
            ax4.set_ylabel('Value')
            ax4.set_title('Morphological Feature Distribution')
            ax4.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.save_dir / f"batch_morphological_features.{output_fmt}", format=output_fmt)
            plt.close(fig4)

        logger.info("Batch analysis reports generated as separate images.")


def main():
    CONFIG = {
        'img_dir': r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\img",
        'mask_dir': r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\mask",
        'save_dir': r"D:\unet_test\Deeplabv3+\Deeplabv3_plus_ours\deeplabv3-plus-pytorch-main-ours\CSF\output_CSF_analysis_csf_analysis_svg",
        'csf_label': 12,
        'pixel_to_mm': None,
        'analysis_config': {
            'smooth_kernel_size': 9,
            'smooth_sigma': 3,
            'savgol_window': 15,
            'savgol_polyorder': 3,
            'min_contour_area': 10,
            'visualization_dpi': 150,
            'heatmap_colormap': 'jet',
            'report_figsize': (15, 12),
            'curve_figsize': (10, 6),
            'output_format': 'svg'
        }
    }

    try:
        analyzer = CSFAnalyzer(
            img_dir=CONFIG['img_dir'],
            mask_dir=CONFIG['mask_dir'],
            save_dir=CONFIG['save_dir'],
            csf_label=CONFIG['csf_label'],
            pixel_to_mm=CONFIG['pixel_to_mm'],
            config=CONFIG['analysis_config']
        )

        results_df = analyzer.analyze_batch()
        analyzer.generate_batch_report()

        print("\n" + "=" * 60)
        print("CSF Analysis Complete!")
        print("=" * 60)
        print(f"Files analyzed: {len(analyzer.results)}")
        print(f"Results directory: {analyzer.save_dir}")
        print(f"Detailed statistics: {analyzer.save_dir}/CSF_statistics_detailed.csv")
        print(f"Excel report: {analyzer.save_dir}/CSF_statistics_detailed.xlsx")
        print("=" * 60)

        if not results_df.empty:
            print("\nSummary Statistics:")
            print("-" * 40)
            total_pixels = results_df['csf_pixel_count'].sum()
            mean_percentage = results_df['csf_percentage'].mean()
            mean_intensity = results_df['intensity_mean'].mean()
            print(f"Total CSF pixels: {total_pixels:,}")
            print(f"Average CSF percentage: {mean_percentage:.2f}%")
            print(f"Average intensity: {mean_intensity:.1f}")
            print(f"Average contour count: {results_df['contour_count'].mean():.1f}")
            print(f"Average circularity: {results_df['circularity'].mean():.3f}")

        print("\nAll analysis tasks completed!")

    except Exception as e:
        logger.error(f"Main program error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Program error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
