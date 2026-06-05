
# UAB-DeepLabV3+: Lumbar Scoliosis Pathological Detection and Multi-Dimensional Quantification
This repository contains full implementation for lumbar MRI semantic segmentation, automatic scoliosis angle measurement, segmentation metrics calculation and result statistical analysis.

## 1. Core Functions
1. MRI spinal tissue segmentation based on improved UAB-DeepLabV3+
2. Automatic spinal curvature measurement
3. Batch calculation of segmentation metrics (mDice, mIoU, DHI etc.)
4. CSF visualization, error statistics and multi-model comparative analysis

## 2. Environment Installation
Python: 3.8 | PyTorch: 2.4.1 | CUDA:12.1
```bash
pip install -r requirements.txt
```

## 3. Project Structure
```
├── VOCdevkit/                 # VOC标注数据集目录
├── img/                       # Original input MRI images
├── nets/                      # Backbone & network auxiliary files
├── phase_img_out/             # Model segmentation output images
├── deeplab.py                 # UAB-DeepLabV3+ model structure
├── train_phase1.py            # Main training script
├── train_ablation.py          # Ablation study training
├── predict.py                 # Inference and segmentation
├── cobb8.py                   # Cobb angle calculation
├── cobb10.py                  # AMACE angle calculation
├── cobb12.py                  # hough angle calculation
├── cobb13.py                  # contour angle calculation
├── every_amace_angle.py       # AMACE angle measurement
├── every_amace_cobb_analysis.py
├── get_mDice.py               # Dice coefficient
├── get_miou.py                # IoU metric
├── CSF_plot.py                # CSF feature visualization
├── csf_signal_visualizer.py
├── three_method_analysis_tardition_amace_hough_contour.py       # Result error & comparison analysis
├── json_to_dataset.py         # JSON annotation to dataset
├── voc_annotation.py          # VOC format annotation
├── requirements.txt            # Dependencies
```

## 4. Usage
### Dataset Preparation
```
python json_to_dataset.py
python voc_annotation.py
```

### Model Training
```
python train_phase1.py
python train_ablation.py
```

### Segmentation Prediction
```
python predict.py
```

### Evaluation Metrics
```
python get_mDice.py
python get_miou.py
python single_mDice.py
python single_mIou.py
```

### Angle Calculation
```
python cobb8.py
python every_amace_angle.py
python every_amace_cobb_analysis.py
```

### Visualization & Analysis
```
python CSF_plot.py
python three_method_analysis_tardition_amace_hough_contour.py
```
