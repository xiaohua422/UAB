UAB-DeepLabV3+:Lumbar Scoliosis Pathological Detection and Multi-Dimensional Quantification

### 所需环境

torch==1.2.0

 

# Project Strucure     

```
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
├── three_method_analysis_tardition_amace_hough_contour.py
├── json_to_dataset.py         # JSON annotation to dataset
├── voc_annotation.py          # VOC format annotation
├── requirements.txt            # Dependencies
└── README.md
```

# Usage

## Dataset Preparation

```
python json_to_dataset.py
python voc_annotation.py
```

## Model Training

```
python train_phase1.py
python train_ablation.py
```

## Segmentation Prediction

```
python predict.py
```

## Evaluation Metrics

```
python get_mDice.py
python get_miou.py
python single_mDice.py
python single_mIou.py
```

## Angle Calculation

```
python cobb8.py
python every_amace_angle.py
python every_amace_cobb_analysis.py
```

## Visualization & Analysis

```
python CSF_plot.py
python three_method_analysis_tardition_amace_hough_contour.py
```

