UAB-DeepLabV3+:Lumbar Scoliosis Pathological Detection and Multi-Dimensional Quantification

---

### 目录

1. [仓库更新 Top News](#仓库更新)

2. [相关仓库 Related code](#相关仓库)

3. [性能情况 Performance](#性能情况)

4. [所需环境 Environment](#所需环境)

5. [文件下载 Download](#文件下载)

6. [训练步骤 How2train](#训练步骤)

7. [预测步骤 How2predict](#预测步骤)

8. [评估步骤 miou](#评估步骤)

9. [参考资料 Reference](#Reference)

   





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

