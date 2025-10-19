@echo off
REM Training script for Real vs AI Detector
REM Run this to start training in a dedicated window

echo ========================================
echo Real vs AI Image Detector - Training
echo ========================================
echo.
echo Model: Hybrid (EfficientNet-B0 + Stats)
echo Dataset: 200,000 images
echo Epochs: 10
echo Batch Size: 32
echo.
echo Training will take approximately 5-7 hours
echo Progress will be displayed below
echo.
echo ========================================
echo.

"D:/Farm Fresh/new/Gen/camera-vs-ai/.venv312/Scripts/python.exe" scripts/train.py --train-dir "archive (2)/my_real_vs_ai_dataset/my_real_vs_ai_dataset" --val-dir "archive (2)/my_real_vs_ai_dataset/my_real_vs_ai_dataset" --epochs 10 --batch-size 32 --model hybrid --lr 0.0001

echo.
echo ========================================
echo Training Complete!
echo Check checkpoints/ folder for models
echo ========================================
pause
