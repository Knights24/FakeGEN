# Classical Forensics Suite

This module adds classical forensic feature extractors and a CLI pipeline for images/videos:

- Laplacian noise estimators: variance, local noise maps, global sigma
- PRNU: residual extraction, fingerprint estimation and correlation
- JPEG/DCT artifacts: blockiness and low-vs-high frequency energy ratio
- Upscaling artifacts: radial power spectrum, high-frequency ratio, aliasing score

Quick start

1. Install dependencies (Windows PowerShell):
   - python -m pip install -r .\forensics_requirements.txt

2. Run on an image:
   - python -m src.forensic_pipeline .\path\to\image.jpg

3. Run on a video (first N frames):
   - python -m src.forensic_pipeline .\path\to\video.mp4 --max-frames 16

Outputs a JSON report with the above metrics. Use these as features for downstream classifiers or to aid model decisions.

Notes

- OpenCV is optional but recommended; without it, the PRNU denoiser falls back to a simple blur and Laplacian uses FFT convolution.
- All functions accept numpy arrays (HxW[xC]) in uint8 or float formats.
