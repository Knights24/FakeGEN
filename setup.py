"""Setup script for Real vs AI Image Detector package."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements_lightweight.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="real-vs-ai-detector",
    version="1.0.0",
    author="Knights24",
    author_email="",
    description="Lightweight deep learning system for detecting AI-generated images vs real camera photos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Knights24/FakeGEN",
    project_urls={
        "Bug Tracker": "https://github.com/Knights24/FakeGEN/issues",
        "Documentation": "https://github.com/Knights24/FakeGEN/blob/main/docs/USER_GUIDE.md",
        "Source Code": "https://github.com/Knights24/FakeGEN",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "real-ai-train=scripts.train:main",
            "real-ai-evaluate=scripts.evaluate:main",
            "real-ai-predict=scripts.predict:main",
            "real-ai-verify=scripts.verify_setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "ai-detection",
        "deepfake-detection",
        "image-classification",
        "pytorch",
        "computer-vision",
        "efficientnet",
        "real-vs-fake",
        "synthetic-images",
    ],
)
