from setuptools import setup, find_packages

setup(
    name="face-blur-tool",
    version="0.1.0",
    description="High-recall face blurring using InsightFace and OpenCV (CPU/GPU)",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "insightface",
        "opencv-python",
        "numpy",
        "onnxruntime; platform_system != 'Windows'",  
    ],
    entry_points={
        "console_scripts": [
            "faceblur = face_blur_tool.cli:main",
        ],
    },
)
