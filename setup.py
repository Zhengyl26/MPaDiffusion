from setuptools import setup, find_packages
import os
from pathlib import Path

# 读取项目描述信息
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# 多模态3D重建相关核心依赖定义
core_dependencies = [
    "blobfile>=1.0.5",  # 数据存储与读取支持
    "torch>=1.11.0",    # 核心深度学习框架
    "tqdm>=4.64.0",     # 进度条显示
    "numpy>=1.19.5",    # 多模态数据矩阵运算
    "torchvision>=0.12.0",  # 视觉模态处理
    "einops>=0.3.0",    # 多维度张量操作（适配3D结构）
    "scipy>=1.7.0",     # 科学计算（支持3D几何处理）
    "Pillow>=8.2.0",    # 图像模态数据处理
]

# 可选依赖（按需安装的功能模块）
extras = {
    "visualization": [
        "matplotlib>=3.3.1",  # 3D模型可视化
        "pyvista>=0.34.0",   # 3D网格渲染
    ],
    "data_processing": [
        "SimpleITK>=2.0.1",  # 医学影像等多模态数据预处理
        "albumentations>=0.5.1",  # 数据增强
    ],
    "logging": [
        "tensorboard>=2.5.0",  # 训练过程日志记录
    ]
}

setup(
    name="mm_property_diffusion",  # 多模态属性扩散模型标识
    version="0.1.0",
    author="The MPaDiffusion Team",
    author_email="mpadiffusion@example.com",
    description="A unified framework for multi-modal property-aware diffusion models, focusing on 3D reconstruction and on-demand design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/mm-property-diffusion",
    packages=find_packages(include=["guided_diffusion", "guided_diffusion.*"]),
    py_modules=["guided_diffusion"],
    python_requires=">=3.8",
    install_requires=core_dependencies,
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Computer Science :: 3D Graphics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="multi-modal, diffusion model, 3D reconstruction, property-aware, on-demand design",
    project_urls={
        "Documentation": "https://mpadiffusion.readthedocs.io/",
        "Source": "https://github.com/example/mm-property-diffusion",
        "Tracker": "https://github.com/example/mm-property-diffusion/issues",
    },
)