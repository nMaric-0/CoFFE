from setuptools import setup, find_packages

setup(
    name="mft-cpea",
    version="0.1.0",
    author="Nikola",
    author_email="nikola@example.com",
    description="Multimodal Fusion Transformer with Class-Aware Patch Embedding Adaptation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mft-cpea",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "omegaconf>=2.1.0",
        "einops>=0.4.0",
        "timm>=0.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
