from setuptools import setup, find_packages

setup(
    name="causal_transparency_framework",
    version="0.1.0",
    description="A framework for evaluating and enhancing model transparency through causal reasoning",
    author="John Marko",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.23.0",
        "networkx>=2.5.0",
        "tqdm>=4.50.0"
    ],
    extras_require={
        "xgboost": ["xgboost>=1.2.0"],
        "notebook": ["jupyter>=1.0.0", "ipython>=7.0.0", "ipykernel>=5.3.0"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)