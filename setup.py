from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal_inference",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "causal_inference=app.main:main",
        ],
        "causal_inference.preprocessing": [
            "economic_preprocessor=app.plugins.preprocessing.economic_preprocessor_plugin:Plugin",
        ],
        "causal_inference.inference": [
            "double_ml=app.plugins.inference.double_ml_plugin:Plugin",
            "causal_forest=app.plugins.inference.causal_forest_plugin:Plugin",
            "meta_learning=app.plugins.inference.meta_learning_plugin:Plugin",
        ],
        "causal_inference.transformation": [
            "time_series_transformer=app.plugins.transformation.time_series_transformer_plugin:Plugin",
        ],
    },
    install_requires=[

    ],
    extras_require={
        "dev": [

        ],
        "docs": [

        ],
    },
    author="Harvey Bastidas",
    author_email="your.email@example.com",
    description=(
        "A causal inference system that supports dynamic loading of preprocessing, "
        "inference, and transformation plugins for generating time series from causal relationships."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harveybc/causal-inference",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="causal inference, time series, machine learning, econml, plugin architecture",
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
