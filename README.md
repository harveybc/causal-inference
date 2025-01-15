# Causal-Inference

## Description

Causal-Inference is a modular and extensible tool designed to transform datasets, such as economic calendars, into time series that reflect the causal influence of events on target variables like trend and volatility. It supports dynamic plugins for preprocessing, causal inference, and transformation, making it adaptable to a wide range of datasets and causal estimation methods. By leveraging advanced methods like Double Machine Learning, Causal Forests, and Meta-Learning, the tool provides robust and customizable pipelines for causal transformation tasks.

### Key Features:
- **Dynamic Plugins:** Easily integrate different preprocessing methods, causal inference techniques, and transformation approaches.
- **Flexible Causal Methods:** Includes support for Double Machine Learning, Causal Forests, and Meta-Learning, with the ability to add new methods as plugins.
- **Time Series Transformation:** Converts inferred causal relationships into hourly time series data for applications like financial forecasting or trading.
- **Extensibility:** Add custom plugins to experiment with different configurations and estimation techniques.
- **Seamless Integration:** Can be used as a standalone tool or integrated into larger machine learning workflows.

This tool is ideal for data scientists and machine learning engineers interested in causal analysis and its applications in financial forecasting, automated trading, and other domains.

## Installation Instructions

To install and set up the Causal-Inference application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/causal-inference.git
    cd causal-inference
    ```

2. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name causal-inference-env python=3.9
        conda activate causal-inference-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the Application**:
    - On Windows, run:
        ```bash
        causal-inference.bat
        ```

    - On Linux, run:
        ```bash
        sh causal-inference.sh
        ```

7. **(Optional) Run Tests**:
    - On Windows, run:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```

## Usage

The application supports several command-line arguments to control its behavior:
```bash
usage: causal_inference.bat --help
```

### Command Line Arguments

#### Required Arguments

- `input_file` (str): Path to the input dataset (e.g., economic calendar).

#### Optional Arguments

- `-pp, --preprocessing_plugin` (str, default='economic_preprocessor'): Name of the preprocessing plugin to use.
- `-ip, --inference_plugin` (str, default='double_ml_plugin'): Name of the inference plugin to use.
- `-tp, --transformation_plugin` (str, default='time_series_transformer'): Name of the transformation plugin to use.
- `-of, --output_file` (str): Path to the output time series file.
- `-sm, --save_model` (str): Filename to save the trained inference model.
- `-lm, --load_model` (str): Filename to load a trained inference model from.
- `-te, --test_mode` (flag): Enable test mode with predefined configurations.
- `-v, --verbose` (flag): Enable verbose output messages.

### Examples of Use

#### Basic Transformation Example

To preprocess an economic calendar dataset, estimate causal effects using Double Machine Learning, and transform the results into time series data:

```bash
causal-inference.bat -pp economic_preprocessor -ip double_ml_plugin -tp time_series_transformer -of output_time_series.csv
```

## Project Directory Structure
```bash
causal-inference/
│
├── app/                           # Main application package
│   ├── cli.py                    # Handles command-line argument parsing
│   ├── config.py                 # Stores default configuration values
│   ├── config_handler.py         # Manages configuration loading, saving, and merging
│   ├── config_merger.py          # Merges configuration from various sources
│   ├── data_handler.py           # Handles data loading and saving
│   ├── data_processor.py         # Processes input data and runs the causal inference pipeline
│   ├── main.py                   # Main entry point for the application
│   ├── plugin_loader.py          # Dynamically loads preprocessing, inference, and transformation plugins
│   └── plugins/                  # Plugin directory
│       ├── preprocessing/        # Preprocessing plugins
│       │   ├── base_preprocessor_plugin.py
│       │   ├── economic_preprocessor_plugin.py
│       │   └── ...
│       ├── inference/            # Inference plugins
│       │   ├── base_inference_plugin.py
│       │   ├── double_ml_plugin.py
│       │   ├── causal_forest_plugin.py
│       │   ├── meta_learning_plugin.py
│       │   └── ...
│       ├── transformation/       # Transformation plugins
│           ├── base_transformer_plugin.py
│           ├── time_series_transformer_plugin.py
│           └── ...
│
├── tests/                         # Test modules for the application
│   ├── test_plugins.py
│   ├── test_transformations.py
│   └── ...
│
├── examples/                      # Examples of use
│   ├── run_double_ml.py
│   ├── run_meta_learning.py
│   └── ...
│
├── causal_inference.py            # Main entry point for the application
├── README.md                      # Overview and documentation for the project
├── requirements.txt               # Lists Python package dependencies
├── setup.py                       # Script for packaging and installing the project
├── set_env.bat                    # Batch script for environment setup
├── set_env.sh                     # Shell script for environment setup
└── .gitignore                     # Specifies intentionally untracked files to ignore
```

## Contributing

Contributions to the project are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

