import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-Inference: A tool with preprocessing, inference, and transformation plugin support.")
    
    # Input and output arguments
    parser.add_argument('--input_file', type=str, help='Path to the input dataset (e.g., economic calendar).')
    parser.add_argument('-of', '--output_file', type=str, help='Path to the output time series file.')
    
    # Plugin selection arguments
    parser.add_argument('-pp', '--preprocessing_plugin', type=str, default='economic_preprocessor', help='Name of the preprocessing plugin to use.')
    parser.add_argument('-ip', '--inference_plugin', type=str, default='double_ml_plugin', help='Name of the inference plugin to use.')
    parser.add_argument('-tp', '--transformation_plugin', type=str, default='time_series_transformer', help='Name of the transformation plugin to use.')

    # Model handling arguments
    parser.add_argument('-sm', '--save_model', type=str, help='Filename to save the trained inference model.')
    parser.add_argument('-lm', '--load_model', type=str, help='Filename to load a trained inference model from.')

    # Configuration and logging arguments
    parser.add_argument('--remote_load_config', type=str, help='URL to remotely load configuration.')
    parser.add_argument('--remote_save_config', type=str, help='URL to remotely save configuration.')
    parser.add_argument('--load_config', type=str, help='Path to a local configuration file to load.')
    parser.add_argument('--save_config', type=str, help='Path to save the configuration file locally.')
    parser.add_argument('--username', type=str, help='Username for remote operations.')
    parser.add_argument('--password', type=str, help='Password for remote operations.')

    # Optional flags
    parser.add_argument('-te', '--test_mode', action='store_true', help='Enable test mode with predefined configurations.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output messages.')

    return parser.parse_args()
