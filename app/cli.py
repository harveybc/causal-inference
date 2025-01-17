import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-Inference: A tool with preprocessing, inference, and transformation plugin support.")
    
    parser.add_argument('--input_file', type=str, help='Path to the input dataset (e.g., economic calendar).')
    parser.add_argument('-pp', '--preprocessing_plugin', type=str, default='economic_preprocessor', help='Name of the preprocessing plugin to use.')
    parser.add_argument('-ip', '--inference_plugin', type=str, default='double_ml_plugin', help='Name of the inference plugin to use.')
    parser.add_argument('-tp', '--transformation_plugin', type=str, default='time_series_transformer', help='Name of the transformation plugin to use.')
    parser.add_argument('-of', '--output_file', type=str, help='Path to the output time series file.')
    parser.add_argument('-sm', '--save_model', type=str, help='Filename to save the trained inference model.')
    parser.add_argument('-lm', '--load_model', type=str, help='Filename to load a trained inference model from.')
    parser.add_argument('-te', '--test_mode', action='store_true', help='Enable test mode with predefined configurations.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output messages.')
    parser.add_argument('--remote_load_config', type=str, help='URL to load remote configuration.')
    parser.add_argument('--remote_save_config', type=str, help='URL to save configuration remotely.')
    parser.add_argument('--load_config', type=str, help='Path to load local configuration.')
    parser.add_argument('--save_config', type=str, help='Path to save local configuration.')
    parser.add_argument('--username', type=str, help='Username for remote operations.')
    parser.add_argument('--password', type=str, help='Password for remote operations.')
    parser.add_argument('--evaluate_model', action='store_true', help='Evaluate an existing model instead of running the pipeline.')

    return parser.parse_args()
