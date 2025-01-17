#config.py
DEFAULT_VALUES = {
    'input_file': 'data/economic_calendar.csv',
    'preprocessing_plugin': 'economic_preprocessor',
    'inference_plugin': 'double_ml_plugin',
    'transformation_plugin': 'time_series_transformer',
    'output_file': 'output_time_series.csv',
    'save_model': 'model.pkl',
    'load_model': None,
    'test_mode': False,
    'verbose': False
}

# Mapping of short-form to long-form arguments
ARGUMENT_MAPPING = {
    'pp': 'preprocessing_plugin',
    'ip': 'inference_plugin',
    'tp': 'transformation_plugin',
    'of': 'output_file',
    'sm': 'save_model',
    'lm': 'load_model',
    'te': 'test_mode',
    'v': 'verbose'
}