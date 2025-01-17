DEFAULT_VALUES = {
    'input_file': 'data/economic_calendar.csv',
    'output_file': 'output_time_series.csv',
    'preprocessing_plugin': 'economic_preprocessor',
    'inference_plugin': 'double_ml_plugin',
    'transformation_plugin': 'time_series_transformer',
    'save_model': 'model.pkl',
    'load_model': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'load_config': None,
    'save_config': None,
    'username': None,
    'password': None,
    'evaluate_model': False,  # Whether to evaluate an existing model
    'test_mode': False,
    'verbose': False
}

# Mapping of short-form to long-form arguments for flexibility
ARGUMENT_MAPPING = {
    'pp': 'preprocessing_plugin',
    'ip': 'inference_plugin',
    'tp': 'transformation_plugin',
    'of': 'output_file',
    'sm': 'save_model',
    'lm': 'load_model',
    'te': 'test_mode',
    'v': 'verbose',
    'rlc': 'remote_load_config',
    'rsc': 'remote_save_config',
    'lc': 'load_config',
    'sc': 'save_config',
    'un': 'username',
    'pw': 'password',
    'em': 'evaluate_model'
}
