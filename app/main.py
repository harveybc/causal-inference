# app/main.py

import sys
import json
import logging
from app.logger import setup_logging, get_logger  # Import the centralized logger
from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log,
)
from app.cli import parse_args
from app.data_processor import (
    preprocess_data,
    infer_causal_effects,
    transform_to_time_series,
)
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    """
    Main entry point for the Causal-Inference application.

    This function orchestrates the overall workflow:
    - Sets up centralized logging.
    - Parses command-line arguments.
    - Loads and merges configurations from default, local, and remote sources.
    - Dynamically loads preprocessing, inference, and transformation plugins.
    - Executes the preprocessing, causal inference, and transformation pipelines.
    - Saves configurations locally or remotely if specified.
    """
    # Initialize centralized logging
    setup_logging(
        log_level=logging.DEBUG,
        log_file="causal_inference.log",
        max_bytes=10*1024*1024,  # 10 MB
        backup_count=5
    )
    logger = get_logger(__name__)
    logger.debug("Starting main application.")

    try:
        logger.info("Parsing command-line arguments...")
        args, unknown_args = parse_args()
        cli_args = vars(args)
        logger.debug(f"Parsed CLI arguments: {cli_args}")

        logger.info("Loading default configuration...")
        config = DEFAULT_VALUES.copy()
        logger.debug(f"Default configuration: {config}")

        file_config = {}

        # Remote config file load
        if args.remote_load_config:
            logger.info(f"Loading remote configuration from: {args.remote_load_config}")
            file_config = remote_load_config(
                args.remote_load_config, args.username, args.password
            )
            logger.debug(f"Loaded remote config: {file_config}")

        # Local config file load
        if args.load_config:
            logger.info(f"Loading local configuration from: {args.load_config}")
            local_config = load_config(args.load_config)
            file_config.update(local_config)
            logger.debug(f"Loaded local config: {local_config}")

        logger.info("Merging configurations...")
        unknown_args_dict = process_unknown_args(unknown_args)
        logger.debug(f"Unknown arguments: {unknown_args_dict}")
        config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)
        logger.debug(f"Merged configuration: {config}")

        # Load and initialize preprocessing plugin
        preprocessing_plugin_name = config.get("preprocessing_plugin")
        logger.info(f"Loading preprocessing plugin: {preprocessing_plugin_name}")
        preprocessing_class, preprocessing_module = load_plugin(
            "causal_inference.preprocessing", preprocessing_plugin_name
        )
        preprocessing_plugin = preprocessing_class()
        logger.debug(f"Loaded preprocessing plugin '{preprocessing_plugin_name}' from '{preprocessing_module}'.")

        # Load and initialize inference plugin
        inference_plugin_name = config.get("inference_plugin")
        logger.info(f"Loading inference plugin: {inference_plugin_name}")
        inference_class, inference_module = load_plugin(
            "causal_inference.inference", inference_plugin_name
        )
        inference_plugin = inference_class()
        logger.debug(f"Loaded inference plugin '{inference_plugin_name}' from '{inference_module}'.")

        # Load and initialize transformation plugin
        transformation_plugin_name = config.get("transformation_plugin")
        logger.info(f"Loading transformation plugin: {transformation_plugin_name}")
        transformation_class, transformation_module = load_plugin(
            "causal_inference.transformation", transformation_plugin_name
        )
        transformation_plugin = transformation_class()
        logger.debug(f"Loaded transformation plugin '{transformation_plugin_name}' from '{transformation_module}'.")

        # Run preprocessing pipeline
        logger.info("Running preprocessing pipeline...")
        preprocessed_data = preprocess_data(config, preprocessing_plugin)
        logger.debug("Preprocessing completed.")

        # Run causal inference pipeline
        logger.info("Running causal inference pipeline...")
        inferred_effects = infer_causal_effects(config, preprocessed_data, inference_plugin)
        logger.debug("Causal inference completed.")

        # Run transformation pipeline
        logger.info("Running transformation pipeline...")
        transformed_data = transform_to_time_series(config, inferred_effects, transformation_plugin)
        logger.debug("Transformation to time series completed.")

        # Save configuration locally if specified
        if config.get("save_config"):
            logger.info(f"Saving configuration to: {config['save_config']}")
            save_config(config, config["save_config"])
            logger.debug("Configuration saved locally.")

        # Save configuration remotely if specified
        if config.get("remote_save_config"):
            logger.info(f"Saving configuration remotely to: {config['remote_save_config']}")
            remote_save_config(
                config,
                config["remote_save_config"],
                config.get("username"),
                config.get("password"),
            )
            logger.debug("Configuration saved remotely.")

        logger.info("Main application completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
