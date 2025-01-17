import sys
import logging
from app.logger import setup_logging, get_logger  # Centralized logger
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
)
from app.config_merger import merge_config, process_unknown_args
from app.plugin_loader import load_plugin
from app.data_processor import run_causal_pipeline, evaluate_causal_model

logger = get_logger(__name__)

def main():
    """
    Main entry point for the causal inference system.

    Orchestrates the following:
    - Sets up centralized logging.
    - Parses CLI arguments.
    - Loads and merges configurations from default, local, and remote sources.
    - Dynamically loads plugins for preprocessing, inference, and transformation.
    - Executes the causal inference pipeline or evaluation.
    - Saves configurations locally or remotely as specified.
    """
    # Step 1: Initialize centralized logging
    setup_logging(
        log_level=logging.DEBUG,
        log_file="causal_inference.log",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5
    )
    logger.debug("Starting causal inference application.")

    try:
        # Step 2: Parse command-line arguments
        logger.info("Parsing command-line arguments...")
        args, unknown_args = parse_args()
        cli_args = vars(args)
        logger.debug(f"Parsed CLI arguments: {cli_args}")

        # Step 3: Load default configuration
        logger.info("Loading default configuration...")
        config = DEFAULT_VALUES.copy()
        logger.debug(f"Default configuration: {config}")

        # Step 4: Load configurations from files or remote sources
        file_config = {}

        # Load remote configuration
        if args.remote_load_config:
            logger.info(f"Loading remote configuration from: {args.remote_load_config}")
            file_config = remote_load_config(
                args.remote_load_config, args.username, args.password
            )
            logger.debug(f"Loaded remote config: {file_config}")

        # Load local configuration
        if args.load_config:
            logger.info(f"Loading local configuration from: {args.load_config}")
            local_config = load_config(args.load_config)
            file_config.update(local_config)
            logger.debug(f"Loaded local config: {local_config}")

        # Step 5: Merge configurations
        logger.info("Merging configurations...")
        unknown_args_dict = process_unknown_args(unknown_args)
        config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)
        logger.debug(f"Merged configuration: {config}")

        # Step 6: Dynamically load plugins
        logger.info("Loading and initializing plugins...")

        # Preprocessing plugin
        preprocessor_plugin_name = config.get("preprocessor_plugin")
        logger.info(f"Loading preprocessor plugin: {preprocessor_plugin_name}")
        preprocessor_class, _ = load_plugin("causal_inference.preprocessing", preprocessor_plugin_name)
        preprocessor_plugin = preprocessor_class()
        logger.debug(f"Loaded preprocessor plugin: {preprocessor_plugin_name}")

        # Inference plugin
        inference_plugin_name = config.get("inference_plugin")
        logger.info(f"Loading inference plugin: {inference_plugin_name}")
        inference_class, _ = load_plugin("causal_inference.inference", inference_plugin_name)
        inference_plugin = inference_class()
        logger.debug(f"Loaded inference plugin: {inference_plugin_name}")

        # Transformation plugin
        transformation_plugin_name = config.get("transformation_plugin")
        logger.info(f"Loading transformation plugin: {transformation_plugin_name}")
        transformation_class, _ = load_plugin("causal_inference.transformation", transformation_plugin_name)
        transformation_plugin = transformation_class()
        logger.debug(f"Loaded transformation plugin: {transformation_plugin_name}")

        # Step 7: Execute pipeline or evaluate model
        if config.get("evaluate", False):
            logger.info("Starting causal model evaluation.")
            evaluate_causal_model(config)
        else:
            logger.info("Starting causal inference pipeline.")
            run_causal_pipeline(config)

        # Step 8: Save configuration locally or remotely
        if config.get("save_config"):
            logger.info(f"Saving configuration to: {config['save_config']}")
            save_config(config, config["save_config"])
            logger.debug("Configuration saved locally.")

        if config.get("remote_save_config"):
            logger.info(f"Saving configuration remotely to: {config['remote_save_config']}")
            remote_save_config(
                config,
                config["remote_save_config"],
                config.get("username"),
                config.get("password"),
            )
            logger.debug("Configuration saved remotely.")

        logger.info("Causal inference application completed successfully.")
        return 0

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
