import sys
import json
import logging
from app.logger import setup_logging, get_logger
from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log,
)
from app.cli import parse_args
from app.data_processor import (
    process_data,
    evaluate_causal_model,
    run_causal_pipeline,
)
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from app.config_merger import merge_config, process_unknown_args

def main():
    """
    Main entry point for the Causal-Inference application.

    This function orchestrates the overall workflow:
    - Sets up centralized logging.
    - Parses command-line arguments.
    - Loads and merges configurations from default, local, and remote sources.
    - Dynamically loads preprocessing, inference, and transformation plugins.
    - Executes the causal pipeline or evaluates a model based on the configuration.
    - Saves configurations locally or remotely if specified.
    """
    # Initialize centralized logging
    setup_logging(
        log_level=logging.DEBUG,
        log_file="causal_inference.log",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5
    )
    logger = get_logger(__name__)
    logger.debug("Starting main application.")

    try:
        logger.info("Parsing command-line arguments...")
        args = parse_args()
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
        unknown_args_dict = process_unknown_args(cli_args)
        logger.debug(f"Unknown arguments: {unknown_args_dict}")
        config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)
        logger.debug(f"Merged configuration: {config}")

        # Determine whether to evaluate a model or run the causal pipeline
        if config.get("evaluate_model"):
            logger.info("Evaluating the causal model...")
            evaluate_causal_model(config)
        else:
            logger.info("Running the causal inference pipeline...")
            run_causal_pipeline(config)

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
