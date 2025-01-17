from app.config_merger import merge_config, process_unknown_args
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
)
from app.data_processor import run_causal_pipeline
from app.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    # Initialize logging
    setup_logging(log_level=logging.INFO, log_file="causal_inference.log")
    logger.info("Causal-Inference application started.")

    try:
        # Parse CLI arguments
        args, unknown_args = parse_args()
        cli_args = vars(args)
        logger.debug(f"CLI arguments: {cli_args}")

        # Load default configuration
        config = DEFAULT_VALUES.copy()
        logger.debug(f"Default configuration: {config}")

        # Load remote configuration if specified
        if args.remote_load_config:
            logger.info(f"Loading remote configuration from: {args.remote_load_config}")
            remote_config = remote_load_config(
                args.remote_load_config, args.username, args.password
            )
            config.update(remote_config or {})

        # Load local configuration if specified
        if args.load_config:
            logger.info(f"Loading local configuration from: {args.load_config}")
            local_config = load_config(args.load_config)
            config.update(local_config)

        # Merge configurations with CLI and unknown args
        logger.info("Merging configurations...")
        unknown_args_dict = process_unknown_args(unknown_args)
        config = merge_config(config, cli_args, unknown_args_dict)
        logger.debug(f"Merged configuration: {config}")

        # Run the causal pipeline
        run_causal_pipeline(config)

        # Save configurations if specified
        if config.get("save_config"):
            save_config(config, config["save_config"])
            logger.info(f"Configuration saved to: {config['save_config']}")

        if config.get("remote_save_config"):
            remote_save_config(
                config,
                config["remote_save_config"],
                config.get("username"),
                config.get("password"),
            )
            logger.info(f"Configuration remotely saved to: {config['remote_save_config']}")

        logger.info("Causal-Inference application completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
