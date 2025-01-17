import pandas as pd
from app.logger import get_logger

logger = get_logger(__name__)

def load_csv(file_path: str, headers: bool = False) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame with optional header processing.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.
    headers : bool, optional
        Indicates whether the CSV file contains headers (default is False).

    Returns
    -------
    pd.DataFrame
        The loaded and processed DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.ParserError
        If the file cannot be parsed as a CSV.
    """
    logger.debug(f"Loading CSV from: {file_path}, with headers: {headers}")
    try:
        if headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
            logger.debug(f"Loaded CSV with headers. Shape: {data.shape}")
        else:
            data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
            logger.debug("Loaded CSV without headers.")
            
            # Assign generic column names if no headers are provided
            data.columns = ['date'] + [f'feature_{i}' for i in range(1, len(data.columns))]
            logger.debug(f"Assigned column names: {data.columns}")
            data.set_index('date', inplace=True)
            logger.debug("Date column set as index.")

        # Ensure all non-date columns are numeric
        non_date_columns = data.columns if 'date' not in data.columns else data.columns[1:]
        data[non_date_columns] = data[non_date_columns].apply(pd.to_numeric, errors='coerce')
        logger.debug(f"Converted non-date columns to numeric types.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        raise

    logger.info(f"Loaded CSV successfully. Shape: {data.shape}")
    return data

def write_csv(file_path: str, data: pd.DataFrame, include_date: bool = True, headers: bool = True) -> None:
    """
    Writes a pandas DataFrame to a CSV file.

    Parameters
    ----------
    file_path : str
        Path to save the CSV file.
    data : pd.DataFrame
        The DataFrame to save.
    include_date : bool, optional
        Whether to include the date index in the output file (default is True).
    headers : bool, optional
        Whether to include column headers in the output file (default is True).
    """
    logger.debug(f"Saving DataFrame to: {file_path}, include_date: {include_date}, headers: {headers}")
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)

        logger.info(f"DataFrame saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")
        raise
