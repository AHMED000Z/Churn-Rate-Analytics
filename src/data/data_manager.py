"""
Database connection and data import utilities.
"""
import os
import logging
import pandas as pd
import psycopg2
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import DATABASE_CONFIG, DATA_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and data operations."""

    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager.

        Args:
            connection_params: Optional database connection parameters
        """
        self.connection_params = connection_params or DATABASE_CONFIG
        self.connection = None

    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Database connection established successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False

    def disconnect(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        if not self.connection:
            raise ConnectionError(
                "No database connection. Call connect() first.")

        try:
            df = pd.read_sql_query(query, self.connection)
            logger.info(
                f"Query executed successfully. Retrieved {len(df)} rows.")
            return df

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def import_view_to_csv(self, view_name: str, output_path: str) -> bool:
        """
        Import a database view to CSV file.

        Args:
            view_name: Name of the database view
            output_path: Path to save the CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            query = f"SELECT * FROM {view_name};"
            df = self.execute_query(query)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"View '{view_name}' exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting view '{view_name}': {str(e)}")
            return False

    def import_multiple_views(self, views: List[str], output_dir: str) -> Dict[str, bool]:
        """
        Import multiple database views to CSV files.

        Args:
            views: List of view names to import
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping view names to success status
        """
        results = {}

        for view in views:
            output_path = os.path.join(output_dir, f"{view}.csv")
            results[view] = self.import_view_to_csv(view, output_path)

        return results


def import_churn_data(output_dir: Optional[str] = None) -> bool:
    """
    Import churn data from database views.

    Args:
        output_dir: Optional directory to save CSV files

    Returns:
        True if all imports successful, False otherwise
    """
    if output_dir is None:
        output_dir = str(DATA_CONFIG["raw_data_path"])

    # List of views to import
    views_to_import = ['vw_churndata', 'vw_joindata']

    # Initialize database manager
    db_manager = DatabaseManager()

    try:
        # Connect to database
        if not db_manager.connect():
            logger.error("Failed to connect to database")
            return False

        # Import views
        results = db_manager.import_multiple_views(views_to_import, output_dir)

        # Check if all imports were successful
        all_successful = all(results.values())

        if all_successful:
            logger.info("All data imports completed successfully")
        else:
            failed_views = [view for view,
                            success in results.items() if not success]
            logger.error(f"Failed to import views: {failed_views}")

        return all_successful

    except Exception as e:
        logger.error(f"Error during data import: {str(e)}")
        return False

    finally:
        db_manager.disconnect()


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the data
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(
            f"Successfully loaded data from {file_path}. Shape: {df.shape}")

        # Basic data validation
        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        return df

    except Exception as e:
        logger.error(f"Error loading CSV data from {file_path}: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, file_path: str) -> bool:
    """
    Save processed data to CSV file.

    Args:
        df: DataFrame to save
        file_path: Path to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving processed data to {file_path}: {str(e)}")
        return False


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
    }

    return summary
