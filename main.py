"""
Main entry point for the Customer Churn Prediction System.
"""
from config.settings import LOGGING_CONFIG
from src.data.data_manager import import_churn_data
from src.api.main import run_server
import click
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Customer Churn Prediction System CLI."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host address to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host, port, reload):
    """Start the FastAPI server."""
    import uvicorn

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option('--output-dir', help='Output directory for CSV files')
def import_data(output_dir):
    """Import data from database to CSV files."""
    logger.info("Starting data import...")

    success = import_churn_data(output_dir)

    if success:
        logger.info("Data import completed successfully")
        click.echo("✅ Data import completed successfully")
    else:
        logger.error("Data import failed")
        click.echo("❌ Data import failed")
        sys.exit(1)


@cli.command()
def streamlit():
    """Start the Streamlit web application."""
    import subprocess

    logger.info("Starting Streamlit application...")

    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'src/web/streamlit_app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0'
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit: {e}")
        click.echo("❌ Failed to start Streamlit application")
        sys.exit(1)


@cli.command()
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--target-column', default='customer_status', help='Target column name')
@click.option('--test-size', default=0.2, help='Test size ratio')
def train(data_path, target_column, test_size):
    """Train the churn prediction model."""
    from src.models.predictor import ChurnPredictor
    from src.utils.preprocessing import load_and_preprocess_data

    logger.info(f"Starting model training with data from {data_path}")

    try:
        # Load and preprocess data
        click.echo("📊 Loading and preprocessing data...")
        df = load_and_preprocess_data(data_path)

        # Initialize predictor
        predictor = ChurnPredictor()

        # Train model
        click.echo("🤖 Training model...")
        metrics = predictor.train_model(df, target_column, test_size)

        # Save model
        click.echo("💾 Saving model...")
        if predictor.save_model():
            logger.info("Model training completed successfully")
            click.echo("✅ Model training completed successfully")

            # Display metrics
            click.echo("\n📈 Training Metrics:")
            for metric, value in metrics.items():
                click.echo(f"  {metric}: {value:.4f}")
        else:
            raise Exception("Failed to save model")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        click.echo(f"❌ Model training failed: {e}")
        sys.exit(1)


@cli.command()
def test():
    """Run the test suite."""
    import subprocess

    logger.info("Running test suite...")

    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/',
            '-v',
            '--cov=src',
            '--cov-report=term-missing'
        ], check=True)

        click.echo("✅ All tests passed")

    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")
        click.echo("❌ Some tests failed")
        sys.exit(1)


@cli.command()
def health():
    """Check system health and dependencies."""
    import pkg_resources
    import psycopg2
    from src.models.predictor import ChurnPredictor

    click.echo("🔍 Checking system health...")

    # Check Python version
    python_version = sys.version_info
    click.echo(
        f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check key dependencies
    key_packages = ['fastapi', 'scikit-learn', 'pandas', 'numpy', 'streamlit']

    click.echo("\n📦 Checking dependencies:")
    for package in key_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            click.echo(f"  ✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            click.echo(f"  ❌ {package}: Not installed")

    # Check model
    click.echo("\n🤖 Checking model:")
    predictor = ChurnPredictor()
    if predictor.load_model():
        model_info = predictor.get_model_info()
        click.echo(
            f"  ✅ Model loaded: {model_info.get('model_type', 'Unknown')}")
    else:
        click.echo("  ❌ Model not found or failed to load")

    # Check database connection (optional)
    click.echo("\n🗄️  Checking database connection:")
    try:
        from config.settings import DATABASE_CONFIG
        conn = psycopg2.connect(**DATABASE_CONFIG)
        conn.close()
        click.echo("  ✅ Database connection successful")
    except Exception as e:
        click.echo(f"  ⚠️  Database connection failed: {e}")
        click.echo("     (This is optional if using CSV files)")

    click.echo("\n✅ Health check completed")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-file', help='Output file for predictions')
def predict_batch(input_file, output_file):
    """Make batch predictions from CSV file."""
    import pandas as pd
    from src.models.predictor import ChurnPredictor

    logger.info(f"Starting batch prediction for {input_file}")

    try:
        # Load data
        click.echo("📊 Loading input data...")
        df = pd.read_csv(input_file)

        # Initialize predictor
        predictor = ChurnPredictor()
        if not predictor.load_model():
            raise Exception("Failed to load model")

        # Make predictions
        click.echo("🔮 Making predictions...")
        predictions = []

        for index, row in df.iterrows():
            try:
                result = predictor.predict(row.to_dict())
                predictions.append({
                    'row_index': index,
                    'prediction': result['prediction'],
                    'prediction_label': result['prediction_label'],
                    'probability': result['probability'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                logger.warning(f"Failed to predict for row {index}: {e}")
                predictions.append({
                    'row_index': index,
                    'prediction': None,
                    'prediction_label': 'Error',
                    'probability': None,
                    'confidence': None,
                    'error': str(e)
                })

        # Save results
        results_df = pd.DataFrame(predictions)

        if output_file:
            results_df.to_csv(output_file, index=False)
            click.echo(f"✅ Predictions saved to {output_file}")
        else:
            click.echo("\n📊 Prediction Results:")
            click.echo(results_df.to_string(index=False))

        # Summary statistics
        total_predictions = len(predictions)
        churn_predictions = len(
            [p for p in predictions if p['prediction'] == 1])
        error_count = len([p for p in predictions if p['prediction'] is None])

        click.echo(f"\n📈 Summary:")
        click.echo(f"  Total rows: {total_predictions}")
        click.echo(f"  Predicted churn: {churn_predictions}")
        click.echo(f"  Errors: {error_count}")

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        click.echo(f"❌ Batch prediction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
