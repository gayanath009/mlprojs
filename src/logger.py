import logging
import os
from datetime import datetime

# Generate log filename with timestamp
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the full path for the log file
log_dir = os.path.join(os.getcwd(), "logs")
log_file_path = os.path.join(log_dir, log_file)

# Create logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Test logging
if __name__ == "__main__":
    logging.info("Logging has started.")
    logging.info("This is a test log message.")
    logging.info("Logging configuration is working correctly.")
    logging.info("Log file created at: %s", log_file_path)
