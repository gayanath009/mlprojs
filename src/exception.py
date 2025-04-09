import sys
import logging
from src.logger import logging # Import the logging module from the src.logger module


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = sys.exc_info() # Get the exception traceback information
    # Extract the file name and line number where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys): # Constructor to initialize the error message and detail
        super().__init__(error_message) # Initialize the base Exception class
        self.error_message = error_message_detail(error_message, error_detail) # Call the error_message_detail function to get the detailed error message

    def __str__(self):
        return self.error_message
    

# testing the custom exception
# if __name__ == "__main__":
#     try:
#         # Simulate an error for testing
#         a = 1 / 0
#     except Exception as e:
#         #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#         logging.info("An error occurred: %s", e)
#         raise CustomException(e, sys) # Raise the custom exception with the error message and detail