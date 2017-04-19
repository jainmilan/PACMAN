__title__ = 'pacman'
__version__ = '0.0.1'
__author__ = 'Milan Jain'
__copyright__ = 'Copyright 2014 Milan Jain'

# init.py module initializes all the modules of pacman library.

# System Libraries
import os

# Set default logging handler to avoid "No handler found" warnings.
import logging
import logging.config

# Custom modules
import model
import learn
import models
import predict
import estimate
import plot_data
import data, stats

# Python 2.7+
try:  
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

# Log file
log_file = "logs/pacman.log"

# Check if directory exists
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
	if not os.path.exists(log_file):
		open(log_file, 'a').close()

# Create handler to log the data
logging.getLogger(__name__).addHandler(NullHandler())
logging.basicConfig(filename=log_file, format='%(asctime)s %(levelname)s %(module)s %(message)s', \
	datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
