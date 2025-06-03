import logging
import sys
import os

file_dir="logger"

os.makedirs(file_dir,exist_ok=True)

file_name="logging.log"

filepath=os.path.join(file_dir,file_name)

logging.basicConfig(
    level=logging.INFO,

    format=" %(asctime)s - %(name)s - %(levelname)s -%(message)s ",

    handlers=[
        logging.FileHandler(filepath)
         ]
    
)

logger=logging.getLogger("Ml_Algos")

