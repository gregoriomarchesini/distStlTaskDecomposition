import logging
import time 

def get_logger(name:str,filename:str |None = None, level=logging.INFO)-> logging.Logger:
    
    logger    =  logging.getLogger(name)
    
    if filename is None:
        dt_string = time.strftime("%Y-%m-%d")
        filename  = "log_file" + "_" + dt_string + ".log"
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("logfile.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
    