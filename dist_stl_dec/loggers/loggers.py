import logging
import time 

dt_string = time.strftime("%Y%m%d-%H%M%S")
filename_now  = "log_file" + "_" + dt_string + ".log"

def get_logger(name:str,filename:str |None = None, level : int = logging.INFO)-> logging.Logger:
    
    logger    =  logging.getLogger(name)
    # logger.setLevel(level)
    
    if filename is None:
        filename =filename_now
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
    
    
if __name__ == "__main__":
    logger = get_logger("test")
    print(logger.handlers)