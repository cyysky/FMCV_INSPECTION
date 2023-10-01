import logging
import inspect
import sys
import traceback

loglevel = 0


def _called_from():
    try:
        #https://stackoverflow.com/questions/17065086/how-to-get-the-caller-class-name-inside-a-function-of-another-class-in-python
        stack = inspect.stack()
        #for a in stack:
        #    print(a)
        #the_class = stack[2][0].f_locals.__class__.__name__
        #https://stackoverflow.com/questions/1095543/get-name-of-calling-functions-module-in-python
        the_class = inspect.getmodule(stack[2][0]).__name__ 
        the_method = stack[2][0].f_code.co_name
        #print("I was called by {}.{}()".format(the_class.__name__, the_method))
        log = logging.getLogger(f'{str(the_class)}.{str(the_method)}')
    except:
        log = logging.getLogger("")
    
    return log 

def called_from():    
    """
    Returns a logger named after the caller's module and function name.
    If there's an issue retrieving the caller information, it returns the root logger.
    """

    try:
        # Get the caller's frame information
        caller_frame = sys._getframe(2)

        # Get the caller's module and function name
        caller_module = caller_frame.f_globals['__name__']
        caller_function = caller_frame.f_code.co_name

        # Create a logger named after the caller's module and function name
        log = logging.getLogger(f'{caller_module}.{caller_function}')
    except Exception as e:
        # Log the error and return the root logger if there's an issue
        #logging.error(f"Error getting caller logger: {e}")
        log = logging.getLogger()
        
    return log

def critical(*args):
    called_from().critical(" ".join(map(str, args)))
    
def error(*args):
    called_from().error(" ".join(map(str, args)))
        
def warning(*args):
    called_from().warning(" ".join(map(str, args))) 
    
def info(*args):
    called_from().info(" ".join(map(str, args)))
       
def debug(*args):
    called_from().debug(" ".join(map(str, args)))
   
def exception(*args):
    called_from().exception(" ".join(map(str, args)))

def log_traceback():
    called_from().exception(traceback.format_exc())

def config_callback(_loglevel : int):
    global start 
    global loglevel
    start.logger.setLevel(_loglevel)

def set_log_level(_loglevel : int):
    global loglevel
    global log
    global consoleHandler
    
    loglevel = _loglevel
    
    #https://www.geeksforgeeks.org/logging-in-python/
    #NOTSET 0
    #DEBUG 10
    #INFO 20
    #WARNING 30
    #ERROR 40
    #CRITICAL 50
    
    #https://zetcode.com/python/logging/
    #logging level
    #log = logging.getLogger()
    log.setLevel(loglevel)    
    
    #StreamHandler loglevel
    #consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(loglevel)   
    

def init(in_start=None):
    global start
    global loglevel
    global log
    global consoleHandler

    if in_start is not None:
        start = in_start
        
        formatter = logging.Formatter(f'%(asctime)s %(name)s \n %(levelname)s:%(message)s')
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        
        log = logging.getLogger()
        log.addHandler(consoleHandler)
        
    set_log_level(loglevel)
    

    

    
    #start.logger.setLevel(logging.NOTSET)
    #start.sub("config/logger",config_callback)
    
    #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)