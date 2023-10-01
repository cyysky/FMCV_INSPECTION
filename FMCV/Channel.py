from FMCV import Logging
import os
import inspect

def init(in_start):
    global start
    global topics
    start = in_start
    topics = {}
    
    #start.log.info('Channel Module loaded')

def sub(topic: str,callback):
    #https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    global start
    global topics
    
    #start.log.debug(os.path.abspath(inspect.getfile(callback)))
    start.log.debug(f'sub:{topic} callback:{inspect.getmodule(callback).__name__}.{callback.__name__}')
    
    callback_list = topics.get(topic)
    if callback_list is None:
        callback_list = []
    callback_list.append(callback)
    #start.log.debug(inspect.getfullargspec(callback))
    
    topics.update({topic: callback_list})
    
    #print(topics)

def list_topics():
    topics_str = ""
    for e in topics:
        topics_str += str(e) + "\n"
    Logging.debug(topics_str)
    return topics_str
    
def pub_(topic: str, *args, **kwargs):
    global topics
    stack = inspect.stack()
    the_class = inspect.getmodule(stack[1][0]).__name__ 
    the_method = stack[1][0].f_code.co_name
    start.log.debug(f'pub:{topic} from:{the_class}.{the_method}')
    result = {}
    callback_list = topics.get(topic)
    if callback_list is not None:
        for callback in callback_list:
            try:
                argspec = inspect.getfullargspec(callback)
                if len(argspec)>0:
                    result.update({callback.__name__:callback(*args, **kwargs)})
                else:
                    result.update({callback.__name__:callback()})
            except:
                start.log.exception(f'{inspect.getmodule(callback).__name__}.{callback.__name__}')
    return result 

def pub(topic: str, *args, **kwargs):
    # Get the current call stack
    stack = inspect.stack()
    
    # Retrieve the class and method from which this function was called
    the_class = inspect.getmodule(stack[1][0]).__name__
    the_method = stack[1][0].f_code.co_name
    
    # Log the topic and calling class/method
    start.log.debug(f'pub:{topic} from:{the_class}.{the_method}')
    
    # Initialize an empty dictionary for results
    result = {}
    
    # Get the list of callbacks associated with the given topic
    callback_list = topics.get(topic)
    
    # If there are any callbacks, process them
    if callback_list is not None:
        for callback in callback_list:
            try:
                # Check if the callback has any arguments
                argspec = inspect.getfullargspec(callback)
                if len(argspec) > 0:
                    # Call the callback with the given arguments and keyword arguments, and update the result dictionary
                    result.update({callback.__name__: callback(*args, **kwargs)})
                else:
                    # Call the callback without arguments, and update the result dictionary
                    result.update({callback.__name__: callback()})
            except:
                # Log any exceptions that occur when calling the callback
                start.log.exception(f'{inspect.getmodule(callback).__name__}.{callback.__name__}')
    
    # Return the result dictionary
    return result   
    
def get_args_spec(callback_function):
    argspec = inspect.getfullargspec(callback_function)            
    callback_arg_string = ""
    for arg in argspec.args:
       callback_arg_string += f"{arg}=kwargs['{arg}'],"
    print(callback_arg_string)
    #exec("callback({})".format(callback_arg_string)) # cannot run in cython
    return callback_arg_string