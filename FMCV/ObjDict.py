from functools import reduce
import json
import configparser

#https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object
#https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
class ObjDict(dict):
    '''
        Object Dict Example dict.a.b
    '''
    def __init__(self, dict_):
        self.__dict__.update(dict_)
        
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)
        
    def __str__(self):
        default = lambda o: self.to_string(o)
        return json.dumps(self,default=default, indent = 4)
    
    def to_string(self, obj):
        output = ""
        try:
            output = json.loads(str(obj))
        except:
            output = f"<<non-serializable: {type(obj).__qualname__}>>"
        return output
    
    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()
    
    #https://csrgxtu.github.io/2020/04/21/Python-Dict-a-new-implementation-by-pure-Python/
    def get(self, key):
        """
        get value by key
        :param key: str
        :return: value
        """
        try:
            return self.__dict__.__getitem__(key)
        except KeyError:
            return None

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

def obj(d={}):
    '''
    dict2obj does nested Dict to Object Dict Example dict.a.b
    :param in_configparser: configparser 
    :return: Object Dict Example dict.a.b
    '''
    return dict2obj(d)

def new_item(name):
    return obj({str(name):obj({})})

def dict2obj(d={}):
    '''
    dict2obj does nested Dict to Object Dict Example dict.a.b
    :param in_configparser: configparser 
    :return: Object Dict Example dict.a.b
    '''
    return json.loads(json.dumps(d), object_hook=ObjDict)

def config2obj(in_configparser):
    '''
    config2obj does configparser type to Dict
    :param in_configparser: configparser 
    :return: Object Dict Example dict.a.b
    '''
    return dict2obj(config2dict(in_configparser))

def config2dict(in_configparser):
    '''
    config2dict does configparser type to Dict
    :param in_configparser: configparser 
    :return: Dict
    '''
    dict_config = {}
    for section in in_configparser.sections(): 
        dict_config.update({section:{}})
        
    for section in in_configparser.sections(): 
        dict_config[section].update(in_configparser[section])
        
    return dict_config

def obj2config(in_obj): 
    temp_config = configparser.ConfigParser()
    
    #Convert Object Dict to ConfigParser
    for keys, values in in_obj.items():  
        temp_config.add_section(keys)
        for options, option_values in in_obj[keys].items():
            if type(option_values) is list:
                temp_str = ""
                for e in option_values:
                    if temp_str == "":
                        temp_str = e
                    else:
                        temp_str = temp_str+","+e
                temp_config[keys][options] = temp_str
            else:
                temp_config[keys][options] = str(in_obj[keys][options])
                
    return temp_config

def merge(a, b, path=None):
    '''
    merge does combine contain of Dict a and b
    #https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    
    :param a: first Dict 
    :param b: second Dict
    :param path: 
    :return: combined Dict
    '''
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
    
def merges(*args): # Need debug
    '''
    merges does combine contain of Dict a ... n
    #https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    
    :param a: first Dict 
    :param ... n: n Dict
    :return: combined Dict
    '''
    return reduce(merge,list(args)) #https://stackoverflow.com/questions/15489091/python-converting-args-to-list
    
# unused code
# class obj2(dict):
    # def __init__(self, *arg,**kw):
        # super(obj2, self).__init__(*arg, **kw)
        
# def dict2obj2(d):
    # return json.loads(json.dumps(d), object_hook=obj)