"""
This will auto include all the '.py' files in __init__.py exclude __init__.py itself
https://stackoverflow.com/questions/16788380/import-all-classes-or-functions-in-all-files-in-folder-as-if-they-were-all-in
"""
import os, sys

dir_path = os.path.dirname(os.path.abspath(__file__))
files_in_dir = [f[:-3] for f in os.listdir(dir_path)
                if f.endswith('.py') and f != '__init__.py']
for f in files_in_dir:
    mod = __import__('.'.join([__name__, f]), fromlist=[f])
    to_import = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]

    for i in to_import:
        try:
            setattr(sys.modules[__name__], i.__name__, i)
        except AttributeError:
            pass