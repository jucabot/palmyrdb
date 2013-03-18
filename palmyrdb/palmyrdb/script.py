import re
import math

def _exec_func_code(function_code,*args):
    exec(function_code)
    return function(*args)
