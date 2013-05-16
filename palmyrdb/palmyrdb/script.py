import re
import math
import datetime

def _exec_func_code(function_code,*args):
    exec(function_code)
    return function(*args)

def compile_func_code(function_code):
    exec function_code in globals()
    return function

