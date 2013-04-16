import re
import datetime


_INT_PATTERN = re.compile(r'[-0-9]*')
_FLOAT_PATTERN = re.compile(r'[-0-9]*[,|.][0-9]*')
_DATE_US_PATTERN = re.compile(r'(0[1-9]|1[012])[- \/.](0[1-9]|[12][0-9]|3[01])[- \/.](19|20)[0-9]{2}')
_DATE_EUR_PATTERN = re.compile(r'(0[1-9]|[12][0-9]|3[01])[- \/.](0[1-9]|1[012])[- \/.](19|20)[0-9]{2}')
_DATE_ISO_PATTERN = re.compile(r'(19|20)[0-9]{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])')

#supported type - add control to is_supported_type
NONE_VALUE = ''

INT_TYPE = 'int'
FLOAT_TYPE = 'float'
TEXT_TYPE = 'str'
DATE_TYPE = 'date'


"""
    String value converter based on the format and the value (regex)
    Supported format : str(default), int, float 
"""
class TypeConverter():
    
    def __init__(self):
        pass
    
    """
        Get the type the value
    """
    def get_type(self,values):
        
        values = filter(lambda v : v != NONE_VALUE,values )
        first_type = type(values[0]).__name__
        
        scaned_type = first_type
        
        if first_type == FLOAT_TYPE or first_type == INT_TYPE:
            types = map(lambda v : type(v).__name__,values)
            if TEXT_TYPE in types:
                scaned_type =  TEXT_TYPE
        
        return scaned_type
    """
        Try to type the value
    """
    def type(self,value):
        
        try:
            
            if _DATE_EUR_PATTERN.match(value) != None:
                d=None
                try:
                    d = datetime.datetime.strptime(value,"%d/%m/%Y")
                except:
                    try:
                        d = datetime.datetime.strptime(value,"%d-%m-%Y")
                    except:
                        d = datetime.datetime.strptime(value,"%d.%m.%Y")
                return datetime.date(d.year,d.month,d.day)
            elif _DATE_US_PATTERN.match(value) != None:
                d=None
                try:
                    d = datetime.datetime.strptime(value,"%m/%d/%Y")
                except:
                    try:
                        d = datetime.datetime.strptime(value,"%m-%d-%Y")
                    except:
                        d = datetime.datetime.strptime(value,"%m.%d.%Y")
                return datetime.date(d.year,d.month,d.day)
            elif _DATE_ISO_PATTERN.match(value) != None:
                d = datetime.datetime.strptime(value,"%Y-%m-%d")
                return datetime.date(d.year,d.month,d.day) 
                
            elif _FLOAT_PATTERN.match(value) != None: #Float may have . or not
                return float(value.replace(',','.'))
            elif _INT_PATTERN.match(value) != None:
                return int(value)
            
        except ValueError:
            pass #probably text
            
        return value    
    
    def is_supported_type(self,type_name):
        return type_name == INT_TYPE or type_name == FLOAT_TYPE or type_name == TEXT_TYPE or type_name == DATE_TYPE