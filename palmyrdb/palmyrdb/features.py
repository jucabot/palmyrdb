from sklearn.preprocessing import scale
from numpy.ma.core import mean, std
from numpy.lib.function_base import median, percentile
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.svm import SVC,SVR
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from palmyrdb.converter import FLOAT_TYPE, INT_TYPE, TEXT_TYPE,NONE_VALUE

from palmyrdb.script import _exec_func_code

def _freqdist(values):
        freq = {}
        values_len = float(len(values))
        for value in values:
            if isinstance(value,float):
                value = round(value)
            try:
                freq[value] +=1
            except KeyError:
                freq[value] = 1
        for value, count in freq.items():
            freq[value] = round(freq[value]/values_len * 100,4)
        
        return freq
    
def _check_is_sentence(documents):
    
    for doc in documents:
        if ' ' in doc:
            return True
    return False

def score_tfidf_freq(tfidfs):
    
    return float(mean(tfidfs))
    #return sum(map(lambda tfidf : 1 if tfidf >=0.05 else 0, tfidfs)) / float(len(tfidfs))
 
def _word_tfidf_dist(documents):
    words_tfidf = {}
    
    if len(documents) > 0: 
        if _check_is_sentence(documents): #if document contains only 1 or 2 or 3 chars --> acronyms
            try:
                v = TfidfVectorizer(ngram_range=(1,2),max_features=50)
                matrix = v.fit_transform(documents).todense()
                
                for vocabulary in v.vocabulary_.items():
                    word = vocabulary[0]
                    indice = vocabulary[1]
                    words_tfidf[word] = score_tfidf_freq(matrix[:,indice]) 
            except ValueError:
                return {}
        else:
            return _freqdist(documents)
        
    
    return words_tfidf

v = TfidfVectorizer(ngram_range=(1,2))
def _decode_document(document):
    
    return v.build_analyzer()(document)
    

class Feature():
    table = None
    name = None
    _type_name = None
    _values = None                 #heavy
    _virtual = None
    seq_order = 0
    _is_class = None
    is_sparse = None
    _usable = None
    classes = None                #heavy?
    common_value = None
    min_value = None
    max_value = None
    mean_value = None
    median_value =None        
    freq_dist = None              #heavy if no class
    num_unique_values = None
    
    default_function_code = None
    virtual_function_code = None
    first_quarter_percentile = None
    third_quarter_percentile = None
    
    """
        Create a feature
    """
    @staticmethod
    def create_feature(table,name,type_name,values,virtual=True):
        return Feature(table,name,type_name,values,virtual)
    
    def __init__(self,table,name,type_name,values,virtual=True):
        self.table = table
        self.name = name
        self._type_name = type_name
        self._set_values(values)
        self._virtual = virtual
        self._usable = True
        self.seq_order = 0
        self._is_class = None
        self.is_sparse = None
        self.classes = None
        self.common_value = None
        self.min_value = None
        self.max_value = None
        self.mean_value = None
        self.median_value =None
        self.std_dev = None
        self.first_quarter_percentile = None
        self.third_quarter_percentile = None
        self.freq_dist = None
        self.num_unique_values =None
        self.default_function_code = "function = lambda table,feature,row_index : feature['common-value']"
        self.virtual_function_code = None
        
     
    """
        Is discarded feature?
    """   
    def is_usable(self):
        return self._usable
    
    """
        Is virtual feature?
    """ 
    def is_virtual(self):
        return self._virtual
    
           
    """
       Get feature type - type is set at feature creation only
        **PUBLIC**
    """  
    def get_type(self):
        return self._type_name
    
    
    """
        Define if the feature is a class Feature
        **PUBLIC**
    """  
    def set_class(self,is_class):
        self._is_class = is_class
    
    """
        Is a classed feature?
       
        **PUBLIC**
    """            
    def has_class(self):
        return self._is_class

    
    # reshape the header for display
    def _reshape_header(self):
        headers = []
        header = self.name
        
        if self.is_virtual():
            header += '*'
            
        if self.has_class():
            if len(self.classes)>2:
                for feature in self.classes:
                    headers.append(header+"=" + str(feature))
            else:
                headers.append(header)
        else:       
            headers.append(header)
        
        return headers

    """
    *************************************************************************************
    """
    """
        Get the defined values and force typing
        Long running task
    """  
    def _get_defined_values(self,row_ids=None):
        if self.get_type() == INT_TYPE or self.get_type() == FLOAT_TYPE:
            return map(lambda x : float(x),filter (lambda a: a != NONE_VALUE, self._get_values(row_ids)))
        elif self.get_type() == TEXT_TYPE:
            return map(lambda x : str(x),filter (lambda a: a != NONE_VALUE, self._get_values(row_ids)))
        else:
            return filter (lambda a: a != NONE_VALUE, self._get_values(row_ids))
    """
        Get the undefined values
        Long running task
    """  
    def _get_undefined_values(self,row_ids=None):
        return filter (lambda a: a == NONE_VALUE, self._get_values(row_ids))
   
    """
        Auto discover the feature properties while creating the analysis object
        Long running task
    """  
    def _discover(self):
        
        #compute stats
        self._refresh()
        
        #classes are fixed by discover
        self.set_class((1.0 - (float(self.num_unique_values) / float(self.num_values))) > 0.99)
        self.classes = self.freq_dist.keys() #classes are fixed by discover
        self.default_function_code = "function = lambda table,feature,row_index : feature.common_value"    
    
    def _filter(self,filter_function=None):
        if filter_function is None:
            values = self._get_defined_values()
        else:
            ids = self.table._get_row_ids(filter_function)
            values = self._get_defined_values(row_ids=ids)
        
        return values
    """
        **PUBLIC**
    """
    def get_frequency_distribution(self,filter_function=None):
        
        values = self._filter(filter_function)

        if self.get_type() == TEXT_TYPE:
            return _word_tfidf_dist(values)
        else:
            return _freqdist(values)
        
        
    
    def _compute_stats(self):
        if self.get_type() == INT_TYPE or self.get_type() == FLOAT_TYPE:
            distinct_values = self.freq_dist.keys()
            self.min_value = min(distinct_values)
            self.max_value = max(distinct_values)
            values = self._get_defined_values()
            self.mean_value = mean(values)
            self.median_value = median(values)
            self.first_quarter_percentile = percentile(values,25)
            self.third_quarter_percentile = percentile(values,75)
            self.std_dev = std(values)
    
    """
        Refresh the feature properties
        Long running task (value metrics)
    """  
    def _refresh(self):
        
        self.num_values = len(self._get_values())
        self.num_undefined_value = len(self._get_undefined_values())
        
        #compute the density distribution (count or tfidf)
        self.freq_dist = self.get_frequency_distribution()
        
        #is completely dense
        self.is_sparse = self.num_undefined_value > 0
        
        self.num_unique_values = len(self.freq_dist.keys())
        
        #what is the most used value
        sorted_fd = sorted(self.freq_dist.iteritems(), key=lambda (k,v): v*-1)
        self.common_value, common_rank = sorted_fd[0]
        
        #compute feature statistics
        self._compute_stats()
    
    """
        Get the feature values - must be externalized to support large values
    """
            
    def _get_values(self,row_ids=None):
        
        if row_ids is not None:
            rows = map(lambda row_id : self._get_value(row_id),row_ids)
            return rows
        else:
            return self._values
    
    
    """
        Get a feature-row value - must be externalized to support large values
    """
    def _get_value(self,row_index):
        if self.get_type() == TEXT_TYPE:
            return str(self._values[row_index])
        else:
            return self._values[row_index]
    
    """
        Set the feature values - must be externalized to support large values
    """
    def _set_values(self,values):
        self._values = values
    
    # reshape the target value fot the dataset
    def _reshape_target(self,row_index):
        value = self._get_value(row_index)
        
        #if undefined value, then compute the default value function
        if value == NONE_VALUE:        
            value = _exec_func_code(self.default_function_code,self.table,self,row_index)
        
        #Convert TEXT target value as Int value for model compliance            
        if self.get_type() == TEXT_TYPE:
            assert self.has_class(),"Unclassed Text Feature cannot be a target"
            value = self.classes.index(value)
        return value
    
    # reshape the non target value fot the dataset
    def _reshape_value(self,row_index):
        reshaped_values = []
        value = self._get_value(row_index)
        
        if value == NONE_VALUE:        
            value = _exec_func_code(self.default_function_code,self.table,self,row_index)
        
                   
        if self.has_class():
            if len(self.classes)>2: 
                for feature in self.classes:
                    if value == feature:
                        reshaped_values.append(1.0)
                    else:
                        reshaped_values.append(0.0)
            else: #Binary class
                if self.get_type() == TEXT_TYPE:
                    if value == self.common_value:
                        reshaped_values.append(1.0)
                    else:
                        reshaped_values.append(0.0)
                else:
                    reshaped_values.append(float(value))
        else: # no class
            if self.get_type() == TEXT_TYPE: #text
                for feature in self.classes:
                    if feature in _decode_document(unicode(value)) :
                        reshaped_values.append(1.0)
                    else:
                        reshaped_values.append(0.0)
            else: #numeric
                reshaped_values.append(float(value))
        
        return reshaped_values
    
    
    """
        Update virtual feature and compute the feature values (long task)
        **PUBLIC**
    """
    def update_feature(self):
        if self.is_virtual():
            values = self.table._load_virtual_values(self.virtual_function_code)
            self._set_values(values)
            self._discover()
    """
        **PUBLIC**
    """
    def get_distribution_by(self, feature,centile=False):
        freq_dist = []
        
        for category in feature.classes:
                filter_function = lambda table,row_index : table.get_feature(feature.name)._is_value_equal(row_index,category) 
                group_freq = self.get_frequency_distribution(filter_function)
                
                serie = {
                         'name' : feature.name + "=" + unicode(int(category) if feature.get_type()==INT_TYPE else category),
                         'data' : map(lambda category : round(group_freq[category]*100 if centile else group_freq[category] ,4) if category in group_freq else 0 ,self.classes)
                         }
                freq_dist.append(serie)
        return freq_dist
    """
        **PUBLIC**
    """
    def get_distribution_stats_by(self,feature,centile=False):
        result = []
        for category in feature.classes:
            filter_function = lambda table,row_index : table.get_feature(feature.name)._is_value_equal(row_index,category) 
            group_values = self._filter(filter_function)
            
            if (len(group_values) >1):
                stat_min = min(group_values)
                stat_max = max(group_values)
                stat_median = median(group_values)
                stat_first_quartile = percentile(group_values,25)
                stat_third_quartile = percentile(group_values,75)
                
                result.append([stat_min,stat_first_quartile,stat_median, stat_third_quartile,stat_max])
        return result
    
    def _is_value_equal(self,row_index, value_to_compare):

        if self.get_type() == TEXT_TYPE: #case insensitive comparaison
            return value_to_compare.lower() in self._get_value(row_index).lower()
        elif self.get_type() == FLOAT_TYPE:
            return round(float(self._get_value(row_index))) == round(float(value_to_compare))
        else:
            return self._get_value(row_index) == value_to_compare
    """
        **PUBLIC**
    """    
    def get_correlation_with(self,feature,filter_function = None):
        row_ids = self.table._get_row_ids(filter_function)
        return map(lambda i : [self._get_value(i),feature._get_value(i)],row_ids)
        
