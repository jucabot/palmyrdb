from palmyrdb.converter import FLOAT_TYPE, INT_TYPE, TEXT_TYPE,NONE_VALUE,\
    DATE_TYPE
from palmyrdb.script import _exec_func_code, compile_func_code
from numpy.ma.core import mean, std
from numpy.lib.function_base import median, percentile
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import uuid
import datetime


def format_int(value):
    if value == NONE_VALUE:
        return NONE_VALUE
    else:
        return int(value)
    
def format_float(value):
    if value == NONE_VALUE:
        return NONE_VALUE
    else:
        return float(value)

def format_date(value):
    """
    if value == NONE_VALUE:
        return NONE_VALUE
    else:
        return datetime.date.strftime(value,"%Y-%m-%d")
    """
    return value

def compare_date(value, value_to_compare):
    
    if isinstance(value, str):
        value = datetime.datetime.strptime(value,"%Y-%m-%d").date()
    if isinstance(value_to_compare, str):
        value_to_compare = datetime.datetime.strptime(value_to_compare,"%Y-%m-%d").date()
    
    return value == value_to_compare 

def compare_text(value, value_to_compare):
    return value_to_compare.lower() in value.lower()

def compare_float(value, value_to_compare):
    return round(float(value)) == round(float(value_to_compare))

def compare(value, value_to_compare):
    return value == value_to_compare 

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
        #remove undefined values
        if NONE_VALUE in freq:
            values_len -= freq[NONE_VALUE]
            del freq[NONE_VALUE]
        
        for value, count in freq.items():
            freq[value] = round(freq[value]/values_len * 100,2)
        
        return freq
    
def _date_freqdist(values):
        freq = {}
        values_len = float(len(values))
        for value in values:
            
            try:
                freq[datetime.date.strftime(value,"%Y-%m-%d")] +=1
            except KeyError:
                freq[datetime.date.strftime(value,"%Y-%m-%d")] = 1
        #remove undefined values
        if NONE_VALUE in freq:
            values_len -= freq[NONE_VALUE]
            del freq[NONE_VALUE]
        
        for value, count in freq.items():
            freq[value] = round(freq[value]/values_len * 100,2)
        
        return freq    


def _compute_stats_function(values):
    stats = None
    if len(values)>1:
        stats = {}
        stats['min'] = min(values)
        stats['max'] = max(values)
        stats['mean'] = mean(values)
        stats['median'] = median(values)
        stats['1st-quartile'] = percentile(values,25)
        stats['3rd-quartile'] = percentile(values,75)
        stats['std-error'] = std(values)
        
    return stats

"""
    pseudo aggregation of plot pairs - should limit the mumber of point in scatter plot while aggregating clusters of points
"""
def scatter_cluster_function(xy_values):
    x_values = xy_values[0]
    y_values = xy_values[1]
    
    return map(lambda i : [x_values[i],y_values[i]],range(len(xy_values[0])))
def _check_is_sentence(documents):
    
    for doc in documents:
        if ' ' in doc:
            return True
    return False

def score_tfidf_freq(tfidfs):
    
    return round(float(mean(tfidfs)*100),2)
    #return sum(map(lambda tfidf : 1 if tfidf >=0.05 else 0, tfidfs)) / float(len(tfidfs))
 
def _word_tfidf_dist(documents):
    words_tfidf = {}
    
    if len(documents) > 0: 
        if _check_is_sentence(documents): #if document contains only 1 or 2 or 3 chars --> acronyms
            try:
                text_analyzer = TfidfVectorizer(ngram_range=(1,2),max_features=50)
                matrix = text_analyzer.fit_transform(documents).todense()
                
                for vocabulary in text_analyzer.vocabulary_.items():
                    word = vocabulary[0]
                    indice = vocabulary[1]
                    words_tfidf[word] = score_tfidf_freq(matrix[:,indice]) 
            except ValueError:
                return {}
        else:
            return _freqdist(documents)
    return words_tfidf




class Feature():
    id = None
    table = None
    name = None
    _type_name = None
    _virtual = None
    seq_order = 0
    _is_class = None
    is_sparse = None
    classes = None                #heavy?
    common_value = None
    min_value = None
    max_value = None
    mean_value = None
    median_value =None        
    freq_dist = None              #heavy if no class
    num_unique_values = None
    format_function = None
    compare_function = None
    default_function_code = None
    virtual_function_code = None
    first_quarter_percentile = None
    third_quarter_percentile = None
    
    text_analyzer = None
    
    
    """
        Create a feature
    """
    @staticmethod
    def create_feature(table,name,type_name,virtual=True):
        f = Feature(table,name,type_name,virtual)
        f.id = str(uuid.uuid4())
        return f
    
    def __init__(self,table,name,type_name,virtual=True):
        self.id=None
        self.table = table
        self.name = name
        self._type_name = type_name
        self._virtual = virtual
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
        
        if type_name == TEXT_TYPE:
            self.text_analyzer = None
            self.format_function = unicode
            self.compare_function = compare_text
        elif type_name == INT_TYPE:
            self.format_function = format_int
            self.compare_function = compare
        elif type_name == FLOAT_TYPE:
            self.format_function = format_float
            self.compare_function = compare_float
        elif type_name == DATE_TYPE:
            self.format_function = format_date
            self.compare_function = compare_date
        else:
            self.format_function = str
            self.compare_function = compare
    
    def get_properties(self):
        
        props = {
            'id' : self.id,
            'name':self.name,
            'type': self._type_name,
            'virtual' : self._virtual,
            'seq' : self.seq_order,
            'has_class' : self._is_class,
            'sparse' : self.is_sparse,
            'classes' : self.classes,
            'common_value' : self.common_value,
            'min_value': self.min_value,
            'max_value' : self.max_value,
            'mean_value' : self.mean_value,
            'median_value' : self.median_value,
            'std_value' : self.std_dev,
            'first_quarter_value' : self.first_quarter_percentile,
            'third_quarter_value' : self.third_quarter_percentile,
            'freq_dist' : self.freq_dist,
            'num_unique_values' : self.num_unique_values,
            'default_function_code' : self.default_function_code,
            'virtual_function_code' : self.virtual_function_code
            
            }
        return props
     
   
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
    
    def get_frequency_distribution_function(self):
        if self.get_type() == TEXT_TYPE:
            df_function = _word_tfidf_dist
        elif self.get_type() == DATE_TYPE:
            df_function = _date_freqdist
        else:
            df_function = _freqdist
        return df_function
    """
        **PUBLIC**
    """
    def get_frequency_distribution(self,filter_function=None):

        return self.table.get_datastore().aggregate(self.name,self.get_frequency_distribution_function(),filter_function)
            
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
    
            
    """
        Refresh the feature properties
        Long running task (value metrics)
    """  
    def _refresh(self):
        
        #count num of values
        self.num_values = self.table.get_datastore().aggregate(self.name,len)
        
        #count num of undefined values
        self.num_undefined_value = self.table.get_datastore().aggregate(self.name,len,lambda dataset,i : not dataset.has_value(self.name,i))
        
        #compute the density distribution (count or tfidf)
        self.freq_dist = self.get_frequency_distribution()
        if self.get_type() == TEXT_TYPE:
            wbag_function = lambda docs : TfidfVectorizer(ngram_range=(1,2),max_features=50).fit(docs) if _check_is_sentence(docs) else None
            self.text_analyzer = self.table.get_datastore().aggregate(self.name,wbag_function)
        
        #is completely dense
        self.is_sparse = self.num_undefined_value > 0
        
        self.num_unique_values = len(self.freq_dist.keys())
        
        #what is the most used value
        sorted_fd = sorted(self.freq_dist.iteritems(), key=lambda (k,v): v*-1)
        self.common_value, common_rank = sorted_fd[0]
        
        #compute feature statistics
        if self.get_type() == INT_TYPE or self.get_type() == FLOAT_TYPE:
            #compute stats
            defined_value_filter = lambda dataset,i : dataset.has_value(self.name,i)
            stats = self._compute_stats(defined_value_filter)
            self.mean_value = stats['mean']
            self.median_value = stats['mean']
            self.first_quarter_percentile = stats['1st-quartile']
            self.third_quarter_percentile = stats['3rd-quartile']
            self.std_dev = stats['std-error']
            self.min_value = stats['min']
            self.max_value = stats['max']
   
    def _compute_stats(self,filter_function):
        return self.table.get_datastore().aggregate(self.name,_compute_stats_function,filter_function)
   
    """
        **PUBLIC**
    """
    def get_distribution_by(self, feature,centile=True,filter_function=None):
        
        stats = self.table.get_datastore().group_by(self,feature,self.get_frequency_distribution_function(),filter_function)
        
        def format_stat(stat):
            serie = {
                     'name' : feature.name + "=" + unicode(int(stat[0]) if feature.get_type()==INT_TYPE else stat[0]),
                         'data' : map(lambda category : round(stat[1][category]*100 if centile else stat[1][category] ,4) if category in stat[1] else 0 ,self.classes)
                    }
            return serie
        
        return map(format_stat,stats)
        
       
    """
        **PUBLIC**
    """
    def get_distribution_stats_by(self,feature,centile=False,filter_function=None):
        
        stats = self.table.get_datastore().group_by(self,feature,_compute_stats_function,filter_function)
        
        def format_stat(stat):
            if stat[1] is not None:
                return [stat[1]['min'],stat[1]['1st-quartile'],stat[1]['median'], stat[1]['3rd-quartile'],stat[1]['max']]
            else:
                return None
        return filter(lambda item : item is not None,map(format_stat,stats))
  
    """
        **PUBLIC**
    """
    def get_metric_by(self,feature,metric_function=sum,filter_function=None):
        
        values = self.table.get_datastore().group_by(self,feature,metric_function,filter_function)
        
        if feature.get_type() == DATE_TYPE:
            values = map(lambda (group,value) : (datetime.date.strftime(group,"%Y-%m-%d"),value),values)
        return values

    """
        Update virtual feature and compute the feature values (long task)
        **PUBLIC**
    """
    def update_feature(self):
        if self.is_virtual():
            self.table.get_datastore().transform(self.name, compile_func_code(self.virtual_function_code))
            self._discover()

    
    """
        **PUBLIC**
    """    
    def get_correlation_with(self,feature,filter_function = None):
        exclude_none_value_function = lambda table,i : (filter_function(table,i) if filter_function is not None else True) and table.has_value(self.name,i) and table.has_value(feature.name,i) and (filter_function(table,i) if filter_function is not None else True)
        feature_ids = [self.name,feature.name]
        return self.table.get_datastore().aggregate_list(feature_ids,scatter_cluster_function,exclude_none_value_function)
    
    # reshape the target value fot the dataset
    def reshape_target(self,value,row_index):
        #if undefined value, then compute the default value function
        if value == NONE_VALUE:        
            value = _exec_func_code(self.default_function_code,self.table,self,row_index)
        
        #Convert TEXT target value as Int value for model compliance            
        if self.get_type() == TEXT_TYPE:
            assert self.has_class(),"Unclassed Text Feature cannot be a target"
            value = self.classes.index(value)
        return value
    
    # reshape the non target value fot the dataset
    def reshape_value(self,value,row_index):
        reshaped_values = []
        
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
                    if feature in self.text_analyzer.build_analyzer()(unicode(value)) :
                        reshaped_values.append(1.0)
                    else:
                        reshaped_values.append(0.0)
            else: #numeric
                reshaped_values.append(float(value))
        
        return reshaped_values
    
