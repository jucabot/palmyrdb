import csv
import numpy as np
from sklearn.preprocessing import scale
from numpy.ma.core import mean, std, asarray, array
from numpy.lib.function_base import median, percentile
from pickle import dump,load
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.svm import SVC,SVR
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
import os
from palmyrdb.converter import TypeConverter, FLOAT_TYPE, INT_TYPE, TEXT_TYPE,NONE_VALUE
import numpy
from copy import copy
import math



def _exec_func_code(function_code,*args):
    exec(function_code)
    return function(*args)

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
            freq[value] = freq[value]/values_len
        
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
        self.set_values(values)
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
        Get the defined values and force typing
        Long running task
    """  
    def get_defined_values(self,row_ids=None):
        if self.get_type() == INT_TYPE or self.get_type() == FLOAT_TYPE:
            return map(lambda x : float(x),filter (lambda a: a != NONE_VALUE, self.get_values(row_ids)))
        elif self.get_type() == TEXT_TYPE:
            return map(lambda x : str(x),filter (lambda a: a != NONE_VALUE, self.get_values(row_ids)))
        else:
            return filter (lambda a: a != NONE_VALUE, self.get_values(row_ids))
    """
        Get the undefined values
        Long running task
    """  
    def get_undefined_values(self,row_ids=None):
        return filter (lambda a: a == NONE_VALUE, self.get_values(row_ids))
   
    """
        Auto discover the feature properties while creating the analysis object
        Long running task
    """  
    def discover(self):
        
        #compute stats
        self.refresh()
        
        #classes are fixed by discover
        self.set_class((1.0 - (float(self.num_unique_values) / float(self.num_values))) > 0.99)
        self.classes = self.freq_dist.keys() #classes are fixed by discover
        self.default_function_code = "function = lambda table,feature,row_index : feature.common_value"    
           
        
    """
       Is this feature is allowed for target (unclassed text is not allowed)
        Long running task
    """  
    def target_allowed(self):
        return not (self.get_type() == TEXT_TYPE and not self.has_class())
    
    """
       Get feature type - type is set at feature creation only
        Long running task
    """  
    def get_type(self):
        return self._type_name
    
    
    """
        Define if the feature is a class Feature
    """  
    def set_class(self,is_class):
        self._is_class = is_class
    
    """
        Is a classed feature?
    """            
    def has_class(self):
        return self._is_class
    
    def filter(self,filter_function=None):
        if filter_function is None:
            values = self.get_defined_values()
        else:
            ids = self.table.get_row_ids(filter_function)
            values = self.get_defined_values(row_ids=ids)
        
        return values
    
    def get_frequency_distribution(self,filter_function=None):
        
        values = self.filter(filter_function)

        if self.get_type() == TEXT_TYPE:
            return _word_tfidf_dist(values)
        else:
            return _freqdist(values)
    
    def _compute_stats(self):
        if self.get_type() == INT_TYPE or self.get_type() == FLOAT_TYPE:
            distinct_values = self.freq_dist.keys()
            self.min_value = min(distinct_values)
            self.max_value = max(distinct_values)
            values = self.get_defined_values()
            self.mean_value = mean(values)
            self.median_value = median(values)
            self.first_quarter_percentile = percentile(values,25)
            self.third_quarter_percentile = percentile(values,75)
            self.std_dev = std(values)
    
    """
        Refresh the feature properties
        Long running task (value metrics)
    """  
    def refresh(self):
        
        self.num_values = len(self.get_values())
        self.num_undefined_value = len(self.get_undefined_values())
        
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
    def get_values(self,row_ids=None):
        
        if row_ids is not None:
            rows = map(lambda row_id : self.get_value(row_id),row_ids)
            return rows
        else:
            return self._values
    
    
    """
        Get a feature-row value - must be externalized to support large values
    """
    def get_value(self,row_index):
        if self.get_type() == TEXT_TYPE:
            return str(self._values[row_index])
        else:
            return self._values[row_index]
    
    """
        Set the feature values - must be externalized to support large values
    """
    def set_values(self,values):
        self._values = values
    
    # reshape the target value fot the dataset
    def _reshape_target(self,row_index):
        value = self.get_value(row_index)
        
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
        value = self.get_value(row_index)
        
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
        Update virtual feature and compute the feature values (long task)
    """
    def update_feature(self):
        if self.is_virtual():
            values = self.table._load_virtual_values(self.virtual_function_code)
            self.set_values(values)
            self.discover()

    def get_distribution(self,centile=False):
        freq_dist = []
        
        for category in self.classes:
               
                group_freq = self.freq_dist
                serie = {
                         'name' : self.name + "=" + unicode(int(category) if self.get_type()==INT_TYPE else category),
                         'data' : map(lambda category : round(group_freq[category]*100 if centile else group_freq[category] ,4) if category in group_freq else 0 ,self.classes)
                         }
                freq_dist.append(serie)
        return freq_dist


    def get_distribution_by(self, feature,centile=False):
        freq_dist = []
        
        for category in feature.classes:
                filter_function = lambda table,row_index : table.get_feature(feature.name).is_value_equal(row_index,category) 
                group_freq = self.get_frequency_distribution(filter_function)
                
                serie = {
                         'name' : feature.name + "=" + unicode(int(category) if feature.get_type()==INT_TYPE else category),
                         'data' : map(lambda category : round(group_freq[category]*100 if centile else group_freq[category] ,4) if category in group_freq else 0 ,self.classes)
                         }
                freq_dist.append(serie)
        return freq_dist
    
    def get_distribution_stats_by(self,feature,centile=False):
        result = []
        for category in feature.classes:
            filter_function = lambda table,row_index : table.get_feature(feature.name).is_value_equal(row_index,category) 
            group_values = self.filter(filter_function)
            
            if (len(group_values) >1):
                stat_min = min(group_values)
                stat_max = max(group_values)
                stat_median = median(group_values)
                stat_first_quartile = percentile(group_values,25)
                stat_third_quartile = percentile(group_values,75)
                
                result.append([stat_min,stat_first_quartile,stat_median, stat_third_quartile,stat_max])
        return result
    
    def is_value_equal(self,row_index, value_to_compare):

        if self.get_type() == TEXT_TYPE: #case insensitive comparaison
            return value_to_compare.lower() in self.get_value(row_index).lower()
        elif self.get_type() == FLOAT_TYPE:
            return round(float(self.get_value(row_index))) == round(float(value_to_compare))
        else:
            return self.get_value(row_index) == value_to_compare
        
    def get_correlation_with(self,feature,filter_function = None):
        row_ids = self.table.get_row_ids(filter_function)
        return map(lambda i : [self.get_value(i),feature.get_value(i)],row_ids)
        
        
CLASSIFICATION_MODEL = 'CLAS'
REGRESSION_MODEL = 'REGR'
CLUSTERING_MODEL = 'CLUS'

class ModelInfo():
    name = None
    model_type = None
    selected_features = None
    score = None
    metrics = None
    target = None
    target_class = None
    
    def __init__(self,name):
        self.name = name
        self.model_type = None
        self.selected_features = None
        self.score = None
        self.metrics = None
        self.target = None
        self.target_class = None

    def get_properties(self):
        props = {
                 'name' : self.name,
                 'model_type' : self.model_type,
                 'selected_features' : self.selected_features,
                 'score' : self.score,
                 'metrics' : self.metrics,
                 'target' : self.target,
                 'target_class' : self.target_class
                 }
        return props

"""
    Set of feature
"""    
class FeatureTable():
    _features = None
    _seq_order = None
    _row_count = None
    target = None
    params = None
    models = None
    current_model = None
   
        
    def __init__(self):
        self._features = {}
        self._seq_order = 0
        self._row_count = 0
        self.target = None
        self.params = {}
        self.models = {}
        self.current_model = None

    def _get_next_seq_order(self):
        self._seq_order +=1
        return self._seq_order
    
    
    def _load_columns(self,reader,headers):
        converter = TypeConverter()
        data_columns = {}
        row_count = 0
        headers_len = len(headers)
        for row in reader:
            row_count +=1
            for i in range(headers_len):
                value = row[i]
                value = converter.type(value)
                try:
                    data_columns[headers[i]].append(value)
                except KeyError:
                    data_columns[headers[i]] = [value]
        
        
        return data_columns,row_count

    #Create (auto discover stats) or update (refresh stats) the feature values (long task)   
    def _load_feature(self,name,type_name,values,virtual=False,virtual_function_code=None):
        
        if name in self._features: #Feature already exist
            feature = self.get_feature(name)
            feature.set_values(values)
            feature.virtual_function_code = virtual_function_code
            feature.refresh()

        else:
            feature = Feature.create_feature(self,name,type_name,values,virtual)
            feature.seq_order = self._get_next_seq_order()
            feature.virtual_function_code = virtual_function_code
            feature.discover()
            self._features[name] = feature
        return feature
    
    def _get_virtual_features(self):
        virtual_features = filter(lambda (k,v) : v.is_virtual(),self.get_features())
        
        return virtual_features
    
    #Compute virtual feature values (long task)    
    def _load_virtual_values(self,function_code):
        values = []
        for row_index in range(self._row_count):
            values.append(_exec_func_code(function_code,self,row_index))
        return values
    
    """
        Recompute the virtual features
    """
    def refresh(self):    
        for name, feature in self._get_virtual_features():
            self.add_feature(name, feature.get_type(), feature.virtual_function_code)
    """
        Add a virtual feature and compute the feature values (long task)
    """
    def add_feature(self,name,type_name,function_code):
        values = self._load_virtual_values(function_code)

        return self._load_feature(name, type_name, values, virtual=True, virtual_function_code=function_code)
    
    """
        Create a feature table from a CSV file based on headers (long task)
    """
    @staticmethod
    def create_from_csv(filename):
        ftable = FeatureTable()
        ftable.load_from_csv(filename)
        
        return ftable
    """
        Create or apply a feature table from a CSV file based on headers (long task)
    """
    def load_from_csv(self,filename):
        converter = TypeConverter()
        csv_reader = csv.reader(open(filename, 'rb'))
        headers = csv_reader.next()
        columns,self._row_count = self._load_columns(csv_reader, headers)
        
        for name in headers:
            column_values = columns[name]    
            feature_type = converter.get_type(column_values)
            self._load_feature(name,feature_type,column_values)
        return self
    
    def get_features(self):
        return sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)
    
    def get_feature_names(self):
        return [ name for (name,feature) in sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)]
    
    def get_feature(self,name):
        return self._features[name]
    
    def has_feature(self,name):
        return name in self._features
    
    def set_feature(self,name,feature):
        self._features[name] = feature
    
    def get_value(self,name,row_index):
        return self.get_feature(name).get_value(row_index)
    
    def has_value(self,name,row_index):
        return self.get_feature(name).get_value(row_index) != NONE_VALUE
    
    def get_shaped_value(self,name,row_index):
        return self.get_feature(name)._reshape_value(row_index)
    
    def get_shaped_row(self,names,row_index,scale_row=True):
        row = []
        for name in names:
            row.extend(self.get_shaped_value(name, row_index))
        
        if scale_row:
            return scale(np.array([row]))
        else:
            return np.array([row])
    
    def get_row_count(self):
        return self._row_count
    
    def set_target(self,name):
        self.target = name
        self.use_feature(name,use=True)
        
    def reset_target(self):
        self.target = None
        
    def get_selected_feature_names(self):
        return [ f[0] for f in filter(lambda (k,v) : v.is_usable() and k != self.target,self.get_features())]
    
    def use_feature(self,name,use=False):
        self.get_feature(name)._usable = use
    
    def discard_features(self,names):
        for name in names:
            self.use_feature(name,use=False)
    
    def display_features(self):
        
        headers = ["Name","Type","class","usable","sparse","virtual","common value"]
        
        print '\t'.join(headers)
        print '-' * 32 * len(headers)
        
        for name, feature in self.get_features() :   
            print '%s\t' * len(headers) % (name if name != self.target else name + '(*)' ,feature.get_type(),feature.has_class() if feature.has_class() == False else feature.num_unique_values ,feature.is_usable(),feature.is_sparse,feature.is_virtual(),feature.common_value)
    """
        Define if the feature may be part of the dataset, based on is_usable
    """
    def _have_to_discard(self,feature):
        return not (feature.is_usable() or feature.name == self.target) #or (feature.get_type() == TEXT_TYPE and not feature.has_class()) #text
    
    """
        Transform and return the feature data as machine learning compliant array
        The classes are splited as different features
        filter_function filters the resulted rows among the dataset
        Discarded features are excluded among the dataset
    """
    def get_dataset(self,filter_function_code=None,scale_X=True):
        X = []
        y = []
        ds_X = None
        ds_y = None
        
        for row_index in range(self._row_count):
            row = []
            
            # filter row using filter function code
            if filter_function_code is not None and _exec_func_code(filter_function_code,self,row_index) == False:
                continue
            
            for name,feature in self.get_features():
                if not self._have_to_discard(feature):
                    if name == self.target:
                        y.append(self.get_feature(name)._reshape_target(row_index))
                    else:
                        row.extend(self.get_feature(name)._reshape_value(row_index))
            X.append(row)
        
        ds_X = np.array(X)
        if scale_X:
            ds_X = scale(ds_X)
        
        if len(y) > 0:
            ds_y = np.array(y)
        
        return ds_X,ds_y
    
    """
        Return the name of the features in the dataset
    """
    def get_dataset_labels(self):
        headers = []
        for name,feature in self.get_features():
            if not self._have_to_discard(feature):
                if name != self.target:
                    headers.extend(self.get_feature(name)._reshape_header())
        return headers
    
    def display_dataset(self,sample=10):
        
        X, y = self.get_dataset(scale_X=False)
        headers = self.get_dataset_labels()

        label_headers = "%-14s | " % self.target if self.target != None else ""
        sep = ""
        for name in headers:
            label_headers += name  +"\t"
            sep += "----------------------"
        print label_headers
        print sep
        
        count=0
        for row in X:
            
            if count>sample:
                break
            row_repr = "%-14s | " % str(y[count]) if  y != None else ""
            for i in range(len(row)):
                row_repr += str(row[i])  +"\t"
            print row_repr
            count +=1

    def refine(self,filter_function_code):
        
        row_to_filter = []
        for row_index in range(self._row_count):
            
            if _exec_func_code(filter_function_code,self,row_index) == False:
                row_to_filter.append(row_index)
            
        for name,feature in self.get_features():
            feature.values = [i for j, i in enumerate(feature.get_values()) if j not in row_to_filter]
            
        self._row_count -= len(row_to_filter)
        self.refresh()
        return self._row_count
    
    @staticmethod
    def open(file_path):
        f = open(file_path,'rb')
        obj = load(f)
        f.close()
        return obj
    
    def save(self,file_path):
        dir_path = file_path.split(os.sep)[:-1]
        dir_path = os.sep.join(dir_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        f = open(file_path,'wb')
        dump(self,f)
        f.close()
        

    def build_model(self,C=1.0,kernel='rbf'):
        dataset = {}
        if self.target is not None: #supervised model
            target_feature = self.get_feature(self.target)
            
            target_filter_code = "function = lambda table,row_index : table.has_value('%s',row_index)" % self.target
            dataset_X,dataset_y = self.get_dataset(filter_function_code=target_filter_code)
    
            train_X,test_X, train_y,test_y = train_test_split(dataset_X,dataset_y)
    
            dataset['train-X'] = train_X
            dataset['train-y'] = train_y
            dataset['test-X'] = train_X
            dataset['test-y'] = train_y
            
            model_info = ModelInfo('model-%s' % self.target)
            labels = self.get_selected_feature_names()
            model_info.selected_features = labels
            model_info.target = self.target
            
            if target_feature.has_class(): #classification model
                model = SVC(C=C,kernel=kernel).fit(train_X,train_y)
                score = model.score(test_X,test_y)
                pred_y = model.predict(test_X)
                
                
                model_info.model_type = CLASSIFICATION_MODEL
                
                model_info.score = score
                model_info.target_class = self.get_feature(self.target).classes
                model_info.metrics = asarray(confusion_matrix(test_y,pred_y),dtype=np.int32).tolist()
                    
                return model,model_info
            
            else: #regression model
                model = SVR(C=C,kernel=kernel).fit(train_X,train_y)
                score = model.score(test_X,test_y)
                
                model_info.model_type = REGRESSION_MODEL
                
                model_info.score = score
                
                return model,model_info
        else: #clustering model
            return None,None
        
    def remove_feature(self,name):
        assert self.get_feature(name).is_virtual(), "The feature %s has to be virtual to be removed" % name
            
        del self._features[name]
        
    def select_best_features(self,k=10):
        if self.target is not None: #supervised model
            dataset_X,dataset_y = self.get_dataset(scale_X=False)
            (nrows,nfeatures) = dataset_X.shape
            kbest = SelectKBest(k=min(k,nfeatures)).fit(dataset_X,dataset_y)
            kbest_score = kbest.scores_
            kbest_mask = kbest.get_support()
            dataset_headers = self.get_dataset_labels()
            best_features = {}
            for i in range(len(dataset_headers)):
                if kbest_mask[i] and not numpy.isnan(kbest_score[i]) :
                    
                    best_features[dataset_headers[i]] = kbest_score[i]
                
            return sorted(best_features.iteritems(), key=lambda (k,v): v*-1)
           
        else:
            return None
    def _write_prediction(self,pred_y,input_filename, output_filename):
        open_file_object = csv.writer(open(output_filename, "wb"))
        pred_file_object = csv.reader(open(input_filename, 'rb')) #Load in the csv file
            
        pred_file_object.next()
        i = 0
        for row in pred_file_object:
            row.insert(0,pred_y[i])
            open_file_object.writerow(row)
            i += 1
        
        
    def apply_prediction(self,model_name,input_filename, output_filename):
        ftable = copy(self) #copy because data are hold inside the object - to be decomissioned
        ftable.load_from_csv(input_filename)
        ftable.refresh()
        
        dataset_X,dataset_y = ftable.get_dataset()
        assert model_name in ftable.models, "model %s is not defined" % model_name
        model = ftable.models[model_name][0] #get model
        
        target_feature = ftable.get_feature(ftable.target)
        
        pred_y = model.predict(dataset_X)
        
        if target_feature.has_class():
            if target_feature.get_type() == INT_TYPE:
                pred_y = pred_y.astype(np.uint8)
            else:
                pred_y = [target_feature.classes[int(y)] for y in pred_y]
        else:        
            if target_feature.get_type() == INT_TYPE:
                pred_y = pred_y.astype(np.uint8)
            elif target_feature.get_type() == FLOAT_TYPE:
                pred_y = pred_y.astype(np.float64)
        self._write_prediction(pred_y, input_filename, output_filename)
        
    def get_datatable(self,filter_function_code=None,page=100,from_page=0):
        result = {}

        result['type'] = 'table'
        
        result['rows'] = []
        result['features'] = self.get_feature_names()
        count = 0
        for row_index in range(self._row_count):
            row = []
           
            # filter row using filter function code
            if filter_function_code is not None and _exec_func_code(filter_function_code,self,row_index) == False:
                continue
            
            if row_index <= from_page * page: # go to page
                continue
             
            if count >= page: #limit
                break
            
            count += 1
            
            for name,feature in self.get_features():
                row.append(self.get_feature(name).get_value(row_index))
            result['rows'].append(row)
        
        result['num_total_rows'] = self.get_row_count()
        result['num_rows'] = count
        num_pages = int(math.ceil(float(self._row_count) / page))
        result['list_pages'] = range(num_pages)
        page_num = from_page
        result['page_num'] = page_num
        result['page_total'] = num_pages
        result['has_prev'] = True if page_num >0 else False
        result['has_next'] = True if page_num < num_pages else False
        return result
    
    def get_row_ids(self,filter_function=None):
        row_ids = []
        if filter_function is not None:
            for row_index in range(self.get_row_count()): 
                if filter_function(self,row_index) == False:
                    continue
                else:
                    row_ids.append(row_index)
        else:
            row_ids = range(self.get_row_count())
        return row_ids
    
