import csv
import numpy as np
from sklearn.preprocessing import scale
from numpy.ma.core import asarray
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
from palmyrdb.script import _exec_func_code
from palmyrdb.model import ModelInfo, CLASSIFICATION_MODEL, REGRESSION_MODEL
from palmyrdb.features import Feature


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
    
    def _get_virtual_features(self):
        virtual_features = filter(lambda (k,v) : v.is_virtual(),self.get_features())
        
        return virtual_features
    
    """
        **PUBLIC**
    """
    def get_features(self):
        return sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)
    """
        **PUBLIC**
    """
    def get_feature_names(self):
        return [ name for (name,feature) in sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)]
    """
        **PUBLIC**
    """
    def get_feature(self,name):
        return self._features[name]
    """
        **PUBLIC**
    """
    def has_feature(self,name):
        return name in self._features
    """
        **PUBLIC**
    """
    def set_feature(self,name,feature):
        self._features[name] = feature
    
    def get_row_count(self):
        return self._row_count
    """
        **PUBLIC**
    """
    def set_target(self,name):
        self.target = name
        self.use_feature(name,use=True)
    """
        **PUBLIC**
    """    
    def reset_target(self):
        self.target = None
        
    def get_selected_feature_names(self):
        return [ f[0] for f in filter(lambda (k,v) : v.is_usable() and k != self.target,self.get_features())]
    """
        **PUBLIC**
    """
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
        Return the name of the features in the dataset
    """
    def get_dataset_labels(self):
        headers = []
        for name,feature in self.get_features():
            if not self._have_to_discard(feature):
                if name != self.target:
                    headers.extend(self.get_feature(name)._reshape_header())
        return headers
    
    """
        **PUBLIC**
    """    
    def remove_feature(self,name):
        assert self.get_feature(name).is_virtual(), "The feature %s has to be virtual to be removed" % name
            
        del self._features[name]
    
    
    """
        **PUBLIC**
    """
    @staticmethod
    def open(file_path):
        f = open(file_path,'rb')
        obj = load(f)
        f.close()
        return obj
    """
        **PUBLIC**
    """
    def save(self,file_path):
        dir_path = file_path.split(os.sep)[:-1]
        dir_path = os.sep.join(dir_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        f = open(file_path,'wb')
        dump(self,f)
        f.close()
    
    
    
    """
    **************************************************************************************
    """
    
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
            self.set_values(name,values)
            feature.virtual_function_code = virtual_function_code
            feature._refresh()

        else:
            feature = Feature.create_feature(self,name,type_name,virtual)
            self.set_feature(name, feature)
            self.set_values(name, values)
            feature.seq_order = self._get_next_seq_order()
            feature.virtual_function_code = virtual_function_code
            feature._discover()
            
        return feature
    
    
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
        **PUBLIC**
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
        **PUBLIC**
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
    
    """
        **PUBLIC** - custom function
    """
    def get_value(self,name,row_index):
        #return self.get_feature(name)._get_value(row_index)
        feature = self.get_feature(name)
        if feature.get_type() == TEXT_TYPE:
            return str(feature._values[row_index])
        else:
            return feature._values[row_index]
    
    
    
    def get_values(self,name,row_ids=None):
        
        if row_ids is not None:
            rows = map(lambda row_id : self.get_value(name,row_id),row_ids)
            return rows
        else:
            return self.get_feature(name)._values
    
    def set_values(self,name,values):
        self.get_feature(name)._values = values
    
    """
        **PUBLIC** - custom function
    """
    def has_value(self,name,row_index):
        return self.get_value(name,row_index) != NONE_VALUE
    
    
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
        **PUBLIC**
    """
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
    """
        **PUBLIC**
    """    
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
        
    """
        **PUBLIC**
    """    
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
    """
        **PUBLIC**
    """    
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
                row.append(self.get_feature(name)._get_value(row_index))
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
    
    def _get_row_ids(self,filter_function=None):
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
    
