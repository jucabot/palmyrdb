from palmyrdb.converter import TypeConverter, NONE_VALUE, TEXT_TYPE, INT_TYPE,\
    FLOAT_TYPE
import csv
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC,SVR
from sklearn.metrics import confusion_matrix
from palmyrdb.model import ModelInfo, CLASSIFICATION_MODEL, REGRESSION_MODEL
from palmyrdb.script import compile_func_code




class FeatureDataSet():
    _dataset = None
    _feature_set = None
    _row_count = None
    
    def __init__(self,feature_set):
        self._dataset = {}
        self._feature_set = feature_set
        self._row_count = 0
    
    def get_row_count(self):
        return self._row_count
    
    def get_value(self,feature_id,row_index):
        return self._dataset[feature_id][row_index]
    
    def has_value(self,feature_id,row_index):
        return self.get_value(feature_id,row_index) != NONE_VALUE
    
    """
    #A degager a terme
    def get_values(self,name,row_ids=None):
        if row_ids is not None:
            rows = map(lambda row_id : self.get_value(name,row_id),row_ids)
            return rows
        else:
            return self._dataset[name]
    """
    def _filter(self,feature_id,filter_function=None):
        if filter_function is None:
            values = filter (lambda v: v != NONE_VALUE, self._dataset[feature_id] )
        else:
            row_ids = []
            if filter_function is not None:
                for row_index in range(self.get_row_count()): 
                    if filter_function(self,row_index) == False:
                        continue
                    else:
                        row_ids.append(row_index)
            else:
                row_ids = range(self.get_row_count())
            values = map(lambda row_id : self.get_value(feature_id,row_id),row_ids)
        
        return values
    
    def aggregate(self,feature_id,aggregation_function,filter_function=None):
        if filter_function is None:
            result = aggregation_function(self._dataset[feature_id])
        else:
            result = aggregation_function(self._filter(feature_id,filter_function))
        return result
    
    def aggregate_list(self,feature_ids,aggregation_function,filter_function=None):
        if filter_function is None:
            result = aggregation_function(map(lambda feature_id : self._dataset[feature_id],feature_ids))
        else:
            result = aggregation_function(map(lambda feature_id : self._filter(feature_id,filter_function),feature_ids))
        return result
    
    
    def map(self,feature_id,map_function):
        values = map(map_function,self._dataset[feature_id])
        self._dataset[feature_id] = values

    
    def transform(self,feature_id,transform_function):
        values = []
        for row_index in range(self.get_row_count()):
            values.append(transform_function(self,row_index))
        self._dataset[feature_id] = values
     
    
    def load_from_csv(self,filename):
        column_list = []
        row_count = 0
        data_columns = {}
        converter = TypeConverter()
        csv_reader = csv.reader(open(filename, 'rb'))
        
        headers = csv_reader.next()
        headers_len = len(headers)
        
        for row in csv_reader:
            row_count +=1
            for i in range(headers_len):
                value = row[i]
                value = converter.type(value)
                try:
                    data_columns[headers[i]].append(value)
                except KeyError:
                    data_columns[headers[i]] = [value]
        
        self._row_count = row_count
        
        for name in headers:
            column_values = data_columns[name]    
            feature_type = converter.get_type(column_values)
            self._dataset[name] = column_values
            column_list.append((name,feature_type))
            
        return column_list
    
    def take(self,feature_ids,filter_function=None,page=100,from_page=0):
        rows = []

        count = 0
        count_collected = 0
        for i in range(self.get_row_count()):
            row = []
           
            # filter row using filter function code
            if filter_function is not None:
                if filter_function(self,i) == False:
                    continue
            
            count += 1
            
            
            if count <= from_page * page: # go to page
                continue
             
            if count_collected >= page: #limit
                continue
            
            
            for feature_id in feature_ids:
                row.append(self.get_value(feature_id,i))
            rows.append(row)
            count_collected += 1
        return rows, count
    
    
    def _get_dataset(self,filter_function=None,scale_X=True):
        X = []
        y = []
        ds_X = None
        ds_y = None
        
        for row_index in range(self.get_row_count()):
            row = []
            
            # filter row using filter function code
            if filter_function is not None and filter_function(self,row_index) == False:
                continue
            
            for name,feature in self._feature_set.get_features():
                value = self.get_value(name,row_index)
                if not self._feature_set.have_to_discard(feature):
                    if name == self._feature_set.get_target():
                        y.append(feature.reshape_target(value,row_index))
                    else:
                        row.extend(feature.reshape_value(value,row_index))
            X.append(row)
        
        ds_X = np.array(X)
        if scale_X:
            ds_X = scale(ds_X)
        
        if len(y) > 0:
            ds_y = np.array(y)
        
        return ds_X,ds_y
    
    def select_best_features(self,k=10, filter_function=None):
        if self._feature_set.get_target() is not None: #supervised model
            dataset_X,dataset_y = self._get_dataset(filter_function,scale_X=False)
            (nrows,nfeatures) = dataset_X.shape
            kbest = SelectKBest(k=min(k,nfeatures)).fit(dataset_X,dataset_y)
            kbest_score = kbest.scores_
            kbest_mask = kbest.get_support()
            dataset_headers = self._feature_set.get_dataset_labels()
            best_features = {}
            for i in range(len(dataset_headers)):
                if kbest_mask[i] and not np.isnan(kbest_score[i]) :
                    
                    best_features[dataset_headers[i]] = kbest_score[i]
                
            return sorted(best_features.iteritems(), key=lambda (k,v): v*-1)
           
        else:
            return None
    
    """
        Train a prediction model for target
    """
    def build_model(self,C=1.0,kernel='rbf',filter_name=None, filter_code=None):
        
        target = self._feature_set.target
        if target is not None: #supervised model
            target_feature = self._feature_set.get_feature(target)
            
            if filter_code is None:
                target_filter = lambda table,row_index : table.has_value(target,row_index)
            else:
                filter_function = compile_func_code(filter_code)
                target_filter = lambda table,row_index : filter_function(table,row_index) and table.has_value(target,row_index)
            dataset_X,dataset_y = self._get_dataset(filter_function=target_filter)
    
            train_X,test_X, train_y,test_y = train_test_split(dataset_X,dataset_y)
    
            model_info = ModelInfo('%s' % target if filter_name is None else '%s (%s)' % (target,filter_name))
            labels = self._feature_set.get_selected_feature_names()
            model_info.selected_features = labels
            model_info.target = target
            model_info.filter_code = filter_code
            model_info.filter_name = filter_name
            
            if target_feature.has_class(): #classification model
                model = SVC(C=C,kernel=kernel).fit(train_X,train_y)
                score = model.score(test_X,test_y)
                pred_y = model.predict(test_X)
                    
                model_info.model_type = CLASSIFICATION_MODEL
                
                model_info.score = score
                model_info.target_class = target_feature.classes
                model_info.metrics = np.asarray(confusion_matrix(test_y,pred_y),dtype=np.int32).tolist()
                    
                return model,model_info
            
            else: #regression model
                model = SVR(C=C,kernel=kernel).fit(train_X,train_y)
                score = model.score(test_X,test_y)
                
                model_info.model_type = REGRESSION_MODEL
                
                model_info.score = score
                
                return model,model_info
        else: #clustering model
            return None,None
    
    
    
    def apply_prediction(self,model, model_info):

        dataset_X,dataset_y = self._get_dataset()
        target_feature = self._feature_set.get_feature(model_info.target)
        
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
                
        return pred_y
        
    