from palmyrdb.converter import TypeConverter, NONE_VALUE
import csv
import cjson
import os
import json
import pickle


class FeatureDataSet():
    _dataset = None
    _feature_set = None
    _row_count = None
    
    def __init__(self):
        self._dataset = {}
        self._row_count = 0
        
    def init(self,feature_set):
        self._feature_set = feature_set
        
    def store(self):
        f = open(self._feature_set.context['datastore-path'] + os.sep + self._feature_set.id,mode='w')
        pickle.dump(self._dataset,f)
        f.close()
    
    def load(self):
        
        filename = self._feature_set.context['datastore-path'] + os.sep + self._feature_set.id
        
        if os.path.isfile(filename):        
            f = open(filename,mode='r')
            self._dataset = pickle.load(f)
            f.close()
            
    
    def get_row_count(self):
        return self._row_count
    
    def get_value(self,feature_id,row_index):
        return self._dataset[feature_id][row_index]
    
    def has_value(self,feature_id,row_index):
        return self.get_value(feature_id,row_index) != NONE_VALUE
    
    
    def _filter_ids(self,filter_function=None):
        row_ids = []
        if filter_function is not None:
            row_ids = filter(lambda row_index : filter_function(self,row_index),range(self.get_row_count()))
        else:
            row_ids = range(self.get_row_count())
        return row_ids
    
    def _filter(self,feature_id,filter_function=None):
        
           
        if filter_function is None:
            values = filter(lambda v: v != NONE_VALUE, self._dataset[feature_id] )
        else:
            values = map(lambda row_id : self.get_value(feature_id,row_id),self._filter_ids(filter_function))
        
        return values
    
    
    def group_by(self,feature, grouping_feature,metric_function=sum,filter_function=None):
        result = []
        
        def global_filter_function(dataset,row_index):
                return (filter_function(dataset,row_index) if filter_function is not None else True) and dataset.has_value(feature.name,row_index)
            
        row_ids = self._filter_ids(filter_function)
        
        values = map(lambda row_id : (self.get_value(grouping_feature.name,row_id),self.get_value(feature.name,row_id)),self._filter_ids(filter_function))
        groups = {}
        for (group,value) in values:
            if value == NONE_VALUE:
                continue
            try:
                groups[group].append(value)
            except KeyError:
                groups[group] = [value]
        
        #remove undefined values
        if NONE_VALUE in groups:
            del groups[NONE_VALUE]
            
        for group,values in groups.items():
            result.append([group,metric_function(values)])
        return result
    
    
    def aggregate(self,feature_id,aggregation_function,filter_function=None):
       
        values = self._dataset[feature_id]
        if filter_function is None:
            result = aggregation_function(values)
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
        values = map(lambda row_index : transform_function(self,row_index),range(self.get_row_count()))
                
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
        
        self.store()
        
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
    
    
    
   
        
    