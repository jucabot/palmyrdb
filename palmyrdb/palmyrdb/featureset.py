from copy import copy
import math
from palmyrdb.script import compile_func_code
from palmyrdb.features import Feature
from datastore import memstore
import csv


"""
    Set of feature
"""    
class FeatureTable():
    _features = None
    _seq_order = None
    context = None
    _datastore = None
    target = None
    params = None
    models = None
    current_model = None
    filters = None
    current_filter = None
   
        
    def __init__(self,context=None):
        self.context = context
        self._features = {}
        self._seq_order = 0
        #self._row_count = 0
        self.target = None
        self.params = {}
        self.models = {}
        self.current_model = None
        self.filters = {}
        self.current_filter = None
        self._datastore = memstore.FeatureDataSet(self) #should be loaded by configuration

    def _get_next_seq_order(self):
        self._seq_order +=1
        return self._seq_order
    
    def get_datastore(self):
        return self._datastore
    
    def _get_virtual_features(self):
        virtual_features = filter(lambda (k,v) : v.is_virtual(),self.get_features())
        
        return virtual_features
    
    """
        Retrive the features of the set sorted by order
    """
    def get_features(self):
        return sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)
    """
        Return the list of feature names sorted by order
    """
    def get_feature_names(self):
        return [ name for (name,feature) in sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)]
    """
        Retrive the feature by name (as set by set_feature)
    """
    def get_feature(self,name):
        return self._features[name]
    """
        Is there a such feature in the feature set
    """
    def has_feature(self,name):
        return name in self._features
    """
        Add a feature to the feature set
    """
    def set_feature(self,name,feature):
        self._features[name] = feature
    
    def get_row_count(self):
        return self._datastore.get_row_count()
    """
        Set the target (predicted) feature
    """
    def set_target(self,name):
        self.target = name
        self.use_feature(name,use=True)
    
    def get_target(self):
        return self.target
    
    """
        Reset the defined target feature
    """    
    def reset_target(self):
        self.target = None
        
    def get_selected_feature_names(self):
        return [ f[0] for f in filter(lambda (k,v) : v.is_usable() and k != self.target,self.get_features())]
    """
        Enable feature in model training
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
    def have_to_discard(self,feature):
        return not (feature.is_usable() or feature.name == self.target)
    
    """
        Return the name of the features in the dataset
    """
    def get_dataset_labels(self):
        headers = []
        for name,feature in self.get_features():
            if not self.have_to_discard(feature):
                if name != self.target:
                    headers.extend(self.get_feature(name)._reshape_header())
        return headers
    
    """
        Remove a virtual feature
    """    
    def remove_feature(self,name):
        assert self.get_feature(name).is_virtual(), "The feature %s has to be virtual to be removed" % name
            
        del self._features[name]
    
    
    """
        Add a virtual feature and compute the feature values
        (long running)
    """
    def add_feature(self,name,type_name,function_code):
        
        self._datastore.transform(name, compile_func_code(function_code))
        
        return self._load_feature(name, type_name, virtual=True, virtual_function_code=function_code)
    
    """
        Force recompute the virtual features
        (long running)
    """
    def recompute_virtual_features(self):    
        for name, feature in self._get_virtual_features():
            self.add_feature(name, feature.get_type(), feature.virtual_function_code)
    
    
    def _load_feature(self,name,type_name,virtual=False,virtual_function_code=None):
        
        if name in self._features: #Feature already exist
            feature = self.get_feature(name)
            feature.virtual_function_code = virtual_function_code
            self.get_datastore().map(name, feature.format_function) #f
            feature._refresh()

        else:
            feature = Feature.create_feature(self,name,type_name,virtual)
            self.set_feature(name, feature)
            feature.seq_order = self._get_next_seq_order()
            feature.virtual_function_code = virtual_function_code
            self.get_datastore().map(name, feature.format_function) #force value types
            feature._discover()
            
        return feature
    
    """
        Clone the feature set metadata
    """    
    def clone(self):
        fs_clone =  copy(self)
        #fs_clone._datastore = memstore.FeatureDataSet(fs_clone) #should be loaded by configuration

        return fs_clone
    """
        Initialize a feature set and load data to the data store from a CSV file based on headers
        (long running)
    """
    def load_from_csv(self,filename):
        columns = self.get_datastore().load_from_csv(filename)
        for fname,ftype in columns:
            self._load_feature(fname,ftype)        
        return self
    
    """
        **PUBLIC**
    """    
    def get_datatable(self,filter_function=None,page=100,from_page=0):
        result = {}

        result['type'] = 'table'
        
        result['rows'] = []
        result['features'] = self.get_feature_names()
        
        result['rows'], count = self.get_datastore().take(self.get_feature_names(),page=page,from_page=from_page,filter_function=filter_function)
        
        result['num_total_rows'] = self.get_row_count()
        result['num_rows'] = count
        num_pages = int(math.floor(float(count) / page) + 1 if count % page > 0 else 0 ) #int(math.ceil(float(self.get_row_count()) / page))
        result['list_pages'] = range(num_pages)
        page_num = from_page
        result['page_num'] = page_num
        result['page_total'] = num_pages
        result['has_prev'] = True if page_num >0 else False
        result['has_next'] = True if page_num < num_pages-1 else False
        return result
    
    """
        **PUBLIC**
    """    
    def select_best_features(self,k=10,filter_code=None):
        filter_function = compile_func_code(filter_code) if filter_code is not None else None
        return self.get_datastore().select_best_features(k=k,filter_function=filter_function)
    
    """
        Train a prediction model for target
        (long running)
    """
    
    def build_model(self,filter_name=None,filter_code=None,C=1.0,kernel='rbf'):
        return self.get_datastore().build_model(C=C, kernel=kernel,filter_name=filter_name,filter_code=filter_code)

        
    """
        Make prediction based on a saved model
        (long running)
    """    
    def apply_prediction(self,target_name,input_filename, output_filename):
        ftable = self.clone()
        ftable.load_from_csv(input_filename)
        ftable.recompute_virtual_features()
        
        #model,model_info = self.models[model_name]
        
        models = self.get_model_for_target(target_name)
        y_preds = {}
        final_ypred = []
        for model, model_info in models:
            y_preds[model_info.name] = ftable.get_datastore().apply_prediction(model, model_info)
        
        for i in range(ftable.get_row_count()):
            votes = {}
            for name, y_pred in y_preds.items():
                if y_pred[i] in votes:
                    votes[y_pred[i]] +=1
                else:
                    votes[y_pred[i]] =1
            sorted_votes = sorted(votes.iteritems(), key=lambda (k,v): -v)
            final_ypred.append(sorted_votes[0][0])
            
        self._write_prediction(final_ypred, input_filename, output_filename)
        
    def add_filter(self,name,code):
        
        filter_info = {'name' : name, 'code': code}  
        self.filters[name] = code
        self.current_filter = filter_info
   
    def get_model_for_target(self,target_name):
        
        selected_models = filter(lambda (model,model_info) : model_info.target == target_name,self.models.values())
        return selected_models

    def _write_prediction(self,pred_y,input_filename, output_filename):
        open_file_object = csv.writer(open(output_filename, "wb"))
        pred_file_object = csv.reader(open(input_filename, 'rb')) #Load in the csv file
            
        pred_file_object.next()
        i = 0
        for row in pred_file_object:
            row.insert(0,pred_y[i])
            open_file_object.writerow(row)
            i += 1