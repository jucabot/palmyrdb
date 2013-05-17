import math
from palmyrdb.script import compile_func_code
from palmyrdb.features import Feature
import uuid
from palmyrdb.converter import DATE_TYPE, INT_TYPE, FLOAT_TYPE
from palmyrdb import load_from_classname
from datastore.memstore import FeatureDataSet
import cjson


"""
    Set of feature
"""    
class FeatureTable():
    _features = None
    _seq_order = None
    context = None
    _datastore = None
    params = None
    filters = None
    current_filter = None
    id = None
   
    @staticmethod
    def create(context=None):
        fs = FeatureTable(context)
        fs.id = str(uuid.uuid4())
        return fs
    
    
    def __init__(self,context=None):
        self.id= None
        self.context = context
        self._features = {}
        self._seq_order = 0
        self.params = {}
        self.filters = {}
        self.current_filter = None
        self._datastore = None #lazy loading


    def save(self):
        return self
    
    def load(self):
        return self

    def _get_next_seq_order(self):
        self._seq_order +=1
        return self._seq_order
    
    def get_datastore(self):
        if self._datastore is None:
            self._datastore = FeatureDataSet() #load_from_classname(context['datastore-engine'])
            self._datastore.init(self)
            self._datastore.load()
        return self._datastore
    
    def _get_virtual_features(self):
        virtual_features = filter(lambda (k,v) : v.is_virtual(),self.get_features())
        
        return virtual_features
    
    """
        Retrieve the features of the set sorted by order
    """
    def get_features(self):
        return sorted(self._features.iteritems(), key=lambda (k,v): v.seq_order)
    
    """
        Retrieve the features by type of the set sorted by order
    """
    def get_features_by_types(self,type_names):
        features = filter(lambda (k,v) : v.get_type() in type_names,self._features.iteritems())
        return sorted(features, key=lambda (k,v): v.seq_order)
    
    """
        Retrieve the date features of the set sorted by order
    """
    def get_date_features(self):
        return self.get_features_by_types([DATE_TYPE]) 
    
    """
        Retrieve the numerical features of the set sorted by order
    """
    def get_numerical_features(self):
        return self.get_features_by_types([INT_TYPE, FLOAT_TYPE]) 
    
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
        return self.get_datastore().get_row_count()
   
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
        
        self.get_datastore().transform(name, compile_func_code(function_code))
        
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
    
    def add_filter(self,name,code):
        
        filter_info = {'name' : name, 'code': code}  
        self.filters[name] = code
        self.current_filter = filter_info
       
    def get_properties(self):
        
        fset = {
                'id' : self.id,
                'params' : self.params,
                'filters' : self.filters,
                'features' : map(lambda v : v[1].get_properties(),self.get_features())
                
                }
        return fset
        