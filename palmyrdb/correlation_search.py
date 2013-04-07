"""
Proto for PySpark on correlation search
"""

from pyspark.context import SparkContext
import os
import json
import cjson
import datetime
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from multiprocessing import cpu_count
from operator import add


def get_date(date_str, date_format="%Y-%m-%d %H:%M:%S"):
    try:
        date = datetime.datetime.strptime(date_str,date_format)
        key =  format_date(date)
    except ValueError:
        key = "Undefined"
    return key


def format_date(date, date_format="%Y-%m"):
    return datetime.datetime.strftime(date,date_format)

def read_date(str_date, date_format="%Y-%m"):
    try:
        return datetime.datetime.strptime(str_date,date_format)
    except ValueError:
        return None   

def serie_join(serie1, serie2):
    list1 = []
    list2 = []
    list_pivot = []
    for key in serie1.keys():
        if key in serie2:
            list_pivot.append(key)
            list1.append(serie1[key])
            list2.append(serie2[key])
    
    return (list_pivot, list1, list2)

def lag_serie(serie,month_lag):
    
    lagged_serie = {}
    
    for key in serie:
        date = read_date(key)
        
        if date != None:
            delta = datetime.timedelta(month_lag * 365/12)
            
            date = date - delta
            
            lagged_serie[format_date(date)] = serie[key]
    
    return lagged_serie

def group_date_map(line,date_feature_seq, value_feature_seq,date_format):
    values = line.split(',')
    return (get_date(values[date_feature_seq],date_format),float(values[value_feature_seq]))



def correlation_search_map(line,search_timeserie_data,lag_variable,kernel_variable):
    
    predictor_data = line.split(';')
    predictor_key = predictor_data[0]
    
    predictor = cjson.decode(predictor_data[1])
    
    
    for variable_key, variable in search_timeserie_data.value.items():
        key = str(variable_key) + '^' + str(predictor_key)
                
        original_predictor = predictor.values()
        original_X = np.array(original_predictor,ndmin=2)
        original_X = original_X.reshape((-1,1))
        #original_X = scale(original_X)
        
        results = {}        
        for i in range(lag_variable.value+1):
            
            lagged_predictor = lag_serie(predictor, i)
            
            (list_key,list_variable, list_predictor) = serie_join(variable, lagged_predictor)
            
            if len(list_predictor) == 0:
                results[i] = None
                continue
            
            y = np.array(list_variable,ndmin=1)
            
            y = scale(y)

            X = np.array(list_predictor,ndmin=2)
            X = X.reshape((-1,1))
            
            X = scale(X)
            
            svr_rbf = SVR(kernel=kernel_variable.value)
    
            svr_rbf.fit(X, y)
            
            r_squared = svr_rbf.score(X, y)
            
            #predicted_y = list(svr_rbf.predict(original_X))
            
            
            result = {}
            
            result["r2"] = r_squared
            #result["y"] = list_variable
            #result["x"] = list_predictor
            #result["predicted_y"] = predicted_y
            
            results[i] = result

        yield (key, results)

class CorrelationSearch():
    
    context = None
    _sc = None
    _index_rdd = None
    search_timeserie_data = None
    
    def __init__(self,context):
        self.context = context
        self._sc = SparkContext(self.context["spark-cluster"], "Correlation search")
        index_file_name = self.context["correlation-index-path"]
        self._index_rdd = self._sc.textFile(index_file_name).cache()

        
    def search(self, timeserie_filename,feature_name,lag=12,kernel='linear'):
    
        timeserie_rrd = self._sc.textFile(timeserie_filename)
      
        timeserie_rrd = timeserie_rrd.map(lambda x : group_date_map(x,0,1,"%Y-%m")).reduceByKey(add).cache()
        #print timeserie_rrd.take(5)
        
        #timeserie_rrd = timeserie_rrd.reduceByKey(add)
        
        dict_timeserie = timeserie_rrd.collectAsMap()
        
        
        search_timeserie = {feature_name : dict_timeserie }
        lag_variable = self._sc.broadcast(lag)
        kernel_variable = self._sc.broadcast(kernel)

        search_timeserie_data = self._sc.broadcast(search_timeserie)
        
        search_result_rdd = self._index_rdd.flatMap(lambda value : correlation_search_map(value,search_timeserie_data,lag_variable,kernel_variable))
        search_result =  search_result_rdd.collect()
        return search_result

    def close(self):
        self._sc.stop()
        
context = {
           'spark-cluster' : "local[%s]" % cpu_count(),
           'correlation-index-path' : "/home/predictiveds/palmyr-data/correlation-search/index.txt",
           }

s = CorrelationSearch(context)
print s.search("/home/predictiveds/palmyr-data/correlation-search/test.txt","montant sinistre")
s.close()

"""
sc = SparkContext(context["spark-cluster"], "Correlation search")
test_rrd = sc.textFile("/home/predictiveds/palmyr-data/correlation-search/test.txt")
test_rrd.map(lambda line : line.replace(';',',')).saveAsTextFile("/home/predictiveds/palmyr-data/correlation-search/test2.txt")
"""