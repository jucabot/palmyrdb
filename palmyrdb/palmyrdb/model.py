        
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