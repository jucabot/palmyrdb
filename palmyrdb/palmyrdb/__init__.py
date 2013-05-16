def load_from_classname(name):
    (mod_name, class_name) = name
    mod = __import__(mod_name, fromlist=[class_name])
    klass = getattr(mod, class_name)
    return klass()