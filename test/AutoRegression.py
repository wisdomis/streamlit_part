import pandas as pd
import numpy as np
import pycaret.regression

def setup(data, target, train, use_gpu, outliar):
    return pycaret.regression.setup(data, target=target, session_id=123, train_size=train, use_gpu=use_gpu, remove_outliers=outliar)

def save_df():
    results = pycaret.regression.pull()
    return results

def search_missing_value(data):
    import pandas as pd
    col = list(data.columns)
    missing_series = data.isnull().sum()
    missing_cols = []
    for i in col:
        if missing_series[i]!=0:
            missing_cols.append(i)
        else:
            missing_series = missing_series.drop(i)
    return missing_cols,missing_series

def interpolation(data,target,method): #target = missing_cols
    import pandas as pd
    for i in range(len(target)):
        data[target[i]] = data[target[i]].interpolate(method = method[i])
    return data

def compare(target_model_list):
    return pycaret.regression.compare_models(include=target_model_list)

def tune(model, opt):
    return pycaret.regression.tune_model(model, optimize=opt, choose_better=True)

def Blend(arr):
    arr[0]=pycaret.regression.create_model(arr[0])
    arr[1]=pycaret.regression.create_model(arr[1])
    arr[2]=pycaret.regression.create_model(arr[2])
    return pycaret.regression.blend_models([arr[0],arr[1],arr[2]])
def single(name):
    return pycaret.regression.create_model(name)

def single_visual(df):
    visual = df.iloc[0:9]
    return visual.plot()

def evaluate(model):
    return pycaret.regression.evaluate_model(model)

def shap(model):
    return pycaret.regression.interpret_model(model)

def prediction(model):
    return pycaret.regression.predict_model(model)

def save_model(model, name):
    return pycaret.regression.save_model(model, name)

def load(name):
    return pycaret.regression.load_model(name)