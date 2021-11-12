import sys

sys.path.append("..")
import pylrpredictor
from pylrpredictor.curvemodels import MCMCCurveModel,MLCurveModel,MlCurveMixtureModel
from pylrpredictor.modelfactory import create_model
from pylrpredictor.ensemblecurvemodel import CurveModelEnsemble
from pylrpredictor.curvefunctions import all_models, curve_combination_models, curve_ensemble_models, model_defaults

import csv

from time import time
import math
import numpy as np

def get_history_x(count):
    bath = "C:\\Users\\t-meli\\Documents\\auto-tuning\\codes\\AdaTune\\simulator\\records\\"
    model_name = 'vgg-cpu'
    task_no = 0
    f = csv.reader(open(bath + model_name+ "\\" + str(task_no) + ".csv"))
    res = []
    i = 1
    for l in f:
        if i>count:
            break
        # res.append(int(float(l[2])))
        val = float(l[2])
        # val = math.sqrt(val)
        res.append(val)
        i += 1
    return res

def test():
    # mcmc = MCMCCurveModel()
    # history_x = np.arange(1, 100)
    count = 80
    history_x = [i for i in range(1,count+1)]
    history_y = get_history_x(count)
    history_x = np.array(history_x)
    history_y = np.array(history_y)
    # print(history_y)
    # # Exponential moving average
    # from scipy.ndimage import gaussian_filter
    # history_y = gaussian_filter(history_y,1)
    # print(history_y)
    # return
    ss = time()
    mymodel = create_model('curve_combination', 100, nthreads=11)
    mm = time()
    mymodel.fit(history_x, history_y)
    ff = time()
    pre = mymodel.predict(count+1)
    ee = time()
    
    predictions = []
    for i in range(count+1, count+11):
        print("at x=%d, pre:%f"%(i, mymodel.predict(i)))
    
    print(pre)
    print("Create:%.2f, fit:%.2f, predict:%.2f"%(mm-ss, ff-mm, ee-ff))

def create_mixture():
    ensemble_models = []
    for model_name in curve_ensemble_models:
        if model_name == "linear":
            m = LinearCurveModel()
        else:
            if model_name in model_defaults:
                m = MLCurveModel(function=all_models[model_name],
                  default_vals=model_defaults[model_name],
                  recency_weighting=False)
            else:
                m = MLCurveModel(function=all_models[model_name],
                  recency_weighting=False)
        ensemble_models.append(m)
    return ensemble_models

def test_ensemble():
    count = 80
    history_x = [i for i in range(1,count+1)]
    history_y = get_history_x(count)
    history_x = np.array(history_x)
    history_y = np.array(history_y)
    ss = time()
    mixture = create_mixture()
    # curve = MlCurveMixtureModel(mixture)
    curve = CurveModelEnsemble(mixture)
    mm = time()
    curve.fit(history_x, history_y)
    ff = time()
    pre = curve.predict(count+1)
    ee = time()
    
    predictions = []
    for i in range(count+1, count+11):
        print("at x=%d, pre:%f"%(i, curve.predict(i)))  

def test_curveModel(func):
    predictions = []
    for count in [10,20,40,80]:
        history_x = [i for i in range(1,count+1)]
        history_y = get_history_x(count)
        history_x = np.array(history_x)
        history_y = np.array(history_y)
        ss = time()
        curve = MLCurveModel(function=all_models[func])
        mm = time()
        curve.fit(history_x, history_y)
        ff = time()
        pre = curve.predict(count+1)
        ee = time()
        for i in range(count+1, count+11):
            print("at x=%d, pre:%f"%(i, curve.predict(i)))
            predictions.append([func, i, curve.predict(i)])
    return predictions
    
if __name__ == "__main__":
    # all_models = all_models[8:]
    # i = 0
    # for mname in all_models.keys():
    #     if i!=7:
    #         i += 1
    #         continue
    #     i += 1
    #     print(mname)
    #     csvfile = open('curveSinglePredictions_%s.csv'%mname, 'w', newline='')
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['iters','predictions'])
    #     predicts = test_curveModel(mname)
    #     writer.writerows(predicts)
    #     csvfile.close()
    # test()
    a = 1
    print(a)
    test_ensemble()