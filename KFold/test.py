
import pandas as pd
import numpy as np
import importlib.util
pyc_path = 'ISSA-XGBOOST- KFold.pyc'
spec = importlib.util.spec_from_file_location('module_name', pyc_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)



#The directory where the genotype data is located
GenePath=r"testdata\06potato2\genotypes.csv"
#The directory where the performance data is located
PhenotypePath=r"testdata\06potato2\TW.csv"
#Result output directory
outpath=r"06potato2\\"


#Only these few parameters need to be set, and the rest of the parameters adopt the default parameters
pop_size=50
max_iters=100
KFolddata=5 #Cross Validated Fold Numbers

lb=[0.1,2,0.01,0.1,0.5]
ub=[1,30,1,1,1]

X=pd.read_csv(GenePath).iloc[:,1:]  #read csv文件
y=pd.read_csv(PhenotypePath,engine='python').iloc[:,0]


module.MYISSA(p_X=X,p_y=y,p_test_size=KFolddata,p_lb=lb,p_ub=ub,p_step=int(X.shape[1]/125)+2,p_pop_size=pop_size, p_max_iters=max_iters,output_path=outpath,p_early_stop_threshold1=50)



