from eneuro.train import Trainer
from eneuro.nn import module
from eneuro.core import functions as fc
from eneuro.nn import loss,optim
from eneuro.data import Dataset,DataLoader


sigmoid=fc.Sigmoid()
mlp=module.MLP(10,sigmoid)
paramlist=mlp.get_params_list()
sgd=optim.SGD(mlp.get_params_list(),lr=0.01)
testtrainer=Trainer(mlp,paramlist,sgd)