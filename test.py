import CFE
import numpy as np
from CFE.utils import helpers
backend = 'PYT'
dataset_name='adult_income'
num_cfs=4
PrototypeMode=True # or False
num_prototype_k=10
test_sample = {'age':34,
                  'workclass':'Private',
                  'education':'Bachelors',
                  'marital_status':'Single',
                  'occupation':'Service',
                  'race': 'White',
                  'gender':'Male',
                  'hours_per_week': 45}
dataset, test_dataset = eval('helpers.load_' + dataset_name + '_dataset()')
for i in range(len(dataset.columns)):
    print(np.sort(dataset.iloc[:, i].unique()))
locals()[dataset_name + '_info'], continuous_features, outcome_name = \
    eval('helpers.get_' + dataset_name + '_data_info()')
d = CFE.Data(dataframe=dataset, continuous_features=continuous_features, outcome_name=outcome_name)
ML_modelpath = eval('helpers.get_' + dataset_name + '_modelpath(backend=backend)')
m = CFE.Model(model_path=ML_modelpath, backend=backend)
exp = CFE.CFE_prototype(d, m)
model_based_train_dataset, model_based_test_dataset = eval(
    'helpers.load_' + dataset_name + '_dataset(model_based=True,model_path=ML_modelpath,train_dataset=d.get_ohe_min_max_normalized_dataset(dataset),'
                                     'test_dataset=d.get_ohe_min_max_normalized_dataset(test_dataset))')
dice_exp = exp.generate_counterfactuals(test_sample, total_CFs=num_cfs, desired_class="opposite",
                                            PrototypeMode=PrototypeMode, num_prototype_k=num_prototype_k,
                                        model_based_train_dataset=model_based_train_dataset)
dice_exp.visualize_as_dataframe()


