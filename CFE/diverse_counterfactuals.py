import numpy as np
import pandas as pd
import copy
from IPython.display import display
import os.path

class CounterfactualExamples:
    """A class to store and visualize the resulting counterfactual explanations."""
    def __init__(self, data_interface=None, final_cfs_df=None, test_instance_df=None, prototype_df=None,test_instance_dfs=None, final_cfs_df_sparse=None, posthoc_sparsity_param=0, desired_range=None, desired_class="opposite", model_type='classifier',num_total_CFs = None,feature_weights_list = None):

        self.data_interface = data_interface
        self.final_cfs_df = final_cfs_df
        self.test_instance_df = test_instance_df
        self.prototype_df = prototype_df
        self.test_instance_dfs = test_instance_dfs
        self.final_cfs_df_sparse = final_cfs_df_sparse
        self.num_total_CFs=num_total_CFs
        self.final_cfs_list = None
        self.posthoc_sparsity_param = posthoc_sparsity_param # might be useful for future additions
        self.feature_weights_list= feature_weights_list
        self.test_pred = self.test_instance_df[self.data_interface.outcome_name].iloc[0]
        if model_type == 'classifier':
            if desired_class == "opposite":
                self.new_outcome = 1.0 - round(self.test_pred)
            else:
                self.new_outcome = desired_class
        elif model_type == 'regressor':
            self.new_outcome = desired_range

    def visualize_new_metric(self,outcome_name):
        display(self.mixed_const_result[self.test_instance_dfs.columns.drop(outcome_name)])

    def filling_empty_DataFrame(self,df):
            if len(df)!=self.num_total_CFs :
                for i in range(self.num_total_CFs-len(df)):
                    df=df.append(pd.Series(), ignore_index=True)
            return df

    def visualize_as_dataframe(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        display(self.test_instance_df) #  works only in Jupyter notebook # ??
        print('\n')
        if isinstance(self.prototype_df,pd.DataFrame):
            prototype_pred = self.prototype_df[self.data_interface.outcome_name].iloc[0]
            print('Prototype instance (Prototype outcome : %i)' % round(prototype_pred))
            display(self.prototype_df)  # works only in Jupyter notebook # ??
            print('\n')
        if len(self.final_cfs_df) > 0:
            if self.posthoc_sparsity_param == None:
                print('\nCounterfactual set (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is not None:
                # CFs
                print('\nDiverse Counterfactual set (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df_sparse, show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is None:
                print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.display_df(self.final_cfs_df, show_only_changes)

            elif 'data_df' not in self.data_interface.__dict__: # for private data !
                print('\nDiverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)

            else:
                # CFs
                print('\nDiverse Counterfactual set without sparsity correction (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)
        else:
            print('\nNo counterfactuals found!')
            self.final_cfs_df_sparse = self.filling_empty_DataFrame(self.final_cfs_df_sparse)
            self.display_df(self.final_cfs_df_sparse, show_only_changes)

    def display_df(self, df, show_only_changes):
        if show_only_changes is False:
            display(df)  # works only in Jupyter notebook
        else:
            newdf = df.values.tolist()
            org = self.test_instance_df.values.tolist()[0]
            for ix in range(df.shape[0]):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

    def visualize_as_list(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        print(self.test_instance_df.values.tolist()[0])

        if len(self.final_cfs) > 0:
            if self.posthoc_sparsity_param == None:
                print('\nCounterfactual set (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is not None:
                # CFs
                print('\nDiverse Counterfactual set (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df_sparse.values.tolist(), show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is None:
                print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            elif 'data_df' not in self.data_interface.__dict__: # for private data
                print('\nDiverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            else:
                # CFs
                print('\nDiverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)
        else:
            print('\n0 counterfactuals found!')

    def print_list(self, li, show_only_changes):
        if show_only_changes is False:
            for ix in range(len(li)):
                print(li[ix])
        else:
            newli = copy.deepcopy(li)
            org = self.test_instance_df.values.tolist()[0]
            for ix in range(len(newli)):
                for jx in range(len(newli[ix])):
                    if newli[ix][jx] == org[jx]:
                        newli[ix][jx] = '-'
                print(newli[ix])

    def to_json(self):
        if self.final_cfs_sparse is not None:
            df = self.final_cfs_df_sparse
        else:
            df = self.final_cfs_df
        return df.to_json()
