"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
from CFE.explainer_interfaces.explainer_base import ExplainerBase
import torch
import numpy as np
import random
import timeit
import copy
import pandas as pd
from CFE import diverse_counterfactuals as exp

class CFE_PyTorch(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method
        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        self.device = torch.device( 'cpu')
        # initiating data related parameters
        super().__init__(data_interface)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()
        self.normalized_dataset=self.data_interface.get_ohe_min_max_normalized_dataset(self.data_interface.data_df)
        # initializing model related variables
        self.model = model_interface
        self.model.load_model() # loading trained model
        ev = self.model.set_eval_mode() # set the model in evaluation mode
        if self.model.transformer.func is not None: # TODO: this error is probably too big - need to change it.
            raise ValueError("Gradient-based DiCE currently (1) accepts the data only in raw categorical and continuous formats, (2) does one-hot-encoding and min-max-normalization internally, (3) expects the ML model the accept the data in this same format. If your problem supports this, please initialize model class again with no custom transformation function.")
        self.num_output_nodes = self.model.get_num_output_nodes_cpu(len(self.data_interface.ohe_encoded_feature_names)).shape[1] # number of output nodes of ML model
        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.metric_names = ['validity', 'continuous_proximity', 'categorical_proximity', 'continuous_sparsity',
                        'continuous_diversity', 'continuous_count_diversity', 'categorical_diversity']
        self.count0=0
        self.count1=0
        self.count_all=0

    def generate_counterfactuals(self, query_instance, total_CFs,PrototypeMode=False,  desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0,prototype_weight=0.2,categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all", permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear",num_prototype_k=10,model_based_train_dataset=None):
        """Generates diverse counterfactual explanations
        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".
        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).
        """
        self.num_total_CFs=total_CFs
        self.PrototypeMode=PrototypeMode
        self.num_prototype_k=num_prototype_k
        self.model_based_train_dataset = model_based_train_dataset
        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)
        # check permitted range for continuous features
        if permitted_range is not None:
            if not self.data_interface.check_features_range(permitted_range):
                raise ValueError(
                    "permitted range of features should be within their original range")
            else:
                self.data_interface.permitted_range = permitted_range
                self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
                self.cont_minx = []
                self.cont_maxx = []
                for feature in self.data_interface.continuous_feature_names:
                    self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                    self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if([yloss_type, diversity_loss_type, feature_weights] != self.loss_weights):
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if([proximity_weight, diversity_weight,prototype_weight] != self.hyperparameters):
            self.update_hyperparameters(proximity_weight, diversity_weight,prototype_weight,categorical_penalty)

        final_cfs_df, test_instance_df,prototype_df,test_instance_dfs, final_cfs_df_sparse = self.find_counterfactuals(query_instance,desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm)

        return exp.CounterfactualExamples(data_interface = self.data_interface,
                                          final_cfs_df = final_cfs_df,
                                          test_instance_df = test_instance_df,
                                          prototype_df = prototype_df,
                                          final_cfs_df_sparse = final_cfs_df_sparse,
                                          posthoc_sparsity_param = posthoc_sparsity_param,
                                          desired_class=desired_class,
                                          num_total_CFs = self.num_total_CFs,
                                          feature_weights_list=self.feature_weights_list,
                                          )

    def get_test_instances(self,query_instances):
        test_instance_dfs=pd.DataFrame(query_instances[0])
        for i in range(len(query_instances)-1):
            test_instance_dfs=pd.concat([test_instance_dfs,pd.DataFrame(query_instances[i+1])])
        return test_instance_dfs

    def get_model_output(self, input_instance):
        """get output probability of ML model"""
        return self.model.get_output(input_instance)[(self.num_output_nodes-1):]

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            # input_instance = torch.tensor(input_instance).float().to(self.device)
            input_instance = torch.tensor(input_instance).float()
        # return self.get_model_output(input_instance).data.cpu().numpy()
        return self.get_model_output(input_instance).data.numpy()

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance = self.data_interface.get_ohe_min_max_normalized_data(input_instance).iloc[0].values
        # return self.predict_fn(torch.tensor(input_instance).float().to(self.device))
        return self.predict_fn(torch.tensor(input_instance).float())

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
            self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = []
                for jx in range(self.minx.shape[1]):
                    one_init.append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(torch.tensor(one_init).float())
                self.cfs[ix].requires_grad = True

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)
            feature_weights_list = []
            for index, feature in enumerate(self.data_interface.ohe_encoded_feature_names):
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = torch.tensor(feature_weights_list)

        # define different parts of loss function
        self.yloss_opt = torch.nn.BCEWithLogitsLoss()

    def update_hyperparameters(self, proximity_weight, diversity_weight,prototype_weight,categorical_penalty):
        """Update hyperparameters of the loss function"""
        self.hyperparameters = [proximity_weight, diversity_weight, prototype_weight]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.prototype_weight = prototype_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based PyTorch optimizers."""
        opt_library = optimizer.split(':')[0]
        opt_method = optimizer.split(':')[1]
        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        for i in range(self.total_CFs):
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow((self.get_model_output(self.cfs[i]) - self.target_cf_class), 2)[0]
            elif self.yloss_type == "log_loss":
                temp_logits = torch.log((abs(self.get_model_output(self.cfs[i]) - 0.000001))/(1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.BCEWithLogitsLoss()
                temp_loss = criterion(temp_logits, torch.tensor([self.target_cf_class]))
            elif self.yloss_type == "hinge_loss":
                temp_logits = torch.log((abs(self.get_model_output(self.cfs[i]) - 0.000001))/(1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.ReLU()
                all_ones = torch.ones_like(self.target_cf_class)
                labels = 2 * self.target_cf_class - all_ones
                temp_loss = all_ones - torch.mul(labels, temp_logits)
                temp_loss = torch.norm(criterion(temp_loss))
            yloss +=  temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted constraint distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), torch.abs(self.feature_weights_list)), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs)) # 굳이 안나눠줘도 되지 않나?

    def compute_prototype_loss(self):
        """Computes the fourth part (constraint) of the loss function."""
        prototype_loss = 0.0
        for i in range(self.total_CFs):
            prototype_loss += self.compute_dist(self.cfs[i], self.prototype)
        return prototype_loss / (torch.mul(len(self.minx[0]), self.total_CFs))  # 굳이 안나눠줘도 되지 않나?

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i,j)] += 0.0001 # ???

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i,j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)


    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += torch.pow((torch.sum(self.cfs[i][v]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else torch.tensor(0.0)
        self.regularization_loss = self.compute_regularization_loss()
        self.prototype_loss =  self.compute_prototype_loss() if self.prototype_weight > 0 and self.PrototypeMode else torch.tensor(0.0)

        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss)\
                   - (self.diversity_weight * self.diversity_loss) \
                    + (self.prototype_weight * self.prototype_loss)\
                    + (self.categorical_penalty * self.regularization_loss)  # categorical penalty = 0.1
        # return torch.max(self.loss,torch.tensor(0)) #
        return self.loss #

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cfs[n].data[i] = query_instance[i]+(n*0.01)
                    else:
                        self.cfs[n].data[i] = np.random.uniform(self.minx[0][i], self.maxx[0][i])
                else:
                    self.cfs[n].data[i] = query_instance[i]

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.cpu().detach().clone().numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                org_cont = (cf[v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i] # continuous feature in orginal scale
                org_cont = round(org_cont, self.cont_precisions[i]) # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[v] = normalized_cont # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if(len(maxs) > 1):
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        try:
                            ix = maxs[0]
                        except:
                            ix= None
                else:
                    try:
                        ix = maxs[0]
                    except:
                        ix = None
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                for jx in range(len(cf)):
                    self.cfs[index].data[jx] = torch.tensor(temp_cfs[index])[jx]

        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if((self.project_iter > 0) & (itr > 0)):
            if((itr % self.project_iter) == 0):
                self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold) # ???
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                temp_cfs = self.round_off_cfs(assign=False)
                test_preds = [self.predict_fn(cf)[0] for cf in temp_cfs]

                if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False

    def get_prototype(self,normalized_query_instance):
        normalized_dataset=self.model_based_train_dataset
        normalized_query_instance=torch.tensor(normalized_query_instance)
        dataset = torch.tensor(normalized_dataset[
                                   normalized_dataset[self.data_interface.outcome_name] == np.array(
                                       self.target_cf_class)].drop(self.data_interface.outcome_name, axis=1).values)
        num_prototype_k = self.num_prototype_k
        if self.PrototypeMode:
            distances =  dataset - normalized_query_instance
            prototype = torch.mean(dataset[torch.sort(torch.sum(torch.mul(torch.abs(distances), torch.abs(self.feature_weights_list)), dim=1))[1][:num_prototype_k]],axis=0)
        return prototype

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Finds counterfactuals by graident-descent."""

        normalized_query_instance = self.data_interface.get_ohe_min_max_normalized_data(query_instance).iloc[0].values
        self.x1 = torch.tensor(normalized_query_instance)

        # find the predicted value of normalized_query_instance
        test_pred = self.predict_fn(torch.tensor(normalized_query_instance).float())[0]

        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)

        self.target_cf_class = torch.tensor(desired_class).float()

        self.prototype=self.get_prototype(normalized_query_instance) if self.PrototypeMode else None

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process - if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100]*loop_find_CFs

        for loop_ix in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                self.initialize_CFs(normalized_query_instance, False)
            else:
                self.initialize_CFs(normalized_query_instance, init_near_query_instance)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate) # adam


            while self.stop_loop(iterations, loss_diff) is False: #

                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()
                # get loss and backpropogate
                loss_value = self.compute_loss()
                self.loss.backward()

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                # update the variables
                self.optimizer.step()

                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

                if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean([abs(pred[0]-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                        self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                        for ix in range(self.total_CFs):
                            self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                            self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].cpu().detach().clone().numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations # iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        if((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for ix in range(self.total_CFs):
                        self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                        self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])

        # convert to the format that is consistent with dice_tensorflow
        normalized_query_instance = np.array([normalized_query_instance], dtype=np.float32) # float64->float32
        for tix in range(max(loop_find_CFs, self.total_CFs)):
            self.final_cfs[tix] = np.array([self.final_cfs[tix]], dtype=np.float32)
            self.cfs_preds[tix] = np.array([self.cfs_preds[tix]], dtype=np.float32)

            if isinstance(self.best_backup_cfs[0], np.ndarray): # checking if CFs are backed
                self.best_backup_cfs[tix] = np.array([self.best_backup_cfs[tix]], dtype=np.float32)
                self.best_backup_cfs_preds[tix] = np.array([self.best_backup_cfs_preds[tix]], dtype=np.float32)

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])
        final_cfs_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(cfs)
        cfs_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds]
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        test_instance_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(normalized_query_instance)
        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred, 3))
        if isinstance(self.prototype,torch.Tensor):
            prototype_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(np.expand_dims(self.prototype.numpy(),axis=0))
            prototype_pred = self.predict_fn(self.prototype.float())
            prototype_df[self.data_interface.outcome_name] = np.array(np.round(prototype_pred, 3))
        else:
            prototype_df=None
        test_instance_dfs=[]

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse, test_instance_df, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            final_cfs_df_sparse = None

        m, s = divmod(self.elapsed, 60)
        if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))] # indexes of valid CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = [] # indexes of valid CFs
            for cf_ix, pred in enumerate(self.cfs_preds):
                if((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0 :
                print('No Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...', '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None: final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)
        final_cfs_df=final_cfs_df.iloc[valid_ix].reset_index(drop=True)

        #validity
        valid_cfs = list(set(map(tuple, final_cfs_df_sparse.values.tolist())))
        num_valid_cfs = len(valid_cfs)

        if num_valid_cfs != 0:
            for i in range(num_valid_cfs):
                valid_cfs[i]=np.array(self.data_interface.get_ohe_min_max_normalized_data(list(valid_cfs[i])[:-1]).iloc[0],dtype=np.float32)
            total_cfs = self.num_total_CFs
            validity = num_valid_cfs / total_cfs

            # continuous_proximity
            dist_cont = 0
            for i in range(num_valid_cfs):
                for j in self.encoded_continuous_feature_indexes:
                    dist_cont += np.abs(valid_cfs[i][j] - normalized_query_instance[0][j])*np.array(self.feature_weights_list[j],dtype=np.float32)
            continuous_proximity=-dist_cont/(len(self.encoded_continuous_feature_indexes)*num_valid_cfs) # continuous는 사실 numerical? prototype 논문에 그렇게 나와있을걸?

            # categorical_proximity
            dist_cat = 0
            for i in range(num_valid_cfs):
                for subindexes in self.encoded_categorical_feature_indexes:
                    dist_cat += not all(valid_cfs[i][subindexes] == normalized_query_instance[0][subindexes])
            categorical_proximity=1-dist_cat/(len(self.encoded_categorical_feature_indexes)*num_valid_cfs)

            # sparsity
            change_count = 0
            for i in range(num_valid_cfs):
                for j in self.encoded_continuous_feature_indexes:
                    change_count += not valid_cfs[i][j] == normalized_query_instance[0][j]
            continuous_sparsity = 1-change_count / (len(self.encoded_continuous_feature_indexes) * num_valid_cfs)

            if num_valid_cfs != 1:
                # continuous_diversity
                dist_cont = 0
                for i in range(num_valid_cfs-1):
                    for j in range(i+1,num_valid_cfs):
                        for k in self.encoded_continuous_feature_indexes:
                                dist_cont += np.abs(valid_cfs[i][k] - valid_cfs[j][k]) * np.array(
                                    self.feature_weights_list[k], dtype=np.float32)
                continuous_diversity= dist_cont / (len(self.encoded_continuous_feature_indexes) * num_valid_cfs*(num_valid_cfs-1)/2)

                # continuous_count_diversity
                dist_cont = 0
                for i in range(num_valid_cfs - 1):
                    for j in range(i + 1, num_valid_cfs):
                        for k in self.encoded_continuous_feature_indexes:
                            dist_cont += not valid_cfs[i][k]== valid_cfs[j][k]
                continuous_count_diversity = dist_cont / (len(self.encoded_continuous_feature_indexes) * num_valid_cfs * (num_valid_cfs - 1) / 2)

                # categorical_diversity
                dist_cat = 0
                for i in range(num_valid_cfs - 1):
                    for j in range(i + 1, num_valid_cfs):
                        for subindexes in self.encoded_categorical_feature_indexes:
                            dist_cat += not all(valid_cfs[i][subindexes] == valid_cfs[j][subindexes])
                categorical_diversity = dist_cat / (len(self.encoded_categorical_feature_indexes) * num_valid_cfs*(num_valid_cfs-1)/2)

            else :
                self.count1+=1; continuous_diversity=0; continuous_count_diversity=0; categorical_diversity=0;
        else :
            self.count0+=1; validity=0; continuous_proximity=0; categorical_proximity=0; continuous_sparsity=0; continuous_diversity=0; continuous_count_diversity=0; categorical_diversity=0; cont_const_dist=0; cat_const_dist=0

        self.count_all+=1
        self.metric_names_dict = {}
        for i in self.metric_names:
            self.metric_names_dict[i] = eval(i)
            print(i+f':{self.metric_names_dict[i]:0.2f}',end=" ")
        print('\n')

        return final_cfs_df, test_instance_df,prototype_df,test_instance_dfs, final_cfs_df_sparse # returning only valid CFs #
