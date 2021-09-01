"""Module containing an interface to trained PyTorch model."""

from CFE.model_interfaces.base_model import BaseModel
import torch
import numpy as np
# from LC_prediction import NeuralNet


class PyTorchModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='PYT', func=None, kw_args=None):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        :param backend: "PYT" for PyTorch framework.
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the dictionary of kw_args, by default.
        """
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(model, model_path, backend)

    def load_model(self):
        if self.model_path != '':
            if self.model_path.find('adult')!=-1:
                self.model = torch.load(self.model_path)
            elif self.model_path.find('model_LC')!=-1:
                # self.model = torch.load(self.model_path)
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            # self.model= NeuralNet(data_length,32,1)
            # self.model = torch.load('model_LC.pth', map_location=torch.device('cpu'))
        if True==1: pass


    def get_output(self, input_tensor, transform_data=False):
        """returns prediction probabilities

        :param input_tensor: test input.
        :param transform_data: boolean to indicate if data transformation is required.
        """
        if transform_data:
            input_tensor = torch.tensor(self.transformer.transform(input_tensor)).float()

        return self.model(input_tensor).float()

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")

    def get_num_output_nodes(self, inp_size): #29 들어옴 ( ohe한 feature 개수가)

        # temp_input = torch.rand(1,inp_size).float().to(self.device) #torch.tensor([np.random.uniform([inp_size])]).float()
        temp_input = torch.rand(1,inp_size).float() #torch.tensor([np.random.uniform([inp_size])]).float()
        return self.get_output(temp_input).data

    def get_num_output_nodes_cpu(self, inp_size):  # 29 들어옴 ( ohe한 feature 개수가)

        temp_input = torch.rand(1, inp_size).float()
        # torch.tensor([np.random.uniform([inp_size])]).float()
        return self.get_output(temp_input).data
