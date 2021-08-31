"""Module pointing to different implementations of CFE based on different frameworks such as  PyTorch """

class CFE_prototype:
    """An interface class to different CFE implementations."""

    def __init__(self, data_interface, model_interface, method="genetic",  **kwargs):
        """Init method

        :param data_interface: an interface to access data related params.
        :param model_interface: an interface to access the output or gradients of a trained ML model.
        :param method: Name of the method to use for generating counterfactuals

        """

        self.decide_implementation_type(data_interface, model_interface, method, **kwargs)

    def decide_implementation_type(self, data_interface, model_interface, method, **kwargs):
        """Decides CFE implementation type."""

        self.__class__  = decide(model_interface, method)
        self.__init__(data_interface, model_interface, **kwargs)

# To add new implementations of DiCE, add the class in explainer_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.

def decide(model_interface, method):
    """Decides CFE implementation type."""


    if model_interface.backend == 'PYT': # PyTorch backend
        from CFE.explainer_interfaces.CFE_pytorch import CFE_PyTorch
        return CFE_PyTorch

    else: # all other backends
        backend_dice = model_interface.backend['explainer']
        module_name, class_name = backend_dice.split('.')
        module = __import__("CFE.explainer_interfaces." + module_name, fromlist=[class_name])
        return getattr(module, class_name)
