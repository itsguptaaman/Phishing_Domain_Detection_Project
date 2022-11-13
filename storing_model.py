import pickle
import os


class StoreModel:

    def __init__(self):
        try:
            os.mkdir("Pickle_Models")
        except Exception as e:
            pass

    def save_model(self, name, model):
        """
        Method name: save_model.

        Description: This method is going to store the model in the form of binary using pickle.

        name: The name of the model you are going to use while saving it.

        model: The object of the model you are going to store..

        examples:
        >>> log_reg=LogisticRegression()

        >>> name="logistic_reg_model"

        >>> save_model(name,log_reg)
        """

        pickle.dump(model, open(f'Pickle_Models\{name}.pkl', 'wb'))

    def read_model(self, name):
        """
        Method name: save_model.

        Description: This method is going to read the model from the binary using pickle.

        name: The name of the model you are going to use while saving it.

        model: The object of the model you are going to store..

        examples:
        >>> name="logistic_reg_model"
        >>> logreg=read_model(name)
        """
        return pickle.load(open(f'Pickle_Models\{name}.pkl', 'rb'))

    def list_store(self, name, my_list):
        with open(f'Pickle_Models\{name}.pkl', 'wb') as data:
            pickle.dump(my_list, data)

    def list_read(self, name):

        return pickle.load(open(f'Pickle_Models\{name}.pkl', 'rb'))
