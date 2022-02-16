import numpy as np
from numpy.lib.recfunctions import repack_fields
import pickle
import warnings
import os

from .data.pins import branch_pins, polin_pins
from .plotting.plot import generate_plot


_sep = os.path.sep
_model_path = os.path.join(os.path.dirname(__file__), f'models{_sep}')


_model_dict = {
    'p5_p6': {'fields': ['p6', 'p5'],
              'n_components': 4},
    'v_p5_p6': {'fields': ['v', 'p5', 'p6'],
                'n_components': 4},
    'm_p5_p6': {'fields': ['m', 'p5', 'p6'],
                'n_components': 4},
    'm_v_p5_p6': {'fields': ['v', 'm', 'p6', 'p5'],
                  'n_components': 4},
    'm_v': {'fields': ['v', 'm'],
            'n_components': 3},
    'm_v_p5': {'fields': ['v', 'm', 'p5'],
               'n_components': 3}
}

_property_fields = ('M_B', 'M_B_err',
                    'vsi', 'vsi_err',
                    'pew_5972', 'pew_5972_err',
                    'pew_6355', 'pew_6355_err')

# TODO: Setup single data attribute for all given properties, and in
#       `_get_ordered_input()`, return specific views, so that numerous copies
#       are not created.
# TODO: Allow more control of plot parameters via args, kwargs


class GMM:
    """GMM object used for group membership prediction.

    Attributes
    ----------
    data : dict
        Structured array or dictionary (anything with keys) corresponding to
        observables. See `_property_fields` above for allowed keys. These
        key/value pairs may also be given as kwargs instead of a single
        structured array/dict.
    model : str
        Model used for prediction. If none is given, a default model is
        chosen. See `_model_dict` above for available model keys.
    """

    def __init__(self, data=None, model=None, *args, **kwargs):
        if data is not None:
            self._data = data
        else:
            self._data = {}
            for field in _property_fields:
                if field not in kwargs:
                    self._data[field] = None
                    continue
                self._data[field] = np.array(kwargs[field])

        self._model = model
        self._n_components = None

    def predict(self, model=None, verbose=True):
        """Predict group membership at given points.

        Parameters
        ----------
        model : str
            Model used for prediction. If none is given, a default model is
            chosen. See `_model_dict` above for available model keys.

        Returns
        -------
        prob : numpy.ndarray, shape (N, n_components)
            Sorted probabilities of being in each group.
            If using an n=3 model, returns in the order Main, Fast, Dim.
            If using an n=4 model, returns in the order CN, SS, BL, CL.
        """
        if model is None:
            model = self.model

        if verbose:
            print(f'Predicting with model {model}')

        # Predict at given data with a certain model and then reorder
        gmm = self.load_model(model)
        prob = self._predict(model, gmm)
        prob = self._reorder_prob(prob, model, gmm)

        return prob

    def get_group_name(self, probability):
        arg_to_name = ('Core-normal', 'Shallow-silicon', 'Broad-line', 'Cool')

        max_ind = np.argmax(probability)
        name = arg_to_name[max_ind]

        return name, probability[max_ind]

    def plot(self, *args, **kwargs):
        return generate_plot(self, *args, **kwargs)

    def load_model(self, model=None):
        """Load the pickled GMM."""
        if model is None:
            model = self.model

        fn = os.path.join(_model_path, f'{self.model}.p')
        with open(fn, 'rb') as file, warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            gmm = pickle.load(file)

        return gmm

    @property
    def model(self):
        if self._model is None:
            self._model = self._default_model()

        return self._model

    @property
    def n_components(self):
        if self._n_components is None:
            self._n_components = _model_dict[self.model]['n_components']

        return self._n_components

    @property
    def pew_5972(self):
        return self._data['pew_5972']

    @property
    def pew_5972_err(self):
        return self._data['pew_5972_err']

    @property
    def pew_6355(self):
        return self._data['pew_6355']

    @property
    def pew_6355_err(self):
        return self._data['pew_6355_err']

    @property
    def M_B(self):
        return self._data['M_B']

    @property
    def M_B_err(self):
        return self._data['M_B_err']

    @property
    def vsi(self):
        return self._data['vsi']

    @property
    def vsi_err(self):
        return self._data['vsi_err']

    def _default_model(self):
        """Detect which model to use based on given inputs.

        Returns
        -------
        model : str
            Model used for prediction.
        """
        is_p5 = self.pew_5972 is not None
        is_p6 = self.pew_6355 is not None
        is_m = self.M_B is not None
        is_v = self.vsi is not None

        if is_p5 and is_p6:
            if not is_m and is_v:
                model = 'v_p5_p6'
            elif is_m and not is_v:
                model = 'm_p5_p6'
            elif is_m and is_v:
                model = 'm_v_p5_p6'
            else:
                model = 'p5_p6'
        elif is_m and is_v:
            if is_p5:
                model = 'm_v_p5'
            else:
                model = 'm_v'
        else:
            raise RuntimeError('Could not determine which model to use.')

        return model

    def _predict(self, model, gmm):
        """Predict at input points using the given GMM object.

        Parameters
        ----------
        model : str
            Model used for prediction.
        gmm : sklearn.mixture.GaussianMixture
            GMM object of model used for prediction.

        Returns
        -------
        prob : numpy.ndarray, shape (N, n_components)
            Unsorted probabilities output by GMM prediction.
        """
        pred_data = self._get_ordered_input(model)

        prob = gmm.predict_proba(pred_data)

        return prob

    def _get_ordered_input(self, model):
        """Get array suitable for input for the GMM objects.

        This unfortunately must be done case-by-case because the GMM objects
        were generated with different ordering of the parameters involved.

        Parameters
        ----------
        model : str
            Model used for prediction.

        Returns
        -------
        pred_data : numpy.ndarray, shape (N, n_dims)
            Ordered input data fit to pass into `gmm.predict_proba()`.
        """
        if model == 'm_v':
            pred_data = np.array((self.vsi, self.M_B)).T
        elif model == 'p5_p6':
            pred_data = np.array((self.pew_6355, self.pew_5972)).T
        elif model == 'm_v_p5':
            pred_data = np.array((self.vsi, self.M_B, self.pew_5972)).T
        elif model == 'v_p5_p6':
            pred_data = np.array((self.vsi, self.pew_5972, self.pew_6355)).T
        elif model == 'm_p5_p6':
            pred_data = np.array((self.M_B, self.pew_5972, self.pew_6355)).T
        elif model == 'm_v_p5_p6':
            pred_data = np.array((self.vsi, self.M_B, self.pew_6355, self.pew_5972)).T

        return pred_data

    def _reorder_prob(self, prob, model, gmm):
        """Reorder probabilities to have consistent output.

        Parameters
        ----------
        prob : numpy.ndarray, shape (N, n_components)
            Sample at which to predict.
        model : str
            Model used for prediction.
        gmm : sklearn.mixture.GaussianMixture
            GMM object of model used for prediction.

        Returns
        -------
        prob : numpy.ndarray, shape (N, n_components)
            Unsorted probabilities output by GMM prediction.
        """
        if self.n_components == 3:
            pins = polin_pins
        elif self.n_components == 4:
            pins = branch_pins

        fields = _model_dict[model]['fields']
        pin_data = pins[fields]

        # sklearn.gmm can't take in structured arrays, so this converts
        # to a normal unstructured array
        arr = repack_fields(pin_data).view((np.float64, len(fields)))
        pin_prob = gmm.predict_proba(arr)

        ordered_indices = [np.argmax(p) for p in pin_prob]

        # Check for duplicates
        if len(set(ordered_indices)) != self.n_components:
            print(f'{model} probabilities were not reordered')
            return prob

        prob[:, list(range(self.n_components))] = prob[:, ordered_indices]

        return prob
