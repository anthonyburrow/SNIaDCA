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
    'p5_p6': {'fields': ['pew_6355', 'pew_5972'],
              'n_components': 4},
    'v_p5_p6': {'fields': ['vsi', 'pew_5972', 'pew_6355'],
                'n_components': 4},
    'm_p5_p6': {'fields': ['M_B', 'pew_5972', 'pew_6355'],
                'n_components': 4},
    'm_v_p5_p6': {'fields': ['vsi', 'M_B', 'pew_6355', 'pew_5972'],
                  'n_components': 4},
    'm_v': {'fields': ['vsi', 'M_B'],
            'n_components': 3},
    'm_v_p5': {'fields': ['vsi', 'M_B', 'pew_5972'],
               'n_components': 3}
}

_property_fields = ('M_B', 'M_B_err',
                    'vsi', 'vsi_err',
                    'pew_5972', 'pew_5972_err',
                    'pew_6355', 'pew_6355_err')


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
        prob = self._predict(self._data, model, gmm)
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

    def __getattr__(self, name):
        try:
            if name not in _property_fields:
                raise KeyError
            return self._data[name]
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None

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

    def _predict(self, data, model, gmm):
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
        fields = _model_dict[model]['fields']

        # sklearn.gmm can't take in structured arrays, so this converts
        # to a normal unstructured array
        if isinstance(data, dict):
            arr = np.zeros((len(data[fields[0]]), len(fields)))
            for i, field in enumerate(fields):
                arr[:, i] = data[field]
        elif isinstance(data, np.ndarray):
            arr = repack_fields(data[fields]).view((np.float64, len(fields)))
        else:
            print('Unable to convert data array to numpy.ndarray')

        prob = gmm.predict_proba(arr)

        return prob

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

        pin_prob = self._predict(pins, model, gmm)

        ordered_indices = [np.argmax(p) for p in pin_prob]

        # Check for duplicates
        if len(set(ordered_indices)) != self.n_components:
            print(f'{model} probabilities were not reordered')
            return prob

        prob[:, list(range(self.n_components))] = prob[:, ordered_indices]

        return prob
