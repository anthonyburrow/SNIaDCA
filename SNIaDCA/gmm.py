import numpy as np
import pickle
from numpy.lib.recfunctions import repack_fields

import os


__model_path = os.path.join(os.path.dirname(__file__), 'models/')


_dt = [('m', np.float64), ('v', np.float64),
       ('p5', np.float64), ('p6', np.float64)]

# Order: core_normal, shallow_si, broad_line, cool
_branch_pins = np.array(
    [(-19.35, 11.8, 23, 105), (-19.55, 10.5, 10, 62),
     (-19.4, 14, 15, 149), (-18.5, 11.25, 53, 125)],
    dtype=_dt)

# Order: main, fast, dim
_polin_pins = np.array(
    [(-19.5, 11.3, 16, 90), (-19.2, 14, 15, 150), (-18.6, 10.7, 55, 125)],
    dtype=_dt)


_models = {
    'm_v': {'fields': ['v', 'm'], 'n_components': 3},
    'p5_p6': {'fields': ['p6', 'p5'], 'n_components': 4},
    'm_v_p5': {'fields': ['v', 'm', 'p5'], 'n_components': 3},
    'v_p5_p6': {'fields': ['v', 'p5', 'p6'], 'n_components': 4},
    'm_p5_p6': {'fields': ['m', 'p5', 'p6'], 'n_components': 4},
    'm_v_p5_p6': {'fields': ['v', 'm', 'p6', 'p5'], 'n_components': 4},
}


def get_gmm(model):
    """Load the pickled GMM."""
    fn = os.path.join(__model_path, f'{model}.p')
    with open(fn, 'rb') as F:
        gmm = pickle.load(F)
    return gmm


class gmm:
    """GMM object used for group membership prediction.

    Attributes
    ----------
    p5 : numpy.ndarray, shape (N, )
        Si II 5972 pEW.
    p6 : numpy.ndarray, shape (N, )
        Si II 6355 pEW.
    m : numpy.ndarray, shape (N, )
        Maximum B magnitude.
    v : numpy.ndarray, shape (N, )
        Si II 6355 velocity.
    """

    def __init__(self, p5=None, p6=None, m=None, v=None):
        self.p5 = p5
        self.p6 = p6
        self.m = m
        self.v = v

    def predict_group(self, model=None):
        """Predict group membership at given points.

        Parameters
        ----------
        pred_data : numpy.ndarray, shape (N, n_dims)
            Sample at which to predict.
        model : str
            Model used for prediction.

        Returns
        -------
        prob : numpy.ndarray, shape (N, n_components)
            Sorted probabilities of being in each group.
            If using an n=3 model, returns in the order Main, Fast, Dim.
            If using an n=4 model, returns in the order CN, SS, BL, CL.
        """
        if model is None:
            model = self._default_model()

        # predict at `pred_data` and reorder
        gmm = get_gmm(model)
        prob = self._predict(model, gmm)
        prob = self._reorder_prob(prob, model, gmm)

        return prob

    def _default_model(self):
        """Detect which model to use based on given inputs.

        Returns
        -------
        model : str
            Model used for prediction.
        """
        is_p5 = self.p5 is not None
        is_p6 = self.p6 is not None
        is_m = self.m is not None
        is_v = self.v is not None

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
            m = ('Could not determine which model to use; specify model in '
                 '`predict_branch()` argument')
            print(m)
            return

        print(f'Predicting with model {model}')
        return model

    def _predict(self, model, gmm):
        """Predict at input points.

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

        """
        # FYI: For structured arrays, use instead something like:

        fields = _models[model]['fields']
        try:
            arr = pred_data[fields].copy().view((float, len(fields)))
            prob = gmm.predict_proba(arr)
        except ValueError:
            arr = repack_fields(pred_data[fields]).view((float, len(fields)))
            prob = gmm.predict_proba(arr)
        """

        return prob

    def _get_ordered_input(self, model):
        """Get `pred_data` array.

        `pred_data` needs to match the order of the respective GMM, and I kept
        changing the order of GMM parameters, so I just did a case-by-case...

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
            pred_data = np.array((self.v, self.m)).T
        elif model == 'p5_p6':
            pred_data = np.array((self.p6, self.p5)).T
        elif model == 'm_v_p5':
            pred_data = np.array((self.v, self.m, self.p5)).T
        elif model == 'v_p5_p6':
            pred_data = np.array((self.v, self.p5, self.p6)).T
        elif model == 'm_p5_p6':
            pred_data = np.array((self.m, self.p5, self.p6)).T
        elif model == 'm_v_p5_p6':
            pred_data = np.array((self.v, self.m, self.p6, self.p5)).T

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
        gmm_p : numpy.ndarray, shape (N, n_components)
            Unsorted probabilities output by GMM prediction.
        """
        n_components = _models[model]['n_components']
        if n_components == 3:
            pins = _polin_pins
        elif n_components == 4:
            pins = _branch_pins

        fields = _models[model]['fields']
        pin_data = pins[fields]

        #  sklearn.gmm can't take in structured arrays, so a workaround...
        try:
            arr = pin_data[fields].copy().view((float, len(fields)))
            pin_prob = gmm.predict_proba(arr)
        except ValueError:
            arr = repack_fields(pin_data[fields]).view((float, len(fields)))
            pin_prob = gmm.predict_proba(arr)

        ordered_indices = [np.argmax(p) for p in pin_prob]

        # check for duplicates
        if len(set(ordered_indices)) != n_components:
            print(f'{model} probabilities were not reordered')
            return prob

        prob[:, list(range(n_components))] = prob[:, ordered_indices]

        return prob
