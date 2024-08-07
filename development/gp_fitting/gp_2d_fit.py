import numpy as np
import george
import scipy
def gp_2d_fit(image_data, kernel='matern32'):
    """Fits data in 2D with gaussian process.

    The package ``george`` is used for the gaussian process fit.

    Parameters
    ----------
    image_data : 2-D array
        An image of N x N dimensions.
    kernel : str, default ``matern32``
        Kernel to be used to fit the light curves with gaussian process. E.g., ``matern52``, ``matern32``, ``squaredexp``.
    use_mcmc: bool, default ``False``
        Whether or not to use MCMC in the optimization of the gaussian process hyperparameters.
    Returns
    -------
    Returns a gaussian-process interpolated copy of the input image.
    """

    # define the objective function (negative log-likelihood in this case)
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    # and the gradient of the objective function
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    # for mcmc
    def lnprob(p):
        gp.set_parameter_vector(p)
        return gp.log_likelihood(y, quiet=True) + gp.log_prior()

    # prevents from changing the original values
    image = np.copy(image_data)

    # extract x values and reshape them for compatibility with george
    N = image.shape[0]
    x1 = np.hstack([list(range(0, N)) for i in range(0, N)])
    x2 = np.hstack([i]*N for i in range(0, N))
    X = np.array([x1, x2]).reshape(2, -1).T

    y = np.hstack(image)
    # normalize data
    y_norm = y.max()
    y /= y_norm

    # define kernel
    kernels_dict = {'matern52':george.kernels.Matern52Kernel,
                    'matern32':george.kernels.Matern32Kernel,
                    'squaredexp':george.kernels.ExpSquaredKernel,
                    }
    assert kernel in kernels_dict.keys(), f'"{kernel}" is not a valid kernel, choose one of the following ones: {list(kernels_dict.keys())}'

    var = np.std(y)
    length = 1  # random value, it can have a smarter initial value

    ker1, ker2 = kernels_dict[kernel], kernels_dict[kernel]
    ker = var * ker1(length**2, ndim=2, axes=0) * ker2(length**2, ndim=2, axes=1)

    mean_function = image_data.min()
    gp = george.GP(kernel=ker, mean=mean_function, fit_mean=False)

    # initial guess
    gp.compute(X)
    print('GP initial guess computed')

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    results = scipy.optimize.minimize(neg_ln_like, p0, jac=grad_neg_ln_like, method="Nelder-Mead")  # Nelder-Mead, L-BFGS-B, Powell, etc
    gp.set_parameter_vector(results.x)

    # steps for the predicted x1 and x2 dimensions
    step = 0.05
    x1_min, x1_max = x1.min(), x1.max()
    x2_min, x2_max = x2.min(), x2.max()
    X_predict = np.array(np.meshgrid(np.arange(x1_min, x1_max+step, step),
                             np.arange(x2_min, x2_max+step, step))).reshape(2, -1).T

    y_pred, var_pred = gp.predict(y, X_predict, return_var=True)
    yerr_pred = np.sqrt(var_pred)
    print('values predicted')

    # de-normalize results
    y_pred *= y_norm

    # Let's reshape the GP output to display it as an image
    temp_array = np.arange(x1.min(), x1.max()+step, step)
    N_pred = temp_array.shape[0]
    image_pred = np.array([y_pred[i*N_pred:(i+1)*N_pred] for i in range(N_pred)])

    return image_pred
