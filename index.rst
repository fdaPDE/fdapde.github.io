Welcome to the fdaPDE 2.0's alpha-testing!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

You can check the status of all open and closed issues, as well as known bugs, `here <https://github.com/orgs/fdaPDE/projects/2/views/1>`_

.. note::
   In the following, we will refer with `fdaPDE-CRAN` to the version 1.1-16 of the library currently available on CRAN (`link <https://cran.r-project.org/web/packages/fdaPDE/index.html>`_). `fdaPDE-2.0` refers to the new version which will replace the currently official one.

This page reports the guidelines to follow during the alpha-testing phase of `fdaPDE-2.0`. It also contains a **not official** draft of the `R` package documentation.
Moreover, this document will be kept updated for the whole alpha-testing phase. Check the :ref:`changelog` for the latest updates.

.. important::
   `fdaPDE-2.0` is currently in active development. You should frequently reinstall the package. Moreover, ignore eventual warnings raised at compile and installation time. They will be fixed, if possible, before the release on CRAN. Interface changes can happen for the whole testing period, up to the official release on CRAN.


Installation
------------

`fdaPDE-2.0` needs the following dependencies:

* a C++17 compiler
* :code:`Rcpp` and :code:`RcppEigen` (we plan to drop this dependency before the official release on CRAN)

Then, to install the library on your system, under the name :code:`fdaPDE2`, you can either:

* use the devtools package. From the R console, execute

  .. code-block::
     
     devtools::install_github("fdaPDE/fdaPDE-R", ref="stable") 

* clone the `fdaPDE-R` repository and install. From a (linux) terminal, execute

  .. code-block::
     
     git clone --recurse-submodules -b stable git@github.com:fdaPDE/fdaPDE-R.git 
     cd path/to/fdaPDE-R

  then, install the package from the R console with

  .. code-block::
   
     install.packages(".", type="source", repos=NULL) 

Changelog
---------

| 30/11/23: release of the alpha-testing version.
| 06/12/23: code tested against the following compilers: gcc, clang, apple-clang (solves `fdaPDE-R#3 <https://github.com/fdaPDE/fdaPDE-R/issues/3>`_, package should install on macos systems).

Bug reports
-----------

If you find a bug, please notify it by opening a draft pull request `here <https://github.com/orgs/fdaPDE/projects/2/views/1>`_. A bug report must contain a MWE (Minimal Working Example) which reproduces the error. Select also a severity level for the error.

Library interface
-----------------

This is just an informal presentation of the R interface. In particular, it is not an API specification.

Functional Space
++++++++++++++++

A functional space represents the concept of a functional basis over a given domain. Currently only Lagrangian basis are supported (aka finite element basis functions).

.. code-block:: R

   library(fdaPDE2)
   data("unit_square", package = "fdaPDE2")
   unit_square <- Mesh(unit_square)

   ## the functional space of finite element functions of order 1, over the unit square
   Vh <- FunctionSpace(unit_square, fe_order = 1)
   basis <- Vh$get_basis() ## recover the (lagrangian) basis system of Vh
   
   ## evaluate the basis system on a given set of locations (compute Psi matrix)
   locations <- unit_square$nodes
   Psi <- basis$eval(type = "pointwise", locations)
   
   ## integrate a function (expressed as basis expansion on Vh) over the domain
   f <- function(p) { p[,1]^2 + p[,2]^2 } ## x^2 + y^2
   Vh$integrate(f)

PDEs
++++

The new version of the library let you explicitly write differential operators in strong form. If you are going to define a statistical model, what reported below is not of any interest if you decide to default to a standard laplacian penalty.

.. code-block:: R

   library(fdaPDE2)
   data("unit_square", package = "fdaPDE2")
   unit_square <- Mesh(unit_square)
   
   ## the functional space of finite element functions of order 1, over the unit square
   Vh <- FunctionSpace(unit_square, fe_order = 1)
   f  <- Function(Vh) ## a generic element of the space Vh

   ## compose the differential operator (in strong form)
   Lf <- -laplace(f) + dot(c(1,1), grad(f)) ## a costant coefficients advection-diffusion problem
   ## define the forcing term
   u <- function(points) { return(rep(1, times = nrow(points))) }
   
   ## create your penalty
   penalty <- pde(Lf, u)

Supported operators are

.. list-table:: 
   :widths: 15 15 70
   :header-rows: 1

   * - operator
     - code
     - note
   * - laplacian
     - :code:`laplacian(f)`
     - observe that expressions are signed, :code:`laplacian(f)` is differnt from :code:`-laplacian(f)`
   * - divergence
     - :code:`div(K*grad(f))`
     - the diffusion tensor K can be either a constant matrix, or a function returning a matrix (space-varying coefficient)
   * - transport
     - :code:`dot(b,grad(f))`
     - the transport coefficient b can be either a constant vector or a vector field (space-varying coefficient)
   * - reaction
     - :code:`c*f`
     - the reaction coefficient c can be either a constant or a scalar field (space-varying coefficient)
   * -
     - :code:`dt(f)`
     - specifies that the problem is time dependent (parabolic PDE)

.. code-block:: R
		
   ## general linear second order parabolic operator
   Lf <- dt(f) - div(K * grad(f)) + dot(b, grad(f)) + c * f

More examples can be found on the `femR documentation <https://fdapde.github.io/femR/articles/Introduction.html>`_ or `on the test scripts <https://github.com/fdaPDE/femR/tree/stable/tests>`_.

SRPDE : Spatial Regression models
+++++++++++++++++++++++++++++++++

Here the interface for a general spatial regression model:

.. code-block:: R

   library(fdaPDE2)
   data("unit_square", package = "fdaPDE2")
   unit_square <- Mesh(unit_square)
   
   data_frame <- ## obtain your data in some way...
   ## currently, the only requirement is data_frame to be a data.frame object, e.g., 
   
   ##            y         x1         x2
   ## 1 -0.04044303  0.1402058 0.00000000
   ## 2  0.15079619  1.1989599 0.03447593
   ## 3  0.02391597 -2.3299685 0.06891086
   ## 4  0.38927632  0.5709451 0.10326387
   ## 5  0.39417457  2.7482761 0.13749409
   ## 6  0.33297548  1.7080400 0.17156085
   
   ## a model is first described in an "abstract" way, think for instance to the continuous
   ## functional J(f, \beta) we minimize when solving a smoothing problem
   
   ## a nonparametric spatial regression model
   model <- SRPDE(y ~ f, domain = unit_square, data = data_frame, lambda = 1e-6)

   ## this will inject in the current environment a Function object named f, representing the 
   ## unknown spatial field and defined on a FunctionSpace(unit_square, fe_order = 1)
   ## the name of the spatial field supplied in formula can be any
   
   model$fit() ## fits the model

   ## we can describe a semi-parametric spatial regression model as
   model <- SRPDE(y ~ x1 + f, domain = unit_square, data = data_frame, lambda = 1e-6)
   model$fit()


The following is an equivalent, but more general, way to describe the above model

.. code-block:: R   

   ## define the differential penalty
   f <- Function(FunctionSpace(unit_square, fe_order = 1))
   Lf <- -laplace(f) ## simple laplacian penalty
   u <- function(points) { return(rep(0, times = nrow(points))) }

   ## supply the penalty to the model and fit
   model <- SRPDE(y ~ f, penalty = PDE(Lf, u), data = data_frame, lambda = 1e-6)
   model$fit()

:code:`SRPDE` exposes a :code:`family` parameter whose default is set to :code:`gaussian`, and which makes the model to behave as a linear spatial regression model, as described, for instance in *Sangalli, L.M. (2021), Spatial regression with partial differential equation regularization, International Statistical Review*. Possible values for the :code:`family` parameters are

.. list-table:: 
   :widths: 15 85
   :header-rows: 1

   * - family
     - note
   * - :code:`poisson`, :code:`gamma`, :code:`bernulli`, :code:`exponential`
     - generalized spatial regression model, solved via FPIRLS, as detailed in *Wilhelm, M., Sangalli, L.M. (2016), Generalized Spatial Regression with Differential Regularization, Journal of Statistical Computation and Simulation*
   * - :code:`quantile`
     - quantile spatial regression model, solved via FPIRLS, as described in *De Sanctis, M., Di Battista, I., Spatial Quantile regression with Partial Differential Equation Regularization, PACS report*


STRPDE : Spatio-Temporal Regression models
++++++++++++++++++++++++++++++++++++++++++

Coming soon

Model fit customization
+++++++++++++++++++++++

When we write something like

.. code-block:: R
		
   model <- SRPDE(y ~ f, penalty = PDE(Lf, u), data = data_frame, lambda = 1e-6)

we are describing a model without fixing any computational detail regarding its resolution. Specifically, we are defining the estimation functional

.. math::

   J(\beta, f) = \sum_{i=1}^n (y_i - f(\boldsymbol{p}_i))^2 + \lambda_{\mathcal{D}} \int_{\mathcal{D}} (Lf - u)^2 d\mathcal{D}.

We can customize the way a model is fit using the :code:`fit` method. The code below shows how to select the smoothing parameter via GCV minimization, using a grid approach.


.. code-block:: R
		
   library(fdaPDE2)
   data("unit_square", package = "fdaPDE2")
   unit_square <- Mesh(unit_square)

   data_frame <- ## obtain your data in some way...
   model <- SRPDE(y ~ f, domain = unit_square, data = data_frame) ## do not fix any lambda...

   ## ... because you might want to select it via GCV minimization!
   lambda_grid = 10^seq(-4, -3, by = 0.1)
   model$fit(
      lambda = gcv(optimizer = "grid", lambda = lambda_grid)
   )

See :ref:`GCV` for more details on the use of the :code:`gcv()` method. The idea is extended to any other type of models. The snippet below shows how to customize the behaviour of the Functional Penalized Iterative Reweighted Least Square (FPIRLS) method used inside a generalized regression model

.. code-block:: R
		
   library(fdaPDE2)
   data("unit_square", package = "fdaPDE2")
   unit_square <- Mesh(unit_square)

   data_frame <- ## obtain your data in some way...
   model <- SRPDE(y ~ f, domain = unit_square, data = data_frame, family = "poisson")
   
   lambda_grid = 10^seq(-4, -3, by = 0.1)
   model$fit(
      lambda = gcv(optimizer = "grid", lambda = lambda_grid),
      fprils_params = list( ## customize FPIRLS execution
         max_iterations = 100,
         tolerance = 1e-6
      )
   )

   ## you are not forced to set all the parameters, if you do not set some (or all) of them,
   ## there will always be a default

.. important::
   
   More on this must be explored and tested. Moreover any parameter should always have a default, so that :code:`model$fit()` is always well defined. Check the changelog to see updates in this direction.
   
GCV
+++

This sections shows the API provided by the :code:`gcv()` method, which can be used to tune the smoothing parameter in a regression model.

.. code:: R

   ## stochastic approximation of degrees of freedom, fast but approximate
   model$fit(
      lambda = gcv(
         edf_computation = "stochastic",
         seed = 143547654,    ## set seed in internal stochastic engine (for reproducibility)
         n_mc_samples = 500,  ## number of mc samples in stochastic algorithm
         optimizer = "...",
         ... ## additional arguments forwarded to the optimizer
   )

   ## exact approximation of degrees of freedom, slow but exact
   model$fit(
      lambda = gcv(
         edf_computation = "exact",
         optimizer = "...",
         ... ## additional arguments forwarded to the optimizer
   )

.. info::
   
   In case :code:`edf_computation = "stochastic"`, the following defaults are available:
   * :code:`seed = NULL` forces a random initialization for the stochastic algorithm
   * :code:`n_mc_samples = NULL` sets the number of monte carlo realizations to `100`

The GCV objective can be optimizer using different methods, reported in the table below

.. list-table:: 
   :widths: 15 85
   :header-rows: 1

   * - optimizer
     - note
   * - :code:`grid`
     - optimization on a fixed and user-defined grid of values. Currently the only possibility for space-time models. Requires

       * :code:`lambda`: vector of smoothing parameters to explore
   * - :code:`newton`
     - Newton optimization. This is an iterative optimization method. Requires

       * :code:`lambda`: starting point of the iterative method
       * :code:`exact_derivative`: a boolean flag which, if set to false, enables the approximate computation of the GCV derivative via central finite differences (:code:`exact_derivatives = TRUE` currently not supported)
       * :code:`tolerance` : tolearance on the error between consecutive iterations to stop the optimization
       * :code:`step`: step value used in the Newton update (adaptve steps :code:`backtracking`, :code:`wolfe` currently not supported)
   * - :code:`gradient_descent`
     - Gradient descent optimization. This is an iterative optimization method.

       .. warning::

	  This optimization method has never been explored in the context of GCV minimization.

       Requires
	  
       * :code:`lambda`: starting point of the iterative method
       * :code:`exact_derivative`: a boolean flag which, if set to false, enables the approximate computation of the GCV derivative via central finite differences (:code:`exact_derivatives = TRUE` currently not supported)

       * :code:`tolerance` : tolearance on the error between consecutive iterations to stop the optimization
       * :code:`step`: step value used in the Newton update (adaptve steps :code:`backtracking`, :code:`wolfe` currently not supported)
   * - :code:`bfgs`
     - The method of Broyden–Fletcher–Goldfarb–Shanno for unconstrained nonlinear optimization. This is an iterative optimization method.

       .. warning::

	  This optimization method has never been explored in the context of GCV minimization.

       Requires
	  
       * :code:`lambda`: starting point of the iterative method
       * :code:`exact_derivative`: a boolean flag which, if set to false, enables the approximate computation of the GCV derivative via central finite differences (:code:`exact_derivatives = TRUE` currently not supported)

       * :code:`tolerance` : tolearance on the error between consecutive iterations to stop the optimization
       * :code:`step`: step value used in the Newton update (adaptve steps :code:`backtracking`, :code:`wolfe` currently not supported)
