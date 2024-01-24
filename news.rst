fdaPDE's developers newsletter
==============================

Jan 2024
--------

**core**

* **1D meshes**: the :code:`mesh` module explicitly supports 1D meshes (intervals). Before of this update, the only way to handle one dimensional intervals was to employ a degenerate linear network. Now :code:`Mesh<1, 1>` is provided to support this functionality. In addition, point location over 1D interval is implemented using a fast :math:`O(\log(n))` binary search without additional memory storage. The class supports also a convenient linspaced constructor for meshing intervals :math:`[a,b]` with equispaced nodes :code:`Mesh<1, 1>::Mesh(double a, double b, int n_nodes)`.
  
* **Discretization of 1D PDEs using B-spline basis**: PDEs can be discretized using a B-spline basis expansion of the solution. The :code:`spline` module allows to define a PDE on a :code:`Mesh<1, 1>` with the following API, using a B-spline discretization:

  .. code-block:: cpp
     :caption: One dimensional bilaplacian operator, discretized via cubic B-Splines

     Mesh<1, 1> unit_interval(0, 1, 10);
     // SPLINE declares the intention to discretize this operator using a B-spline basis expansion
     // of its solution.
     auto Lt = -bilaplacian<SPLINE>();    // strong formulation of the differential operator
     PDE<Mesh<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(unit_interval, Lt);

  .. note::
     :title: Still missing
       
     Diffusion and transport operators, non-homogeneous forcing terms, Dirichlet and non-homogeneous Neumann boundary conditions, time-dependent problems, non-linearities.

* **Binary matrices**: the linear algebra module supports the definition and manipulation of binary valued matrices, via the template :code:`BinaryMatrix<Rows, Cols>`. Template parameters :code:`Rows` and :code:`Cols` can be set to :code:`fdapde::Dynamic` to express a matrix whose dimesions are not known at compile time. Due to its particularly efficient implementation, binary matrices should always be preferred over :code:`std::vector<bool>` or :code:`DMatrix<bool>` in the library, whenever there is the need to manipulate vectors (or matrices) of binary values.

  .. code-block:: cpp
     :caption: BinaryMatrix API

     // a statically stored binary matrix (coefficients set to 0 by default)
     BinaryMatrix<2,2> m1;
     m1.set(0,0);    // set to 1 the coefficient in position (0,0)

     // another statically stored binary matrix
     BinaryMatrix<2,2> m2;
     m2.set(1,1);

     // bitwise operations
     BinaryMatrix<2,2> m3 = m1 | m2;    // bitwise or (addition modulo 2)
     BinaryMatrix<2,2> m4 = m1 & m2;    // bitwise and (product modulo 2)
     BinaryMatrix<2,2> m5 = m1 ^ m2;    // bitwise xor
     BinaryMatrix<2,2> m6 = ~m1;        // binary negation

     // bitwise expression templates :)
     auto e = (m1 | m2) & ~m1;

     // a 5 x 100, dynamically sized, binary matrix
     BinaryMatrix<fdapde::Dynamic> m7(5, 100);
     m7.set(4, 70);

     // block operations
     auto r = m7.row(2);    // extract the third row
     auto c = m7.col(4);    // extract the fourh column
     auto b = m7.block(2, 40, 3, 30);            // extract a 3 x 30 block starting at position (2,40)
     auto static_block = m7.block<3, 30>(2, 40); // static sized version of the above

     // obtain a new binary matrix by repating m7 2 times by rows and 4 times by columns
     BinaryMatrix<Dynamic> m8 = m7.blk_repeat(2, 4);

     // visitors
     bool v1 = m3.all(); // are all the coefficients of m3 set to true?
     bool v2 = m3.any(); // is there at least one coefficient of m3 set to true?
     std::size_t count = m3.count(); // how many coefficients of m3 set to true?

     // binary vectors are defined as BinaryMatrix<Rows, 1>, all the API above remains valid
     BinaryVector<Dynamic> vec(500);
     vec.set(10);
     vec = m7.row(3);
		  
  .. tip::

     Binary matrices are expecially convenient to express bitmasks, e.g., to express the presence or absence of an observation at a given location. :code:`BinaryMatrix<Rows, Cols>` exposes a :code:`select()` method which can be used to mask a given dense or sparse Eigen expression.

     .. code-block:: cpp
	:caption: Mask an Eigen matrix using a BinaryMatrix

	SpMatrix<double> A(10, 10);
	BinaryMatrix<fdapde::Dynamic> mask(10, 10);

	// produce a (sparse) matrix B keeping only those coefficients of A which matches with ones in the mask,
	// sets all the others to zero
	SpMatrix<double> B = mask.select(A);

	// the same holds for dense expressions.
     
  .. info::

     A :code:`BinaryMatrix<Rows, Cols>` does not store its coefficients using one integer for each coefficient. Instead, each integer is used to store :code:`8*sizeof(std::uintmax_t)` coefficients (this value is architecture dependent, for instance, each integer can store 64 bits on a 64-bit architecture). This means that a binary matrix with less than 64 coefficients is stored using a single integer (with a space-consumption of 8 bytes on a 64-bit architecture).

     This memory representation makes the datatype extremely efficient. Indeed, operations between binary matrices are performed at batches of :code:`8*sizeof(std::uintmax_t)` coefficients, e.g., the logical sum (addition modulo 2) between two binary matrices with less than :code:`8*sizeof(std::uintmax_t)` is performed with one single machine instruction, instead of using a costly loop coefficient by coefficient.

* **Mass lumping**: the linear algebra module supports the computation of the lumped matrix of a given Eigen expression. Both sparse and dense expressions are supported. The implemented lumped operator is the classical row-sum operator.

  .. code-block:: cpp
     :caption: Mass-lumping of a matrix

     SpMatrix<double> R0;                   // some sparse matrix
     SpMatrix<double> R0_lumped = lump(R0); // mass-lumped R0
     // obtain the mass lumped matrix of eigen expressions
     SpMatrix<double> lumped_matrix = lump(2*R0 + R0);

     // the above holds also for dense expresions.

  .. info::

     :code:`lump(A)` returns the mass-lumped matrix of A, not the inverse of its mass-lumped matrix.
     
* **Optimizers can be type-erased**: the optimization module provides a template :code:`Optimizer<F>` which is a type-erasure wrapper for optimization algorithms optimizing functors of type :code:`F`. :code:`Optimizer<F>` exposes the standard API of the optimization module. Check any optimizer in the optimization module for details.

  .. example::

     Thanks to the type-erasure technique, optimizers can be set and assigned using run-time decisions.
     
     .. code-block:: cpp
	:caption: Assign optimizer based on run-time decision
		     
        ScalarField<2> f([](const SVector<2>& p) -> double { return p[0] + 2*p[1]; });
	// an optimizer for 2D scalar fields
	Optimizer<ScalarField<2>> opt;

	// bound to opt any optimization algorithm at runtime
	if(some_runtime_condition) {
	    opt = BFGS<2, WolfeLineSearch>(max_iter, tolerance, step);              // BFGS with Wolfe step
	} else {
            opt = Newton<2, BacktrackingLineSearch> opt(max_iter, tolerance, step); // Newton with Backtracking step
	}
	// this works whenever f is a ScalarField<2>, independently on the implementation of f
	opt.optimize(f, SVector<2>(1,1));


     The above is used, e.g., in :code:`calibration::GCV` (see below) to set at run-time the type of optimizer used for GCV minimization. :code:`calibration::GCV` stores a member of type :code:`Optimizer<GCV>`, to enable the optimization of the GCV objective using any optimization strategy.

     
**cpp**

* **General PDEs for space-time separable penalized problems**: it is now possible to provide a generic 1D PDE as time penalty in a space-time separable penalized problem.

  .. note::
     
     The functionality is not tested outside the classical time-penalty usually encountered in literature, e.g. :math:`\int_{\mathcal{D} \times T} (\frac{\partial f}{\partial t})^2`, neverthless from this update on the internal infrastructure allows for generic operators in time.

  .. example::
     
           .. code-block:: cpp
	      :caption: A space-time separable STRPDE smoothing problem with general spatial and temporal penalties

	      // a spatio-temporal STRPDE model with separable penalty (details omitted)
	      // define temporal and spatial domain... 

	      // spatial regularization
	      auto Ld = -laplacian<FEM>(); // simple laplacian penalty in space
	      PDE<Mesh<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(space_domain, Ld, u);
	      // temporal regularization
	      auto Lt = -bilaplacian<SPLINE>(); // penalty on the second derivative in time
	      PDE<Mesh<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_domain, Lt);
	      
	      STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::mesh_nodes);  

	   The writing above implements an STRPDE model as usually encountered in literature. Neverthless :code:`Lt` can now be any operator time. It is also worth to mention that :code:`-bilaplacian<SPLINE>` refers to the fourth order problem one gets by developing the math. This might be misleading, as we are actually penalizing for a laplacian (second order derivative in time). Name changes are possible in this respect.

* **K-fold Cross Validation**: support for a general implementation of a K-fold cross validation strategy with random partition in train and test set. :code:`KCV` fulfills the calibrator concept (see below for details).

  .. code-block:: cpp
     :caption: K-Fold CV fit signature
	       
     template <typename ModelType, typename ScoreType>
     DVector<double> fit(ModelType& model, const std::vector<DVector<double>>& lambdas, ScoreType cv_score);

  Specifically :code:`ScoreType` must be a functor with the following singature

  .. code-block:: cpp
     :caption: ScoreType call operator signature
	       
     double operator()(
      const DVector<double>& lambda, const BinaryVector<fdapde::Dynamic>& train_mask,
      const BinaryVector<fdapde::Dynamic>& test_mask);

  and must return the model score for a given smoothing parameter and train/test partition. Check :code:`RMSE` for an example.
  
  .. info::

     :code:`KCV` splits the data (previously shuffled if requested) in K folds, and just invokes the provided cross validation index with the currently explored smoothing parameter and train/test partition. As such, the specific scoring logic, i.e., the core of the calibration strategy, is completely moved on the :code:`ScoreType` data type.

     Moreover, there is no actual data splitting, nor data replication, while producing the data folds. Instead, properly defined masks, implemented as :code:`BinaryVector<Dynamic>`, are produced to implement the partitioning in train and test sets. 

  .. example:: 

     The code below shows how to calibrate the smoothing parameter of an SRPDE model using a 10-fold CV strategy minimizing the model's RMSE.
     
     .. code-block:: cpp
	:caption: 10-fold CV smoothing parameter selection via RMSE minimization
		     
	// define some statistical model
	SRPDE model(problem, Sampling::mesh_nodes);
	// define KCV engine and search for best lambda which minimizes the model's RMSE
	std::size_t n_folds = 10;
	KCV kcv(n_folds);
	std::vector<DVector<double>> lambdas;
	for (double x = -6.0; x <= -3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
	kcv.fit(model, lambdas, RMSE(model));
	
     For an higher-level API, check the calibrator concept below.

  .. note::
     :title: Still missing
     
     The functionality is still considered unstable, as extensive numerical tests for all the supported models are required.
		     
* **Calibrators**: the calibrator concept introduces a unified way to calibrate a statistical model (e.g. select its smoothing parameters). The only requirement for a type T to be a calibrator is to expose a :code:`fit` method with the following signature

  .. code-block:: cpp
     :caption: Calibrator concept fit signature
		  
     template <typename ModelType, typename... Args> DVector<double> fit(ModelType& model, Args&&... args);

  :code:`fit` takes the model whose parameters must be selected and additional arguments required for the specific calibration algorithm. It returns the selected smoothing parameter. Are examples of calibrators, :code:`calibration::KCV` and :code:`calibration::GCV`.

  .. abstract::
     
     :code:`calibration::GCV` must not be confused with :code:`model::GCV`. While the latter is a functor representing the GCV objective, the former represents a calibrator. :code:`model::GCV` offers a lower-level API than its calibrator. To see the differences, check the following code snippets:


		 
     .. code-block:: cpp
	:caption: :code:`model::GCV` API

	// define some statistical model
	SRPDE model(pde, Sampling::mesh_nodes);
	// request its GCV objective (use approximated Tr[S])
	std::size_t seed = 476813;
	auto GCV = model.gcv<StochasticEDF>(100, seed);
	// optimize GCV (require a grid optimization)
	DVector<double> opt_lambda = core::Grid<fdapde::Dynamic>{}.optimize(GCV, lambda_grid);

     .. code-block:: cpp
	:caption: :code:`calibration::GCV` API

	// define some statistical model
	SRPDE model(pde, Sampling::mesh_nodes);
	// define GCV calibrator (pay attention that a calibrator is model independent)
	std::size_t seed = 476813;
  	auto calibrator = calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(100, seed)};
	// fit the model using the calibrator
	DVector<double> opt_lambda = calibrator(lambda_grid).fit(model);

     Pay attention that **a calibrator never depends on a statistical model**. It allows for a functional way to express a calibration strategy which does not depend on a specific model instance. For instance

     .. code-block:: cpp
		     
  	auto calibrator = calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(100, seed)};

     represents a calibration strategy for a (regression) model based on GCV minimization, optimized over a grid of smoothing parameters, and using a stochastic approximation for the edfs. Note that in the above definition no model is specified. Moreover, it is copy/move assignable, i.e., it can be stored and given as argument to other functions.

     The first argument of :code:`calibrator::GCV` can be any optimizer in the core module, for instance a calibrator so defined

     .. code-block:: cpp

  	auto calibrator = calibration::GCV {Newton<fdapde::Dynamic, BacktrackingLineSearch> (10, 0.05, 1), StochasticEDF(100, seed)};

     express a calibration strategy for a (regression) model whose GCV is optimized using a newton method with adaptive step size (backtracking line search), using a stochastic approximation for the edfs. Check the optimization module for further details.

     :code:`calibration::GCV` is a functor, exposing a call operator which forwards its arguments to the optimizer (e.g., the initial point for an iterative optimization routine, or a grid of points for a brute force optimization). The result is an instance of :code:`ConfiguredCalibrator` with a :code:`fit` method accepting the model instance. The calibration is lazily evaluated, e.g., computation starts only when fit is invoked.

     .. code-block:: cpp
		     
	// set up the internal optimization algorithm with the choosen grid of smoothing parameters and fit the model
	DVector<double> opt_lambda = calibrator(lambdas).fit(model);

  .. note::
     :code:`calibration::KCV` allows for the selection of the smoothing parameter of a statistical model, using a K-Fold Cross Validation approach. Observe that thanks to the low requirements for the model type accepted by :code:`calibration::KCV`, any model class (not only regression models) can be provided to this calibrator. The snippet below shows the provided API

     .. code-block:: cpp
	:caption: K-fold CV based calibration of an SRPDE model using a calibrator
		     
	// define some statistical model
	SRPDE model(pde, Sampling::mesh_nodes);
	// define KCV calibrator minimizing the Root Mean Squared Error (RMSE) of the model
	std::size_t n_folds = 10;
	std::size_t seed = 476813;
	auto calibrator = calibration::KCV {n_folds, seed}(lambda_grid, RMSE());
	// fit the model with the selected calibration strategy
	DVector<double> opt_lambda = calibrator.fit(model);
	      

  Functions accepting a calibration strategy should accept a :code:`ConfiguredCalibrator` instance. In this way, the routine is abstracted from the calibration strategy, allowing to provide any type of calibration to the algorithm. For an example, see the :code:`center` routine for the functional centering of a data matrix.

* **Functional centering**: the functional module now offer support for the smooth centering of a given data matrix :math:`X` via the :code:`center` routine. It returns the centered data togheter with the expansion coefficients of the mean field.

  .. example::

     The functional centering of a data matrix :math:`X`, which provides the following signature

     .. code-block:: cpp
	:caption: center signature
		  
	template <typename SmootherType_, typename CalibrationType_>
	CenterReturnType center(const DMatrix<double>& X, SmootherType_&& smoother, CalibrationType_&& calibration);

     is an example of the flexibility of the calibrator concept. The :code:`center` function does not assume any type of smoothing algorithm to produce the smooth mean, nor any type of calibration strategy to find the optimal smoothing parameter for the smoother. Users of the centering algorithm define whatever they find more appropriate for their use case.

     .. code-block:: cpp
	:caption: centering of a data matrix X using a GCV-calibrated SRPDE model
		     
	// center the data matrix X, using the smooth mean field obtained from an SRPDE model tuned according to its GCV index
	// (optimized over a grid of smoothing parameters) applied on the pointwise mean estimator of X
	auto centered_data = center(
	   X, SRPDE {pde, Sampling::mesh_nodes}, calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(100)}(lambda_grid));

	// centered_data is of type CenterReturnType, a struct providing access to:
	centred_data.fitted // centred data X - \mu
	centred_data.mean   // mean field expansion coefficients

     .. note::
	
	The requirements on the smoother are so low that also a :code:`RegressionModel<void>` instance (type-erased wrapper for a regression model without any assumption on its penalty) is a valid smoother.

* **functional PCA**: official support for sequential fPCA (*Lila, E., Aston, J.A.D., Sangalli, L.M. (2016), Smooth Principal Component Analysis over two-dimensional manifolds with an application to Neuroimaging, Annals of Applied Statistics, 10 (4), 1854-1879.*) with GCV and KCV calibration of the optimal smoothing parameter for each component. Space-only version.

  In the initialization step, SVD is now placed outside the PC functions computational for loop.

  PC functions are always normalized with respect to the functional :math:`L^2` norm, loadings are the evaluation of these :math:`L^2`-normalized fields at the data locations (they are no more normalized in euclidean norm).

  Official support for monolithic fPCA based on Regularized SVD with fixed smoothing parameter.

  .. code-block:: cpp
     :caption: Functional Principal Component Analysis cpp API

     // fPCA with fixed lambda for each component, sequential solver
     FPCA<SpaceOnly, fdapde::sequential> model(pde, Sampling::mesh_nodes, Calibration::off);
     // replacing Calibration::off, with Calibration::gcv or Calibration::kcv makes the model to
     // switch the selection of the level of smoothing for each compoent to the desired strategy

     // solve the same problem with a monolithic (RSVD-based) solver
     FPCA<SpaceOnly, fdapde::monolithic> model(pde, Sampling::mesh_nodes);     

  Check :code:`test/src/fpca_test.cpp` for the detailed API.
  
**R (base)**

* no notable changings. Moving the internal implementation to R6 classes. At this stage still in an early development phase.
