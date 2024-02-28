:hero: fdaPDE's developers newsletter

Feb 2024
========

.. warning::
   :title: Upgrade of the C++ standard

   From this update on, code is written conforming to the C++20 standard. Starting from R 4.0.0 packages can specify C++20 as requirement, and from R 4.3.0 even the newest C++23. As an indicator that R is pushing toward latest standards, from R version 4.3.0 *the default C++ standard has been changed to C++17 where available (which it is on all currently checked platforms)* (check the `latest patch note <https://cran.rstudio.com/doc/manuals/r-patched/NEWS.pdf>`_).

   Nowadays there is extensive support from all major compilers for the C++20 standard (`C++ compiler support <https://en.cppreference.com/w/cpp/compiler_support/20>`_). C++20 introduces several interesting core language features, among which, concepts, coroutines, templated lambdas, etc. :code:`fdaPDE` will slowly migrate and exploit the capabilities of C++20 during this year. Migration to C++23 is planned for the next year.

   .. error::
      :title: danger

      At the time of this update, raising the standard to C++20 causes compilation problems at the R level, due to the package :code:`RCppEigen`. The package is going to be updated soon, with the latest version of Eigen (which is the 3.4, see `here <https://github.com/RcppCore/RcppEigen/issues/103>`_). This will solve the issue. At this stage, since the R package is still in development, there is no such issue.
   
**core**

* **Binary trees**: the implementation of the :code:`BinaryTree` data structure has been completely revised. The new implementation is more space-time efficient, and offers a way better API.

  .. code-block:: cpp
     :caption: BinaryTree signature

     template <typename T> class BinaryTree;
     
  :code:`BinaryTree` provides a low-level API, which should be used to device more sophisticated data structures, using the adaptor pattern. See, for instance, :code:`BST` and :code:`KDTree`.

  .. code-block:: cpp
     :caption: BinaryTree API

     // custom construction of a binary tree
     BinaryTree<int> tree(1);               // root stores 1
     // push operations return an iterator to the inserted node
     auto n2 = tree.push_left (root(), 2);  // push node storing 2 as left  child of root
     auto n3 = tree.push_right(root(), 3);  // push node storing 3 as right child of root
     tree.push_left(n3, 4);                 // push node storing 4 as left child of n3
     auto n5 = tree.push_left(n2, 5);       // push node storing 5 as left child of n2
     tree.emplace_left(n5, 6);              // emplace 6 as left child of n5

     // while push operations create a copy of the provided element, emplace constructs the
     // element in place (here useless for int types, can be beneficial for more complex ones)

     // the insertion point can be determined in a functional way, i.e., by supplying a functor
     // to the push() method inducing a partial ordering relation. For instance,
     struct sorted_insertion {
        template <typename T>
	bool operator()(const T& data, typename BinaryTree<T>::node_pointer node) {
	    return data < node->data;
	}
     };
     tree.push(10, sorted_insertion {});    // sorted insertion, push 10 as right child of n3

     // check the BST data type below, which implements a binary search tree in this way
     
     tree.empty();   // is tree empty?
     tree.size();    // how many nodes in the tree?
     tree.root();    // dfs_iterator to root
     tree.clear();   // deallocate all nodes in the tree
     
     tree[n5] = 20;  // write access to value pointed by n5
     tree.at(n3);    // read-only access to value pointed by n3
     
     // range-for loop, performs a depth-first traversal of the tree 
     for(auto& node : tree) { node = 2 * node; } // multiply all the stored values by 2

     // BinaryTree exposes different iterators to traverse the tree
     // depth-first traversal:   prints 1 2 5 6 3 4
     for(auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it) { std::cout << *it << std::endl; }
     // breadth-first traversal: prints 1 2 3 4 5 6
     for(auto it = tree.bfs_begin(); it != tree.bfs_end(); ++it) { std::cout << *it << std::endl; }
     // cycle over all leaf nodes, in dfs order, and store them in a vector
     std::vector<int> leafs;
     for(auto it = tree.leaf_begin(); it != tree.leaf_end(); ++it) {
	leafs.push_back(*it);
     } // leafs vector contains: 6 4

     // BinaryTree is copy constructable/assignable
     BinaryTree<int> copied_tree = tree; // O(n), copies nodes one by one
     // BinaryTree move semantic
     BinaryTree<int> moved_tree = std::move(tree); // O(1), just some pointer swaps.
     // ... after the move tree is left empty, e.g. tree.size() evaluates to 0

  .. info::
     :title: Binary Search Trees

     The :code:`BST<T>` class is an adaptor of :code:`BinaryTree<T>` that gives the functionality of a Binary Search Tree, i.e., *a tree in which the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree*. The partial ordering is provided by something similar to :code:`sorted_insertion` in the code above.

     Specifically, the data structure exposes a subset of the interface of :code:`BinaryTree<T>` which guarantees the ordering property (which can instead be easily violated using a plain :code:`BinaryTree<T>`). Check the code snippet below for the details:

     .. code-block:: cpp
	:caption: BST API

	// a binary search tree of integers
	BST<int> bst = {7, 2, 1, 3, 4, 5, 6, 8};    // constructs a BST by pushing the values in the list
	bst.push(9);   // inserts 9 using the ordering induced by sorted_insertion (i.e., as right child of 8)
     
	for(const auto& node : tree) { std::cout << node << std::endl; } // prints 7, 2, 1, 3, 4, 5, 6, 8, 9

	// the whole API of BinaryTree is available, with the exception of:
	// - push_left, push_right
	// - push with a generic ordering relation
	// - emplace, emplace_left, emplace_right

	// you can search in a binary tree with O(log(n)) complexity
	auto found = bst.find(4); // found is a dfs_iterator pointing to 4
	bst[found] = 14;
	
	// you can test if an element is contained as follow
	if(bst.find(10) == bst.end()) { std::cout << "10 is not in bst" << std::endl; }

* **geometry module**: This is just a name change. From this update on, any geometric data structure or algorithm (mesh management, point location, nearest neighbors and range searches, voronoi tasselations, etc.) are placed under the `geometry` module (previously known as `mesh` module).
	
* **KD-trees**: The geometry module provides support for a `KD-Tree <https://en.wikipedia.org/wiki/K-d_tree>`_ data structure, built on top of a :code:`BinaryTree<int>`. A KD-Tree is used to index a set of :math:`n` given points and provide a quick answer to nearest neighbors queries, i.e., find the nearest point (in :math:`\| \cdot \|_2` distance) among the :math:`n` indexed ones to a given query point. The data structure can also serve to solve range queries, i.e., find all points contained in a given rectangle.

  The construction of a KD-Tree takes :math:`O(nlog(n))` time and :math:`O(n)` space. The average complexity to answer to a nearest neighbor query, as well as a range query, is :math:`O(log(n))` (worst case complexity is still :math:`O(n)`). 

  .. example::
     :title: KD-Tree usage example

     The geometry module implements the KD-Tree data structure by means of the template :code:`KDTree<K>`. Check the code snippet below for an example of its API
	     
     .. code-block:: cpp

	// let X be a set of K-dimensional points (assume K = 2)
	DMatrix<double> X = ...;
	// index the set X using a KD-Tree data structure
	KDTree<2> kdtree(X);

	// let p a given query point
	SVector<2> p(1,1);
	kdtree.nn_search(p); // what is the nearest point (in euclidean distance) in X to p?

	// a range query consists in finding the set of points in X which lie inside a given rectangle.
	// the query is defined by a pair of K-dimensional points, indicating the lower-left and upper-right
	// corner of the query rectangle
	auto ids = kdtree.range_search({SVector<2>(0.5, 0.5), SVector<2>(2,2)});

	// ids is a set containing the ids of all the points in X which fall inside [0.5, 2] x [0.5, 2]

  .. info::

     The previously available tree search strategy for the location of a point over a triangulation has been reimplemented using a :code:`KDTree` data structure. This improves its performances, thanks to the better balancing provided by a :code:`KDTree` with respect to the previous implementation. :code:`KDTree` indeed builds a partion of the space with a criterion which takes into account the distribution of the points, while the previous implementation was blind to such information. This could potentially result in a highly unbalanced tree, with a lookup complexity near to :math:`O(n)`. :code:`KDTree` guarantees instead a well-balanced tree, which in turn guarantees an average :math:`O(log(n))` lookup.

  .. info::

     :code:`KDTree` is of fundamental importance for *efficiently* solving the point-location problem on a Voronoi tasselation. Indeed, by definition of Voronoi diagram, a point :math:`p` lies inside a vornoi cell if the center of the cell is the nearest point to :math:`p`. Observe that other considerations must be made to effectively solve the point location problem over a **constrained** voronoi tasselation (where the constrain is given by a border). Neverthless, solving in logaritmic time the nearest neighbor problem guarantees a logaritmic complexity for the point location problem over a tasselation.
       
**cpp**

* **Regularized Singular Value Decomposition**: due to its central role for the implementation of functional models, check for instance fPCA and fPLS, the Regularized SVD (RSVD) is now standardized as an independent solver, and is exposed via the template :code:`RegularizedSVD<SolutionPolicy>`. The :code:`SolutionPolicy` template parameter configures the RSVD to work either sequentially or monolithically.

  The template is not designed to work without a model, as informations related to the penalty term are derived from a model instance. Most of the times it should be used by a model to implement its :code:`solve()` method. Details on the different solution policies follow:
  
  .. abstract::
     :title: Sequential RSVD API

     :code:`RegularizedSVD<fdapde::sequential>` solves the RSVD problem by rank-one steps, e.g. it sequentially minimizes in :math:`(\boldsymbol{s}, f)` the functional :math:`\| X - \boldsymbol{s}\boldsymbol{f}_n^\top \|_F^2 + \boldsymbol{s}^\top \boldsymbol{s} \mathcal{P}_{\lambda}(f)` up to a desired rank. Due to the presence of the smoothing parameter :math:`\lambda`, :code:`RegularizedSVD<fdapde::sequential>` requires a calibration strategy to be well-defined. Supported calibration strategies are

     .. list-table:: 
	:widths: 25 75

	* - :code:`Calibration:off`
	  - no calibration, the smoothing parameter is kept fixed for each component. :math:`\lambda` is obtained from the calling model.
	* - :code:`Calibration::gcv`
	  - smoothing parameters selected via minimization of the GCV index related to the internal smoothing step.
	* - :code:`Calibration::kcv`
	  - smoothing parameters selected with a K-fold cross validation strategy, looking for a minimum in the reconstruction error.

	
     :code:`RegularizedSVD<fdapde::sequential>` provides a :code:`compute` method with the following signature

     .. code-block:: cpp
	:caption: :code:`RegularizedSVD<sequential>::compute` signature

	template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, std::size_t rank)

     which computes the RSVD of the pair (X, model) up to a a desired rank :code:`rank` in one single run. This makes the algorithm behave, in its interface, like a monolithic approach. Since the algorithm is sequential in its nature, :code:`RegularizedSVD<fdapde::sequential>` exposes also an iterator-like API, via the :code:`rank_one_stepper()` method, which enables to manage the single components :math:`(\boldsymbol{s}, f)` as they are computed. Check the code example below:

     .. code-block:: cpp
	:caption: Sequential RSVD solver API

	// let m some model, and X some data

	// define RSVD sequential solver
	RegularizedSVD<sequential> rsvd (Calibration::off);
	// extract the first 3 components of X at once
	std::size_t rank = 3;
	rsvd.compute(X, m, rank);
	rsvd.loadings();      // matrix of L^2 normalized functional components [f_1, f_2, f_3]
	rsvd.scores();        // matrix of associated scores [s_1, s_2, s_3]
	rsvd.loadings_norm(); // vector of L^2 norms of rsvd.loadings()

	// use the rank-one stepper (iterator-like) API
	for(auto it = rsvd.rank_one_stepper(X, m); it != rank; ++it) {
	   it.loading(); // expansion coefficients of L^2 normalized functional component f_i
	   it.scores();  // associated score vector
	   it.norm();    // L^2 norm of it.loading()
	   it.lambda();  // best smoothing parameter selected for this component
	   
	   // do whatever operation you need...
	}

     .. tip::

	The rank-one stepper approach might, for instance, be used by higher-level algorithms which must interleave the computation of single components with other, algorithm-specific, computations.

     For a detailed API on how to configure the algorithm, in case of :code:`Calibration::gcv` or :code:`Calibration::kcv`, check the code example below:
     
     .. code-block:: cpp
	:caption: Sequential RSVD solver API

	RegularizedSVD<fdapde::sequential> rsvd(Calibration::off);
	// configuration for the internal power-iteration method (check power_iteration.h for details)
	rsvd.set_tolerance(tol);      // tolerance before forced stop
	rsvd.set_max_iter(n_iter);    // maximum number of iterations

	RegularizedSVD<fdapde::sequential> rsvd(Calibration::gcv);
	// all the API available for Calibration::off, plus
	rsvd.set_seed(seed);          // seed used, e.g., for the stochastic approximation of Tr[S] involved in the computation of the GCV
	rsvd.set_lambda(lambda_grid); // grid of smoothing parameters for GCV (grid-based) minimization

	RegularizedSVD<fdapde::sequential> rsvd(Calibration::kcv);
	// all the API available for Calibration::off, plus
	rsvd.set_seed(seed);          // seed used, e.g., for the initial shuffling of the data before splitting the data in folds
	rsvd.set_lambda(lambda_grid); // grid of smoothing parameters 
	rsvd.set_folds(n_folds);      // number of folds employed in the K-fold cross validation
	
  .. abstract::
     :title: Monolithic RSVD API

     :code:`RegularizedSVD<fdapde::monolithic>` solves the RSVD problem in a single step, seeking for a rank :code:`rank` matrix :math:`U`, which factorizes as :math:`S F^\top`, minimizing :math:`\| X - U \Psi^\top \|_F^2 + \text{Trace}[U P_{\lambda}(f) U^\top]`. The data type offers a single :code:`compute()` method which provides the wanted factors :math:`S` and :math:`F`. Because the method works setting a unique level of smoothing :math:`\lambda`, the solver does not require any type of internal calibration.

     .. code-block:: cpp
	:caption: :code:`RegularizedSVD<monolithic>::compute` signature

	template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, std::size_t rank)

     Check the code below for an example of its API

     .. code-block:: cpp
	:caption: Monolithic RSVD solver API

	// let m some model, and X some data

	// define RSVD sequential solver
	RegularizedSVD<monolithic> rsvd;
	// extract the first 3 components of X at once
	std::size_t rank = 3;
	rsvd.compute(X, m, rank);
	rsvd.loadings();      // matrix of L^2 normalized functional components [f_1, f_2, f_3]
	rsvd.scores();        // matrix of associated scores [s_1, s_2, s_3]
	rsvd.loadings_norm(); // vector of L^2 norms of rsvd.loadings()     

  Thanks to the unified interface, it is possible to type erase the RSVD solvers, therefore making possible to switch solver at run-time. Check :code:`FPCA` for an example. Observe, anyway, that it is not possible to expose the :code:`rank_one_stepper` API by a type-erasure wrapper (as not shared with the monolithic solver).

* **functional PLS**: official support for functional Partial Least Squares (fPLS), space-only sequential version, with GCV and KCV calibration of the optimal smoothing parameter, for both the correlation maximization step (solved by an application of a sequential rank-one RSVD) and the regression step (solved by an application of an SRPDE model, for space-only data).

  .. info::

     fPLS assumes input matrices already centered. You can use the :code:`center()` routine from the functional module to obtain a smooth centering of the covariate matrix. The interface allows for a further degree of flexibility, as the smooth mean field can use a different calibration strategy than those supplied to fPLS.

  .. tip::

     fPLS does not avoid to set a different calibration strategy for the correlation maximization step and the smoothing step.

     If no calibrator for the smoothing step is supplied, fPLS defaults to a fixed-lambda calibration strategy, if the provided RSVD solver is instantiated with :code:`Calibration::off`, fallbacks to a GCV strategy otherwise (which results to be more computationally efficient than a KCV). In this case the GCV is optimized over the same grid supplied to the RSVD solver.

  

  For an example of the provided API, check the code example below.
     
  .. example::
     
     .. code-block:: cpp
	:caption: fPLS model API

	// assume mesh and laplacian penalty already defined...

	// definition of a functional PLS model for space-only data
	RegularizedSVD<fdapde::sequential> rsvd {Calibration::gcv};
	rsvd.set_lambda(lambda_grid);
	FPLS<SpaceOnly> model(pde, Sampling::mesh_nodes, rsvd);

	// one-liner equivalent version
	FPLS<SpaceOnly> model(pde, Sampling::mesh_nodes, RegularizedSVD<fdapde::sequential> {Calibration::gcv}.set_lambda(lambda_grid));
	
	// configure the calibrator for the internal smoothing step
	model.set_smoothing_step_calibrator(
	   fdapde::calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(1000, seed)}(lambda_grid));
	   
	// set model's data
	BlockFrame<double, int> df;
	df.insert(OBSERVATIONS_BLK, DMatrix<double>(Y.rowwise() - Y.colwise().mean()));   // pointwise centred responses
	// smooth centred functional covariates (select optimal smoothing by GCV)
	auto centered_covs = center(
	   X, SRPDE {pde, Sampling::mesh_nodes},
	   fdapde::calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(1000, seed)}(lambda_grid));
	df.insert(DESIGN_MATRIX_BLK, centered_covs.fitted);
	model.set_data(df);
	// solve FPLS problem
	model.init();
	model.solve();

     .. warning::

	Internally, fPLS stores a :code:`RegularizedSVD<sequential>` instance. Trying to assign to fPLS a :code:`RegularizedSVD<monolithic>` is wrong, and will cause a compilation failure.
  
* **fPCA**: fPCA is now conform to the standard model signature, i.e. :code:`template <typename RegularizationType_> class FPCA`. Previous to this update, the resolution strategy used to solve the fPCA problem was provided as a template argument as :code:`template <typename RegularizationType_, typename SolutionPolicy_> class FPCA`. Now, the strategy used to extract the princiapl components is defined at run-time by providing a proper :code:`RegularizedSVD` instance. Check the code example below:

  .. example::
     
     .. code-block:: cpp
	:caption: fPCA model API

	// assume mesh and laplacian penalty already defined...
		  
	// definition of a functional PCA model, sequential version, space-only data
	RegularizedSVD<fdapde::sequential> rsvd {Calibration::gcv};
	rsvd.set_lambda(lambda_grid);
	FPCA<SpaceOnly> model(pde, Sampling::mesh_nodes, rsvd);

	// one-liner equivalent version
	FPCA<SpaceOnly> model(pde, Sampling::mesh_nodes, RegularizedSVD<fdapde::sequential> {Calibration::gcv}.set_lambda(lambda_grid));
	
	// set model's data
	BlockFrame<double, int> df;
	// smooth centered functional data (select optimal smoothing)
	auto centered_data = center(
	   X, SRPDE {pde, Sampling::mesh_nodes},
	   fdapde::calibration::GCV {Grid<fdapde::Dynamic> {}, StochasticEDF(1000, seed)}(lambda_grid));
	df.insert(OBSERVATIONS_BLK, centered_data.fitted);
	model.set_data(df);
	// solve fPCA problem
	model.init();
	model.solve();

     Internally, fPCA type-erases the RSVD solver, so that it is possible to provide to the model any type of RSVD, with any configuration. 
     
* **minor changes** :

  * **QSRPDE**: official support for space-time (separable) quantile regression. Use of smoothed pinball loss function in the computation of the GCV's numerator for both space and space-time data.
  * **fPCA**: official support for space-time fPCA.
  * **GCV**: the :code:`calibrator::GCV` datatype is now a template of the regularization type (either :code:`SpaceOnly` or :code:`SpaceTime`). This is necessary to let :code:`calibrator::GCV` interface with types which do not enforce any regularization at compile-time, see, e.g., a generic :code:`RegressionModel<void>`. In this sense :code:`calibrator::GCV<SpaceOnly>` is a calibrator for space-only (regression) models. Similarly :code:`calibrator::GCV<SpaceTime>` can be used to fit space-time models. :code:`calibrator::GCV` is still valid but the user is responsible to indicate the class of models (possibly with some runtime decision) before the fitting. Check code below:

    .. example::

       .. code:: cpp

	  // a GCV calibrator explicitly for space-only models
	  auto GCV_ = fdapde::calibration::GCV<SpaceOnly> {Newton<fdapde::Dynamic>(10, 0.05, 1), StochasticEDF(100, seed)};
	  // calibrate model
	  DVector<double> optimal_lambda = GCV_(lambda_start).fit(model);

	  // defer the specification of the penalty type
	  auto GCV_ = fdapde::calibration::GCV {Newton<fdapde::Dynamic>(10, 0.05, 1), StochasticEDF(100, seed)};
	  // before fitting the model, need to fix the GCV type
	  if(... some runtime decision ...) {
	      GCV_.set<SpaceOnly>();
	  } else {
	      GCV_.set<SpaceTime>();
	  }
	  DVector<double> optimal_lambda = GCV_(lambda_start).fit(model);
	  // not providing any type of regularization before fit() is potentially unsafe, and raises a runtime assert
	  
**R (base)**

The R wrapper officially adopts the R6 object system (see `here <https://r6.r-lib.org/articles/Introduction.html>`_ for the details).

* The :code:`Mesh` class supports 1D, 2D, 2.5D and 3D geometries.

  .. info::

     At the time of this update, linear networks are still not supported, as they require special care from the R side.

  .. code-block:: r
     :caption: Mesh API

     ## let M be the tangent space dimension. Define a list so formatted
     mesh_data <- list(
	nodes    = ## an n_nodes \times M matrix of coordinates
	elements = ## an n_elements \times M+1 matrix of indices
	boundary = ## an n_nodes \times 1 vector of integers, with 1 indicating boundary nodes, 0 otherwise
     )
     ## create a Mesh object
     mesh <- Mesh(mesh_data)
     
     ## in addition to active bindings (check R6 docs) to recover the input informations, we get
     mesh$neighbors ## for each element, the ids of adjacent elements
     mesh$edges     ## for each edge, the pair of nodes composing it
     
     ## solves the point location problem over the mesh:
     ## returns the ids of elements containing points (0.5, 0.5), (0.75, 0.75)
     mesh$locate(matrix(c(0.5, 0.5, 0.75, 0.75), byrow=TRUE, ncol = 2))

     ## special constructors
     unit_interval <- IntervalMesh(a, b, n)  ## interval [a, b] splitted using n nodes
     square        <- SquareMesh(c(a, b), n) ## square [a, b] \times [a, b]
     rectangle     <- RectangularMesh(c(ax, bx), c(ay, by), nx, ny)
       
* :code:`FunctionalSpace` now accepts a :code:`type` argument to specify the type of functional basis to instantiate. Additional arguments can be provided to specify some details of the basis system. Moreover, it is possible to take the tensor product of two basis systems using the :code:`%X%` operator. Check the code snippet below:
     
  .. code-block:: r
     :caption: FunctionalSpace API

     unit_square <- SquareMesh(c(0, 1), 100)

     ## A Lagrange finite element space of order 1 (CG = Continuous Galerkin finite elements)
     Vs <- FunctionalSpace(unit_square, type = "CG", order = 1)

     ## A (cubic) B spline basis function over the unit interval [0,1] 
     unit_interval <- IntervalMesh(0, 1, 10)
     Vt <- FunctionalSpace(unit_interval, type = "BSpline")
     Phi <- Vt$eval(seq(0, 1, by = 0.01)) ## basis function evaluation
     ## Phi is a n_basis \times n_locations sparse matrix (dgCMatrix) having its i-th column equal to the
     ## evaluation of the i-th basis function over the provided locations
     
     ## tensor product of basis systems
     Vh <- Vs %X% Vt ## the classical functional space used in separable regularization

     ## evaluation grid
     s_grid <- expand.grid(seq(0, 1, by = 0.01), seq(0, 1, by = 0.01)) ## space evaluation grid
     t_grid <- seq(0, 1, by = 0.01)                                    ## time  evaluation grid
     
     Psi <- Vh$eval(s_grid, t_grid) ## the matrix \Psi of separable regularization (obtained by tensorization)

  .. note::
     :title: Defaults

     :code:`FunctionalBasis` defaults to linear finite elements, so that :code:`Vh <- FunctionalSpace(mesh)` always constructs a linear finite element system over :code:`mesh`.

* Partial support for numerical integration.

  .. warning::
     :title: Experimental support
     
     The feature is still considered experimental. Internally, the provided function is first written as a linear combination of linear Lagrangian elements, and then numerically integrated with an exact quadrature for those elements.

     Therefore, there is no way to choose the order (degree of exactness) of the quadrature rule. It is also not possible to integrate a function over portions of the domain (properly encoded with a binary matrix).

  .. code-block:: r
     :caption: Numerical integration API

     unit_square <- SquareMesh(c(0, 1), 100)
     ## compute domain measure by integrating the constant 1
     f <- function(points) rep(1, times = nrow(points))
     integrate(f, unit_square) ## outputs 1

     f <- function(p) p[,1]^2 + p[,2]^2
     integrate(f, unit_square) ## outputs 0.667063

     ## compute integral using the FE basis expansion coefficient vector
     integrate(expansion_coeffs, unit_square)


  .. tip::

     This is just for reference. R6 (probably any attempt to recover some form of OOP inside the R language) has some limitations, for instance lack of multiple inheritance, as only linear inheritance is possible, and lack of protected fields (either you set them as public, or use the :code:`object$.__enclos_env__$private` trick). I would recommend to not (ab)use of the inheritance mechanism provided by R6, as it is quite limiting and does not work as one would expect.

     Neverthless, one of the greatest point of R stands in its reflecting capabilities (you can literally change the code of a function while it is running). We can have S3 dispatching on top of the :code:`self` attribute exposes by an :code:`R6` class, which provides the same mechanism of a function overloading but without inheritance. The pattern is depicted in the code below:

     .. code-block:: r
	:caption: S3 dispatching over R6

	method <- function(x, ...) UseMethod("method") ## enable S3 dispatch on method calls
	method.type1 <- function(x, ...) print("type1") 
	method.type2 <- function(x, ...) print("type2")
		  
	Class <- R6::R6Class(
	   "Class",
	   private = list(
	       cpp_backend = NULL ## our "safely" encapsulated cpp module
	   )
	   public = list(
	       initialize = function(...) {
		  ...
	       },
	       call_method = function() method(self) ## reflect on self class to dispatch to method call
	   )
	)

	obj <- Class$new()
	class(obj) <- append("type1", class(obj)) ## exploit the fact that in R types do not really exist...
	obj$call_method() ## prints type1

     In this way, :code:`Class` can be fully generic, and dispatch specific logic to external functions, in pure R style! This is one of the core pattern behind how the statistical models are implemented at R level.
     
..
   * Regression models interface: it is always possible to define and fit a **regression model** using the following 5 lines of code (omitting eventual details (mesh generation/refinement, data import, preprocessing, etc.) which are not responsibility of :code:`fdaPDE`)

     .. code-block:: r

	unit_square <- SquareMesh(c(0, 1), 100)         ## define pyhisical domain
	Vh <- FunctionalSpace(unit_square)              ## defaults to P1 finite elements
	f <- SpatialField(Vh)                           ## define unknown spatial field on pyhisical domain
	model <- SRPDE(y ~ x1 + f, data = problem_data) ## defaults to linear regression with laplacian penalty
	model$fit(lambda = 1e-2)                        ## fit

     
     
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
     auto c = m7.col(4);    // extract the fifth column
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

  .. warning::
     
     The functionality is still considered unstable, as extensive numerical tests for all the supported models are required.
		     
* **Calibrators**: the calibrator concept introduces a unified way to calibrate a statistical model (e.g. select its smoothing parameters). The only requirement for a type T to be a calibrator is to expose a :code:`fit` method with the following signature

  .. code-block:: cpp
     :caption: Calibrator concept fit signature
		  
     template <typename ModelType, typename... Args> DVector<double> fit(ModelType& model, Args&&... args);

  :code:`fit` takes the model whose parameters must be selected and additional arguments required for the specific calibration algorithm. It returns the selected smoothing parameter. Are examples of calibrators, :code:`calibration::KCV` and :code:`calibration::GCV`.

  .. abstract::
     :title: some details on the GCV calibrator
     
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

  .. abstract::
     :title: some details on the KCV calibrator

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
