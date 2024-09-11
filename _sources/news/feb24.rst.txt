:hide-footer:
:hide-toc:

February 2024
=============

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

.. error::
   :title: Deprecated

   What written below has been deprecated on May 2024.

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
