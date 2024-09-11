:hide-footer:
:hide-toc:

May 2024
========

Reached 10K lines of code.

.. success::
   :title: RcppEigen version upgraded to 0.3.4

   :code:`RcppEigen` has been updated to Eigen v3.4.0 (version 0.3.4 of the :code:`RcppEigen` package released on CRAN on 2024-02-28). There are no more restrictions for using Eigen v3.3.9, therefore the whole C++ codebase will upgrade the Eigen version to the 3.4.0. Code will not compile anymore with Eigen v3.3.9 as it already makes use of `slicing, indexing <https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html>`_  and `Eigen support for STL iterators <https://eigen.tuxfamily.org/dox/group__TutorialSTL.html>`_ available only in the 3.4 version.

   The fdaPDE-R package DESCRIPTION and the various CMakeLists.txt files have been updated. femR will need to update its DESCRIPTION file when aligned to the mainstream fdaPDE-core library.

   As a consequence of this upgrade, there are no limitations for using the C++20 standard, as announced in February 2024.

.. danger::
   :title: Warnings are errors

   From this update on, to achieve high-quality code, tests are compiled with the options :code:`-Wall -Wpedantic -Wextra -Werror`. This setting enables the majority of the interesting warnings, and considers warnings as errors (:code:`-Werror` flag). That said, warnings cannot be ignored anymore. This options should help also for a less painfull interface with the CRAN rules.

**core**

* **geometry**: the geometry module is under a major rewriting, and is the first step toward a more general core library system. While work is still in progress, here a list of the already official features:

  .. abstract::
     :title: Nomenclature

     Inside the geometry module, we will refer with *cell* what in finite element analysis is usually named element, that is, depending on the dimensionality of the domain, a *cell* will be a segment (1D, 1.5D), a triangle (2D, 2.5D) or a tetrahedron (3D).

     We will name *node* a vertex of a *cell*. For triangular meshes, the name *edge* refers to a boundary segment of a triangle. For tetrahedral meshes, the name *face* denotes a boundary triangle of a tetrahedron, while the name *edge* refer to a boundary segment of such triangle, i.e. is a 3D segment (to be cristal clear, a tetrahedron has 4 faces and 6 edges).

  * template :code:`Element<int LocalDim, int EmbedDim>` has been removed. :code:`Element` was a generic way to indicate a single segment/triangle/tetrahedron inside a mesh. Its presence was a problem, as the resulting interface was a bit controversial (for instance, a triangle had to define both a concept of face and of edge, which in this case are equivalent). Moreover, it was impossible to work on a simple cell without having an entire mesh structure, which was a bit annoying.

  The following structure has been introduced:

  * **Simplex**: :code:`Simplex<Order, EmbedDim>` represents a generic `simplex <https://en.wikipedia.org/wiki/Simplex>`_ embedded in an :code:`EmbedDim`-dimensional space. The template parameter :code:`Order` sets the order of the simplex, e.g. depdending on its value, :code:`Simplex` represents a segment (:code:`Order` = 1), a triangle (:code:`Order` = 2) or a tetrahedron (:code:`Order` = 3). Higher orders are available, but from a geometrical viewpoint there is no interest in having :code:`Order > 3`. The following geometric operations are available on a simplex:

    .. list-table:: **Simplex API**
       :widths: 40 60

       * - :code:`measure()`
	 - measure of the simplex.
       * - :code:`bounding_box()`
	 - returns the smallest rectangle containing the simplex.
       * - :code:`barycenter()`
	 - returns the midpoint (center of mass) of the simplex.
       * - :code:`barycentric_coords(const NodeType& p)`
	 - given a point :math:`p`, rewrites it in the barycentric coordinate system of the simplex.
       * - :code:`circumcenter()`
	 - returns the center of the (hyper)-sphere passing through the vertices of the simplex.
       * - :code:`circumradius()`
	 - returns the radius of the (hyper)-sphere passing through the vertices of the simplex.
       * - :code:`contains(const NodeType& p)`
	 - given a point :math:`p` determines if the point is inside, on a boundary cell, on a vertex or outside of the simplex.
       * - :code:`supporting_plane()`
	 - the :code:`HyperPlane<Order, EmbedDim>` passing through the simplex.
       * - :code:`normal()`
	 - the normal direction to the simplex (meaningfull only if :code:`Order != EmbedDim`, otherwise returns zero).
       * - :code:`nearest(const NodeType& p)`
	 - given a free point :math:`p`, finds the best approximation of :math:`p` in the simplex (e.g., the point in the simplex which is nearest to :math:`p`).


    :code:`Simplex<Order, EmbedDim>` also provides a :code:`boundary_iterator` type which let iterate over the boundary cells of the simplex as :code:`Simplex<Order - 1, EmbedDim>` simplices (which means that you have all the API above for each boundary cell).

    .. note::

       The :code:`Simplex<Order, EmbedDim>` data type is mainly for internal usage. It is in turn inherited by other high-level concepts. Check the :code:`Triangle`, :code:`Tetrahedron`, etc. types below. The point of :code:`Simplex` is to be a free geometric object, i.e. without any connectivity information related to the presence of an underlying triangulation. Neverthless, having the ability to work on a free geometric object turns out to be usefull in many applications.

    .. code-block:: cpp
       :caption: Simplex API

       // create a 2D triangle (Simplex of order 2, embedded in a 2D space)
       SMatrix<2, 3> vertices;
       vertices <<
	  0.0, 0.0,
	  0.5, 0.0,
	  0.0, 0.8;
       Simplex<2, 2> triangle(vertices);

       triangle.measure();
       triangle.circumcenter();
       // ... see the table above ...

       // compute the perimeter of the triangle using STL algorithms :)
       double p = std::accumulate(
	  triangle.boundary_begin(), triangle.boundary_end(), 0.0,
	  [](double perimeter, auto& f) {
	     return perimeter + f.measure();
	  });

       // explicit for loop over boundary edges
       for(typename Simplex<2, 2>::boundary_iterator it = boundary_begin(); it != boundary_end(); ++it) {
	  it->normal();   // normal direction to edge
	  it->measure();  // length of edge
	  // ... all the standard Simplex API ...
       }

  * **Elementary geometric entities**: introduced templates :code:`Triangle`, :code:`Tetrahedron` and :code:`Segment`.

    .. code-block:: cpp
       :caption: :code:`Triangle`, :code:`Tetrahedron`, :code:`Segment` signature

       template <typename Triangulation> class Triangle    : public Simplex<2, Triangulation::embed_dim>;
       template <typename Triangulation> class Tetrahedron : public Simplex<3, 3>;
       template <typename Triangulation> class Segment     : public Simplex<1, Triangulation::embed_dim>;

    While :code:`Simplex<Order, EmbedDim>` is a free geometric object, each of the above types have no meaning if not bounded to a triangulation, e.g. a :class:`Triangle` exists only as part of a :code:`Triangulation<2, EmbedDim>` object.

    In addition to the whole :code:`Simplex<Order, EmbedDim>` API, the following methods related to connectivity information are available:

    .. list-table:: **Triangle specific API**
       :widths: 40 60

       * - :code:`id()`
	 - the triangle id in the physical triangulation.
       * - :code:`neighbors()`
	 - returns the ids of neighboring triangles.
       * - :code:`node_ids()`
	 - returns the ids of the triangle vertices.
       * - :code:`on_boundary()`
	 - true if the triangle has at least one edge on the triangulation boundary.
       * - :code:`edge(int i)`
	 - returns the i-th edge as an :code:`EdgeType` instance. An :code:`EdgeType` inherits from :code:`Simplex<1, Triangulation::embed_dim>`, i.e. it represents a geometrical segment, and provides the following:
	      * :code:`id()`: the ID of the edge in the physical triangulation.
	      * :code:`node_ids()`: the ids of the nodes composing the edge.
	      * :code:`adjacent_cells()`: the ids of the cells (i.e., triangles) which share this edge.
	      * :code:`on_boundary()`: true if the edge is on boundary.
       * - :code:`edges_begin()/edges_end()`
	 - first and last iterator to triangle edges.

    :code:`Tetrahedron` exposes a simlar interface, with the addition of iterators and access methods to each face of the tetrahedron. Check the :code:`Triangulation` code snippet below for a detailed exposition of the API.

  * **Triangulation**: template :code:`Mesh<int LocalDim, int EmbedDim>` has been removed and replaced by the new :code:`Triangulation<int LocalDim, int EmbedDim>` type.

    .. code-block:: cpp
       :caption: Triangulation signature

       template <int LocalDim, int EmbedDim> class Triangulation;

    .. tip::

       The main point of the :code:`Triangulation` data type is the exposition of iterators to logically iterate over different parts of the mesh. Moreover, the provided iterators are compliant to the :code:`std::bidirectional_iterator` concept, which let use any STL algorithm over :code:`Triangulation` iterator ranges. For instance, it is straightforward to compute the measure of the border:

       .. code-block:: cpp

	  // just one line, for a complex operation :)
	  std::accumulate(mesh.boundary_edges_begin(), mesh.boundary_edges_end(), 0.0,
	     [](double p, auto& cell) { return p + cell.measure(); }
	  );

       .. note::

	  This let us easily parallelize different operations, once we link fdaPDE to the `oneTBB <https://oneapi-src.github.io/oneTBB/>`_ library shipped by Intel, thanks to the STL support for parallel algorithms. It is immediate, for instance, to compute the measure of the mesh by a parallel :code:`std::reduce`.

    Check the code snippet below for a detailed description of the exposed API:

    .. code-block:: cpp
       :caption: Triangulation API

       // a planar triangulation (the API below is also available for triangulated surfaces)
       Triangulation<2, 2> mesh(nodes, cells, boundary);
       // Triangulation<2, 2> will compute for you the edges, the neighboring structure and
       // other connectivity informations

       mesh.cell(i);   // request i-th cell as a Triangle<2, 2>
       mesh.node(i);   // request i-th node

       // a lot of other informations, such as matrix of edges, neighbors, etc. number of edges, cells,
       // boundary edges and boundary nodes, triangulation bounding box, etc.
       // check the source code for more details!

       // iterators
       for(typename Triangulation<2, 2>::cell_iterator it = mesh.cells_begin();
	   it != mesh.cells_end(); ++it) {
	   // all the interface of Triangle<2, 2> is accessible from the iterator it
	   it->measure();      // measure of triangle
	   it->circumcenter(); // circumcenter of triangle
	   // ...
       }

       std::for_each(mesh.cells_begin(), mesh.cells_end(), [](auto& cell) {
	   // whatever complex operation on your cell (even the assembly of a differential weak form)
       });
       // and the above can be paralellized thanks to the STL :)

       // cycle over the border
       for(typename Triangulation<2, 2>::boundary_edge_iterator it = mesh.boundary_edges_begin();
	   it != mesh.boundary_edges_end(); ++it) {
	   // all the interface of Triangle<2, 2>::EdgeType available from the iterator

	   it->measure();        // length of edge
	   it->barycenter();     // mid-point of the edge
	   it->adjacent_cells(); // the id of the triangle sharing this edge
	   // ...
       }

       // you can also iterate on
       // - the whole set of edges with: edges_begin() / edges_end()
       // - the boundary nodes with    : boundary_nodes_begin() / boundary_nodes_end()

       DVector<int> ids = mesh.locate(locs);               // O(log(n_locs)) point location
       std::unordered_set<int> patch = mesh.node_patch(n); // ids of all triangles having n as vertex

       // 3D triangulation (tetrahedralizations)
       Triangulation<3, 3> mesh(nodes, cells, boundary);

       // all the interface of a Triangulation<2, 2> is available, with the additional capability
       // of indexing and iterating over the faces of each tetrahedron

       for(typename Triangulation<3, 3>::boundary_face_iterator it = mesh.boundary_faces_begin();
	   it != mesh.boundary_faces_end(); ++it) {
	   // all the interface of Tetrahedron<3, 3>::FaceType (i.e., a 3D triangle with connectivity infos)
	   // available from the iterator it

	   it->normal();   // normal direction to the face
	   it->measure();  // area of the triangle
	   it->edge_ids(); // global ids in the 3D triangulation of the edges making this triangle

	   // you can in turn cicle on each 3D edge of the current boundary face it
	   for(auto jt = it->edges_begin(); jt != it->edges_end(); ++jt) { ... }
	      // jt is a Simplex<1, 3> exposing also
	      jt->on_boundary(); // well, this is true, we are iterating over the border :)
	      jt->id();          // id of the edge in the 3D triangulation
	      jt->node_ids();    // ids of nodes composing the edge
	   }
       }

       // for a 3D mesh, you can get also its surface mesh as a Triangulation<2, 3> instance
       Triangulation<2, 3> surface = mesh.surface();

       double p1 = std::accumulate(mesh.boundary_faces_begin(), mesh.boundary_faces_end(), 0.0,
	  [](double s, auto& f) { return s + f.measure(); }
       );
       double p2 = std::accumulate(surface.cells_begin(), surface.cells_end(), 0.0,
	  [](double s, auto& f) { return s + f.measure(); }
       );
       // and not surprisingly, p1 == p2 :)

  * **Projection**: template class :code:`Projection<TriangulationType>` implements an exact and a non-exact method for point projection over a :code:`Triangulation<LocalDim, EmbedDim>`. Given a free point :math:`p`, the algorithm searches for the best approximation of :math:`p` in the triangulation. In this sense, it works both for manifold and non-manifold domains (in the last case, the algorithm returns the point on the 2D/3D triangulation border which is nearest to :math:`p`).

    Computational complexity: let :math:`N` be the number of cells and :math:`n` the number of points to project.

    * Exact version is :math:`O(nN)`.
    * Assuming :math:`n \gg N` (number of points to project much larger than number of cells), approximate version is :math:`O(n \log(N))` (it was :math:`O(nN)` in :code:`fdaPDE-1.1-17`).

    .. info::

       The approximate algorithm computes a :code:`KDTree` for fast locating the nearest mesh node to :math:`p` (with computational cost :math:`O(N \log(N))`. This cost is negligible if :math:`n \gg N`, and performed just once ). Once the nearest point :math:`q` is found, a search restricted in the patch of :math:`q` is performed, i.e., in the set of cells having :math:`q` as vertex (computing the patch costs :math:`O(\log(N))`), therefore avoiding a brute force search over the entire mesh (which would cost :math:`O(N)`). This is an approximate approach since for highly non-convex region the computed point might be not the nearest to :math:`p`.

    .. code-block:: cpp
       :caption: Projection API

       Triangulation<2, 3> surface(nodes, cells, boundary);

       DMatrix<double> points; // free points in 3D space
       // perform projection (use C++ automatic template deduction + tag dispatching)
       DMatrix<double> proj_points = Projection(surface)(points, fdapde::Exact);
       DMatrix<double> proj_points = Projection(surface)(points, fdapde::NotExact);

       // NotExact projection requires a O(N log(N)) initialization, this is done just once
       // at first call. You can create a Projection instance and cache
       Projection<Triangulation<2, 3>> project(surface);
       project(points, fdapde::NotExact); // silent initialization here
       project(points, fdapde::NotExact); // just perform fast approximate projection

       project(points); // defaults to approximate algorithm

* **Minor changes**:

  * **Optimizers support for objective stopping criterion callback**: if the objective functor provided to the optimizer exposes a method with the following signature:

    .. code-block::

       template <typename OptimizerType> bool opt_stopping_criterion(OptimizerType& opt);

    any optimizer in the optimization module will execute it to evaluate if convergence has been reached. Users of the optimization module can hence define objective functions with a custom stopping criterion (see, e.g., density estimation in fdaPDE-cpp).

  * **Binary matrix**: binary matrices have proved to be extremely usefull for handling bitmasks and are getting more and more used in the codebase. The following additional methods are now exposed:

    .. code-block:: cpp

       int n = 50, m = 50;
       BinaryMatrix<Dynamic> bitmask(n, m);

       // writable block expressions
       bitmask.block(0, 0, 10, 10) = BinaryMatrix<Dynamic>::Ones(10, 10);
       // reshape operation
       bitmask.reshape(25, 100); // reshape bitmask to a 25 x 100 matrix (no-cost operation)
       bitmask.vector_view();    // linearize the bitmask into a column vector (no-cost operation)

       // returns all the indexes set to true/false in the bitmask: O(nm) operation
       bitmask.which(true);  // which true?
       bitmask.which(false); // which false?
       which(bitmask);       // implicitly returns true indexes (R style)

**cpp**

* **Density estimation**: official support for density estimation models. Below the API exposed by the :code:`DensityEstimationBase` core class for the density module:

  .. list-table:: **DensityEstimationBase API**
     :widths: 40 60

     * - :code:`n_obs()`
       - number of active data locations. A data location is active if is it not masked.
     * - :code:`n_locs()`
       - number of all data locations (e.g., the overall number of observations). It coincides with :code:`n_obs()` if no observation is masked.
     * - :code:`Psi()`
       - the matrix :math:`\Psi` (evaluation of spatial basis functions at data location) for space-only problems, the matrix :math:`\Upsilon` for space-time problems (as defined in *Begu, B., Panzeri, S. (2022), Space-Time Density Estimation with Partial Differential Equation Regularization. PACS project. Pag 9.*)
     * - :code:`Upsilon()`
       - if some observation is masked, returns the matrix provided by :code:`Psi()` where rows corresponding to masked observations are set to zero, otherwise is equivalent to calling :code:`Psi()`. Upper models should mainly interact with this method, instead of directly calling :code:`Psi()`
     * - :code:`PsiQuad()`
       - the matrix of evaluations of the reference basis system at the quadrature nodes. Already tensorized for space-time problems.
     * - :code:`w()`
       - weights of the quadrature rule used for the approximation of :math:`\int_{\Omega} e^g`. Already tensorized for space-time problems.
     * - :code:`int_exp(const DVector<double>& g)`
       - evaluation of :math:`\int_{\Omega} e^g`
     * - :code:`int_exp()`
       - evaluation of :math:`\int_{\Omega} e^{\hat g}`, where :math:`\hat g` is the current estimation of the log density field.
     * - :code:`grad_int_exp(const DVector<double>& g)`
       - evaluation of the gradient of :math:`\int_{\Omega} e^g`
     * - :code:`grad_int_exp()`
       - evaluation of the gradient of :math:`\int_{\Omega} e^{\hat g}`, where :math:`\hat g` is the current estimation of the log density field.
     * - :code:`g()`
       - expansion coefficient vector of the log density field.
     * - :code:`f()`
       - expansion coefficient vector of the density field, e.g. :math:`f = e^g`.
     * - :code:`masked_obs()`
       - :code:`BinaryVector<Dynamic>` of masked observations.

  As always, in addition, a model inheriting from :code:`DensityEstimationBase` has access to the specific API induced by the choosen regularization. Check the source code for details.

  .. info:: The masking mechanism

     This is something already shown in February 2024. At the statistical level, "masking" means to remove some observations (the masked ones) from the fitting. This corresponds to set to zero all the rows of the matrix :math:`\Psi` (or :math:`\Upsilon`) corresponding to masked observations. This mechanism is used, for instance, by the :code:`KCV` class to perform CV selection of the smoothing parameters.

     By doing so, all models inheriting from :code:`DensityEstimationBase` have immediate support for smoothing parameter selection by K-fold Cross Validation.

  Class :code:`DEPDE` implements the density estimation model shown in *Ferraccioli, F., Arnone, E., Finos, L., Ramsay, J.O., Sangalli, L.M. (2021), Nonparametric density estimation over complicated domains, Journal of the Royal Statistical Society* (space-only) and *Begu, B., Panzeri, S., Arnone, E., Carey, M., and Sangalli, L.M. (2024), A nonparametric penalized likelihood approach to density estimation of space-time point patterns, Spatial Statistics* (space-time).

  .. code-block:: cpp

     template <typename RegularizationType_>
     class DEPDE : public DensityEstimationBase<DEPDE<RegularizationType_>, RegularizationType_> { ... };

  Because the resolution strategy for a density estimation model is a penalized log-likelihood minimization, :class:`DEPDE` exposes an interface compatible with the optimization module, i.e. it acts exactly as an optimizer objective function (see the optimization module API for details). 

  .. list-table:: **DEPDE API**
     :widths: 40 60

     * - :code:`operator(const DVector<double>& g)`
       - evaluates the penalized log-likelihood functional at :math:`g`, i.e. computes :math:`L(g) = - 1^\top \Upsilon g + \int_{\Omega} e^g + g^\top P_{\lambda} g`	 
     * - :code:`derive()`
       - returns a :code:`std::function<DVector(const DVector<double>&)>` encoding the gradient of the penalized log-likelihood functional.
     * - :code:`bool opt_stopping_criterion(OptimizerType&)`
       - stops the optimization algorithm if the relative difference between the log-likelihood or the penalized log-likelihood is below a user defined tolerance (defaults to :math:`10^{-5}`).
     * - :code:`set_tolerance(double)`
       - sets the tolerance for the custom stopping criterion (the one triggered by :code:`opt_stopping_criterion()`).
     * - :code:`void set_g_init(const DMatrix<double>&)`
       - sets the initial log-density expansion coefficient vector.

	 .. note::

	    :code:`DEPDE` does not compute any initialization density (e.g., by heat-process). Instead, it requests the initialization point, which must be externally computed.
     * - :code:`void set_optimizer(OptimizerType&&)`
       - sets the optimization algorithm for the minimization of the penalized log-likelihood functional :math:`L(g)`. The optimizer is internally type-erased.
     * - :code:`void init()`
       - initializes the model stack.
     * - :code:`void solve()`
       - triggers the optimizer for the minimization of the penalized log-likelihood.



  Check the code snippet below for the provided API:

  .. example::

     .. code-block:: cpp
	:caption: DEPDE API

	// assume mesh and laplacian penalty already defined...

	// space-only model
	DEPDE<SpaceOnly> model(penalty);
	model.set_lambda_D(0.1);

	// data in point-pattern processes coincide with locations
	BlockFrame<double, int> df;
	df.insert(SPACE_LOCS, ...);

	model.set_tolerance(1e-5); // set tolerance on custom stopping criterion
	model.set_data(df);

	// set optimization algorithm (here you have access to the whole optimization API)
	int max_iter = 500;
	double opt_tolerance = 1e-5; // set optimizer tolerance (looks for the norm of the objective gradient)
	double step = 1e-2;
	model.set_optimizer(BFGS<fdapde::Dynamic> {max_iter, opt_tolerance, step}); // optimizer must be Dynamic

	// gradient descent with adaptive step	
	model.set_optimizer(GradientDescent<fdapde::Dynamic, BacktrackingLineSearch> {max_iter, opt_tolerance, step});
	// in general, you can set any optimization algorithm in the optimization module

	// initialize and solve
	model.set_g_init(...); // optimization algorithm init point
	model.init();
	model.solve();

	model.g(); // estimated log-density field

	// you can also approach the fitting as a pure optimization problem (emphasis on the optimizer)

	BFGS<fdapde::Dynamic, WolfeLineSearch> optimizer(max_iter, opt_tolerance, step);
	optimizer.optimize(model, g_init);
	optimizer.optimum(); // estimated log-density field

	// and by just changing the RegularizetionType template, you get space-time :)
	DEPDE<SpaceTimeSeparable> model(penalty_space, penalty_time);
	// all the API above stays valid

     .. code-block:: cpp
	:caption: DEPDE KFold-CV API

	// assume mesh and laplacian penalty already defined...

	DEPDE<SpaceOnly> model(penalty);
	model.set_data(df);
	model.set_optimizer(BFGS<fdapde::Dynamic, WolfeLineSearch> {max_iter, opt_tolerance, step});

	int n_folds = 10;
	int seed = fdapde::random_seed;
	KCV kcv(n_folds, seed);

	DMatrix<double> lambda_grid; // the grid of smoothing parameters to explore
	DMatrix<double> g_init_grid; // for each value of lambda, the initial density field (computed in some way)
	model.set_g_init(g_init_grid);

	// calibrate the model
	kcv.fit(model, lambda_grid); // uses DEPDE::CVScore scoring function, see below for details

	// at the end you get
	kcv.avg_scores();
	kcv.std_scores();
	kcv.optimum(); // optimal smoothing parameter

     .. info::

	:code:`DEPDE` internally defines its cross-validation scoring index as a functor of type :code:`DEPDE::CVScore`, exposing a call operator compatible with the :code:`KCV` requirement. :code:`DEPDE::CVScore` implements Equation (1.18) of *Begu, B., Panzeri, S. (2022), Space-Time Density Estimation with Partial Differential Equation Regularization. PACS project. Pag 17.*

     .. tip::

	When a model of type :code:`ModelType` exposes a public type :code:`ModelType::CVScore`, calling :code:`KCV::fit(model, lambda_grid)` fallbacks to the use of :code:`ModelType::CVScore` as cross validation index (rises a static assert otherwise). Specifically, :code:`ModelType::CVScore` must expose a constructor with the following signature:

	.. code-block:: cpp

	   CVScore(ModelType& model);

	and expose a call operator

	.. code-block:: cpp

	   double operator()(
	      int fold, const DVector<double>& lambda,
	      const BinaryVector<fdapde::Dynamic>& train_mask, const BinaryVector<fdapde::Dynamic>& test_mask);

**R (base)**

Be prepared, almost ready (at least on paper).

..
   * Support for the definition of general differential operators. Given a :code:`SymbolicFunction`, it is now possible to write in :code:`R` any kind of differential operator. Check the code below.

     .. code-block:: r
	:caption: Writing of differential operators in R

	## create domain
	n <- 10
	unit_square <- SquareMesh(c(0, 1), n)
	## Linear finite elements space over the square
	Vh <- FunctionalSpace(unit_square)

	## define a mathematical function over Vh
	f <- Function(Vh) ## f is an instance of a SymbolicFunction class

	## follow some examples of valid operators

	K <- matrix(c(1,1,1,1), nrow = 2, ncol = 2) ## diffusion tensor
	Lf <- -div(K*grad(f))

	## you can provide coefficients which are not bound to any variable, as the transport term below
	Lf <- -div(K*grad(f)) + inner(c(1,1), grad(f))

	## you can essentially type whatever operator you want
	Lf <- dt(f) - div(K*grad(f)) - alpha*f*(1-f)           ## Fisher KPP
	Lf <- -div(f*grad(f))                                  ## non linear diffusion
	Lf <- -div(f*(1-f)*grad(f)) + inner(b, grad(f)) + f*f  ## probably nonsense (but accepted) equation

     .. info::
	:title: Implementation details

	The key point is to move R code into something which can be later inspected by other code (either written in R or a different language). This is achieved by transforming the input operator into a string. Once we have a string representation, we can build whatever kind of parsing logic for it. Check the code below to understand how the R wrapper internally sees an operator.

	.. code-block:: r

	   ## you input this code
	   f <- Function(Vh)
	   K <- matrix(c(1,1,1,1), nrow = 2, ncol = 2) ## diffusion tensor
	   Lf <- -div(K*grad(f))

	   ## internal representation of the operator

	   ## Symbolic Expression:
	   ## - div(<22qls1tc5n> * grad(<d9tu9y3gxf>))
	   ##
	   ## Symbols Table:
	   ## <22qls1tc5n>:
	   ##      [,1] [,2]
	   ## [1,]   -1   -1
	   ## [2,]   -1   -1
	   ## <d9tu9y3gxf> : <SymbolicFunction>	

	* The first thing to notice is that there is no point in knowing the exact names of the variables. What is important is the structure of the expression, and the value bounded to each symbol. Notice also that, for values not bounded to any variable (as is the case for the vector :code:`c(1,1)` in :code:`inner(c(1,1), grad(f))`) and for the SymbolicFunction :code:`f` itself, we either have no name or we cannot know it (indeed, in the assignment :code:`f <- Function(Vh)`, there is no possibility to provide to :code:`Function(...)` the simbol :code:`f` to which its output is assigned, as :code:`f` is out of scope for :code:`Function(...)`. In addition, :code:`<-` is not an S3 method, so that we cannot overload it for specific class instances). For these reasons, we get different :code:`<random_string>` names in the internal operator representation.

	* A symbol table bounds each unique :code:`<random_string>` identifier to its value. This is clearly necessary, as, if we ask for the diffusion tensor of the operator, we must have the possibility to recover the actual value bounded to :code:`K`. This task can be achieved by searching patterns matching the regular expression :code:`div\\\\(<(.*)> \\\\* grad\\\\(<(.*)>\\\\)\\\\)` in the symbolic expression. Once recovered the identifier associated to the diffusion tensor, we get the associated value from the symbol table.

	.. code-block:: r
	   :caption: Fisher KPP with space-varying diffusion tensor

	   ## you input this code
	   f <- Function(Vh)
	   K <- function(p) matrix(c(2*p[,1], 2*p[,1], 2*p[,1], 2*p[,1]), ncol = 4)
	   alpha <- 0.4
	   Lf <- dt(f) - div(K*grad(f)) + alpha*f*(1-f)

	   ## internal representation of the operator

	   ## Symbolic Expression:
	   ## dt(<mvednubbdp>) - div(<yhei1mod67> * grad(<mvednubbdp>)) + <9z2rklf28l> * <mvednubbdp> * (<hn9c2om7d8> - <mvednubbdp>)
	   ##
	   ## Symbols Table:
	   ## <mvednubbdp> : <SymbolicFunction>
	   ## <yhei1mod67> : function(p) matrix(c(2*p[,1], 2*p[,1], 2*p[,1], 2*p[,1]), ncol = 4)
	   ## <9z2rklf28l> : 0.4
	   ## <hn9c2om7d8> : 1

	Non linearities are more difficult to manage, as the grammar generated by regular expressions is not powerful enought to recognize them in their generality. Neverthless, we can pattern match for very specific types of nonlinearities using regular expressions (i.e., to support a finite set of known nonlinearities).

	.. quote::

	   *This is a personal note: for a serious non-linear operator support, we should drop the strong formulation in favor of the weak one, or, at least, support both. Neverthless, this might be too early, and too much, for the 2.X series of the library. This will be the most limiting point of the R (and python) package interface for the entire 2.X series (the C++ layer can, and will, instead support whatever we want). The only solution can be provided by an intermediate Domain Specific Language (DSL), parsed by an* `LL(1) parser <https://en.wikipedia.org/wiki/LL_parser>`_, *and compiled on-the-fly into executable code. But this is stuff for the 3.0 version of fdaPDE.*
