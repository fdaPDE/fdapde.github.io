:hide-footer:
:hide-toc:

Example 2: The Heat equation
============================

In this example we show how to use the fdaPDE-core library to build a numerical scheme for the solution of the heat equation

.. math::

   \begin{aligned}
   &\frac{\partial}{\partial t} u(\boldsymbol{x}, t) - \Delta u(\boldsymbol{x}, t) &&= f(\boldsymbol{x}, t) && \qquad \boldsymbol{x} \in \Omega, t > 0 \\
   &\frac{\partial}{\partial \boldsymbol{n}}u(\boldsymbol{x}, t) &&= g_N(\boldsymbol{x}, t) && \qquad \boldsymbol{x} \in \Gamma_N, t > 0 \\
   &u(\boldsymbol{x}, t) &&= g_D(\boldsymbol{x}, t) && \qquad \boldsymbol{x} \in \Gamma_D, t > 0 \\
   &u(\boldsymbol{x}, 0) &&= u_0(\boldsymbol{x}) && \qquad \boldsymbol{x} \in \Omega
   \end{aligned}

where :math:`\Gamma_D, \Gamma_N \in \partial \Omega` are such that :math:`\Gamma_D \cup \Gamma_N = \partial \Omega`, :math:`g_D(\boldsymbol{x}, t), g_N(\boldsymbol{x}, t)` denote the Dirichlet and Neumann boundary data respectively and :math:`u_0(\boldsymbol{x})` represents the given initial condition. :math:`\boldsymbol{n}` is the outward normal vector to :math:`\partial \Omega`.

As always, we start by deriving the weak formulation, by multiplying for each :math:`t > 0` the differential equation by a test function :math:`v \in V = H^1_{\Gamma_D}(\Omega)` and integrating on the spatial domain only :math:`\Omega`. Specifically, for each :math:`t > 0` we seek :math:`u(t) \in V` such that

.. math::

   \int_{\Omega} \frac{\partial}{\partial t} u(\boldsymbol{x}, t) v - \Delta u(\boldsymbol{x}, t)v = \int_{\Omega} f(\boldsymbol{x}, t) v \qquad \forall v \in V

Integrating by parts the Laplacian leads to the weak formulation

.. math::

   \int_{\Omega} \frac{\partial}{\partial t} u(\boldsymbol{x}, t) v - \nabla u(\boldsymbol{x}, t) \nabla v = \int_{\Omega} f(\boldsymbol{x}, t) + \int_{\Gamma_N} \frac{\partial}{\partial \boldsymbol{n}} u(\boldsymbol{x}, t) v \qquad \forall v \in V

As usual, the Galerkin approach considers a finite dimensional space :math:`V_h \subset V` and writes :math:`u(\boldsymbol{x}, t) = \sum_{j=1}^N u_j(t) \psi_j(\boldsymbol{x})`. Substituting in the weak formulation, computations lead to the following semi-discretization

.. math::

   \sum_{j=1}^N \Biggl[ \dot{c_j}(t) \int_{\Omega} \psi_i(\boldsymbol{x}) \psi_j(\boldsymbol{x}) + c_j(t) \int_{\Omega} \nabla \psi_i(\boldsymbol{x}) \cdot \nabla \psi_j(\boldsymbol{x}) \Biggr] = \int_{\Omega} f(\boldsymbol{x}, t) \psi_j(\boldsymbol{x}) + \int_{\Gamma_N} g(\boldsymbol{x}, t) \psi_i(\boldsymbol{x}) \qquad \forall i = 1, \ldots, N

Let :math:`M = [m_{ij}] = \int_{\Omega} \psi_i \psi_j` be the mass matrix, :math:`A = [a_{ij}] = \int_{\Omega} \nabla \psi_i \cdot \nabla \psi_j` be the stiff matrix and :math:`\boldsymbol{F}(t) = \int_{\Omega} f(\boldsymbol{x}, t) \psi_j(\boldsymbol{x}) + \int_{\Gamma_N} g(\boldsymbol{x}, t) \psi_i(\boldsymbol{x})` the discretization of the righ hand side of the above relation, we are asked to solve in :math:`\boldsymbol{u}(t)` the following systems of ODEs

.. math::

   M \boldsymbol{\dot{u}}(t) + A \boldsymbol{u}(t) = \boldsymbol{F}(t)

We decide to solve it via the Crank-Nicolson method, which is a second order method in the time step size :math:`\Delta t`. Specifically we get

.. math::

   M \frac{\boldsymbol{u}^{k+1} - \boldsymbol{u}^k}{\Delta t} + \frac{1}{2} A ( \boldsymbol{u}^{k+1} + \boldsymbol{u}^k) = \frac{1}{2} ( \boldsymbol{F}^{k+1} + \boldsymbol{F}^k)

The above is solved iteratively for all the time steps :math:`k = 1, \ldots, N_T`. Follows a step by step description of the program which enables us to find a solution to the considered problem in less than 50 lines of code. For the first example we consider homogeneous Neumann and Dirichlet conditions on the domain boundary and a space-time depending forcing term.

We initially load the geometry, define a finite element space and the Laplacian bilinear form (observe that (the bilinear form of) any elliptic operator could have been placed here, in case of general parabolic equations).

.. code-block:: cpp

   Triangulation<2, 2> unit_square = read_mesh<2, 2>("../data/mesh/unit_square", cache_cells);   // import mesh
   FiniteElementSpace Vh(unit_square, P1);
   // create trial and test functions
   TrialFunction u(Vh);
   TestFunction  v(Vh);;
   // define bilinear form for laplacian operator
   auto a = integral(unit_square)(dot(grad(u), grad(v)));

We then define the forcing term. Because we are dealing with a space-time problem, we use the specialized :code:`SpaceTimeField` type, which extends the :code:`ScalarField` capabilities to handle an explicit time coordinate. Specifically, the code above implements the following forcing term

.. math::

   f(\boldsymbol{x}, t) = \begin{cases} 1 & \qquad x < 0.5, y < 0.5, 0.0 \leq t \leq 0.2 \\ 1 & \qquad x > 0.5, y > 0.5, 0.2 \leq t \leq 0.4 \\ 0 & \qquad \text{otherwise} \end{cases}

.. code-block:: cpp

   // define forcing functional
   SpaceTimeField<2, decltype([](const SVector<2>& p, double t) {
       if        (p[0] < 0.5 && p[1] < 0.5 && t >= 0.0 && t <= 0.2) {
           return 1;
       } else if (p[0] > 0.5 && p[1] > 0.5 && t >= 0.2 && t <= 0.4) {
           return 1;
       } else {
           return 0;
       }	 
   })> f;

You can fix the time coordinate calling :code:`f.at(t)`. Subsequent calls of :code:`f` at a spatial point will act as calls to a :code:`ScalarField` where the time coordinate has been fixed to :code:`t`. Next we proceed to fix homogeneous boundary conditions on all the domain's boundary

.. code-block:: cpp

   ScalarField<2, decltype([](const SVector<2>& p) { return 0; })> g;
   DofHandler<2, 2>& dof_handler = Vh.dof_handler();
   dof_handler.set_dirichlet_constraint(/* on = */ BoundaryAll, /* data = */ g);

Finally, we fix the time step :math:`\Delta t`, set up room for the solution fixing the initial condition to :math:`u_0(\boldsymbol{x}) = 0` and discretizing once and for all the mass matrix :math:`M` and the stiff matrix :math:`\frac{M}{\Delta T} + \frac{1}{2} A`, togheter with the forcing term :math:`F(t)`. Since the matrix :math:`\frac{M}{\Delta T} + \frac{1}{2} A` is SPD and time-invariant, we factorize it outside the time integration loop using a Cholesky factorization:

.. code-block:: cpp
		
   SpMatrix<double> M = integral(unit_square)(u * v).assemble();    // mass matrix
   SpMatrix<double> A = M / DeltaT + a.assemble() * 0.5;            // stiff matrix (SPD)

   // discretize time-dependent forcing field
   DMatrix<double> F(dof_handler.n_dofs(), n_times);
   for (int i = 0; i < n_times; ++i) { F.col(i) = integral(unit_square)(f.at(DeltaT * i) * v).assemble(); }

   dof_handler.enforce_constraints(A);    // set dirichlet constraints
   Eigen::SimplicialLLT<SpMatrix<double>> lin_solver(A);

   
Finally, the crank-nicolson time integration loop can start:

.. code-block:: cpp

   for (int i = 1; i < n_times; ++i) {
       DVector<double> b =
           (M / DeltaT - A / 2) * solution.col(i - 1) + 0.5 * (F.col(i) + F.col(i - 1));   // update rhs
       dof_handler.enforce_constraints(b);
       solution.col(i) = lin_solver.solve(b);
   }
   
.. abstract:: The complete script

   .. code-block:: cpp
      :linenos:

      #include <fdaPDE/fields.h>
      #include <fdaPDE/geometry.h>
      #include <fdaPDE/finite_elements.h>

      using namespace fdapde;
      
      int main() {
         Triangulation<2, 2> unit_square = read_mesh<2, 2>("../data/mesh/unit_square", cache_cells);   // import mesh
	 FiniteElementSpace Vh(unit_square, P1);
	 // create trial and test functions
	 TrialFunction u(Vh);
	 TestFunction  v(Vh);;
	 // define bilinear form for laplacian operator
	 auto a = integral(unit_square)(dot(grad(u), grad(v)));

	 // define forcing functional
	 SpaceTimeField<2, decltype([](const SVector<2>& p, double t) {
	     if        (p[0] < 0.5 && p[1] < 0.5 && t >= 0.0 && t <= 0.2) {
	         return 1;
	     } else if (p[0] > 0.5 && p[1] > 0.5 && t >= 0.2 && t <= 0.4) {
	         return 1;
	     } else {
	         return 0;
	     }	 
         })> f;

	 // dirichlet data (homogeneous and fixed in time)
	 ScalarField<2, decltype([](const SVector<2>& p) { return 0; })> g;
	 DofHandler<2, 2>& dof_handler = Vh.dof_handler();
	 dof_handler.set_dirichlet_constraint(/* on = */ BoundaryAll, /* data = */ g);

	 double T = 0.5, DeltaT = 0.02;
	 int n_times = std::ceil(T/DeltaT);
	 DMatrix<double> solution(dof_handler.n_dofs(), n_times);
	 solution.col(0) = DVector<double>::Zero(dof_handler.n_dofs());   // zero initial condition

	 // crank-nicolson integration
	 SpMatrix<double> M = integral(unit_square)(u * v).assemble();    // mass matrix
	 SpMatrix<double> A = M / DeltaT + a.assemble() * 0.5;            // stiff matrix (SPD)
	 dof_handler.enforce_constraints(A);
	 Eigen::SimplicialLLT<SpMatrix<double>> lin_solver(A);
	 // discretize time-dependent forcing field
	 DMatrix<double> F(dof_handler.n_dofs(), n_times);
	 for (int i = 0; i < n_times; ++i) { F.col(i) = integral(unit_square)(f.at(DeltaT * i) * v).assemble(); }
    
	 for (int i = 1; i < n_times; ++i) {
             DVector<double> b =
                 (M / DeltaT - A / 2) * solution.col(i - 1) + 0.5 * (F.col(i) + F.col(i - 1));   // update rhs
	     dof_handler.enforce_constraints(b);
	     solution.col(i) = lin_solver.solve(b);
	 }
	 return 0;
      }

.. image:: heat.gif
   :width: 400
   :align: center

We here report a slight variation of the problem above, where we consider a null-forcing term, but a non-homogeneous time-dependent Neumann condition on the left side of the square, while we impose a zero dirichlet condition on the remaining part of the boundary.

.. tip::

   We pose the attention on the mechanism which enables us to define the different portions of the domain's boundary :math:`\Gamma_D` and :math:`\Gamma_N`. Specifically, every boundary element of the geometry can be associated to a numerical non-negative marker, so that, boundary elements with the same marker contributes to the definition of the same boundary subset :math:`\Gamma \subset \partial \Omega`.

   Each :code:`Triangulation` object starts with an empty set of boundary markers (in this case, we name the boundary elements which have no marker as :code:`Unmarked`). You can fix a value for the boundary markers using the :code:`mark_boundary()` method of a :code:`Triangulation` instance. For instance, to fix all the markers of the triangulation to 0, just use

   .. code-block:: cpp

      unit_square.mark_boundary(/* as = */ 0);    // mark all nodes as zero

   You can use a geometric predicate to obtain a more selective marking. In the considered example, we can mark only the left side of the unit square :math:`[0,1]^2` by setting the marker of all the edges on the left side to 1 using the following

   .. code-block:: cpp

      unit_square.mark_boundary(/* as = */ 1, /* where = */ [](const typename Triangulation<2, 2>::EdgeType& edge) {
          return (edge.node(0)[0] == 0 && edge.node(1)[0] == 0);     // select only edges on the left side
      });

   Be aware that **markers with higher values have higher precedence on markers with lower values**, that is, markers with higher values will overwrite existing markers with lower values, the viceversa is not true.

   Once you have fixed the markers, you can iterate on all the boundary elements having a fixed marker using the overload of the :code:`boundary_begin()` and :code:`boundary_end()` methods taking the marker as parameter. For instance, to iterate over :math:`Gamma_N` only (assuming being identified with marker 1) you execute:

   .. code-block:: cpp

      for(auto it = unit_square.boundary_begin(1); it != unit_square.boundary_end(1); ++it) {
          // all and only the boundary edges marked as 1
      }

The script is mostly similar to the Crank-Nicolson time-stepping scheme implemented before, apart for the definition of the boundary conditions and the introduction of the non-homogeneous neumann boundary term at line 46

.. code-block:: cpp

   integral(unit_square.boundary(/* on = */ 1))(g_N.at(DeltaT * i) * v)
      
.. abstract:: The complete script

   .. code-block:: cpp
      :linenos:

      #include <fdaPDE/fields.h>
      #include <fdaPDE/geometry.h>
      #include <fdaPDE/finite_elements.h>

      using namespace fdapde;
      
      int main() {
         Triangulation<2, 2> unit_square = read_mesh<2, 2>("../data/mesh/unit_square", cache_cells);   // import mesh
	 // label boundary
	 unit_square.mark_boundary(/* as = */ 0);    // mark all nodes as zero
	 // mark left side of square (where we will impose non-homegenous Neumann BCs) with 1
	 unit_square.mark_boundary(/* as = */ 1, /* where = */ [](const typename Triangulation<2, 2>::EdgeType& e) {
	     return (e.node(0)[0] == 0 && e.node(1)[0] == 0); 
	 });
	 
	 FiniteElementSpace Vh(unit_square, P1);
	 // create trial and test functions
	 TrialFunction u(Vh);
	 TestFunction  v(Vh);;
	 // define bilinear form for laplacian operator
	 auto a = integral(unit_square)(10 * dot(grad(u), grad(v)));

	 // define forcing functional (this could have been omitted, but placed here just for completeness)
	 ScalarField<2, decltype([](const SVector<2>& p) { return 0; })> f;

	 // dirichlet homoegeneous data (fixed in time)
	 ScalarField<2, decltype([](const SVector<2>& p) { return 0; })> g_D;
	 DofHandler<2, 2>& dof_handler = Vh.dof_handler();
	 dof_handler.set_dirichlet_constraint(/* on = */ 0, /* data = */ g_D);
	 // neumann inflow data
	 SpaceTimeField<2, decltype([](const SVector<2>& p, double t) { return p[1] * (1 - p[1]) * t * (0.5 - t); })> g_N;

	 // set up Crank-Nicolson time integration scheme
	 double T = 0.5, DeltaT = 0.02;
	 int n_times = std::ceil(T/DeltaT);
	 DMatrix<double> solution(dof_handler.n_dofs(), n_times);
	 solution.col(0) = DVector<double>::Zero(dof_handler.n_dofs());   // zero initial condition
	 SpMatrix<double> M = integral(unit_square)(u * v).assemble();    // mass matrix
	 SpMatrix<double> A = M / DeltaT + a.assemble() * 0.5;            // stiff matrix (SPD)
	 dof_handler.enforce_constraints(A);
	 Eigen::SimplicialLLT<SpMatrix<double>> lin_solver(A);
	 // compute matrix of rhs (here we include non-homogeneous neumann BCs)
	 DMatrix<double> F(dof_handler.n_dofs(), n_times);
	 for (int i = 0; i < n_times; ++i) {
	     F.col(i) = (integral(unit_square)(f * v) +    // forcing term
	                 integral(unit_square.boundary(/* on = */ 1))(g_N.at(DeltaT * i) * v)    // neumann BCs
			 ).assemble();
	 }
	 // time integration
	 for (int i = 1; i < n_times; ++i) {
             DVector<double> b =
                 (M / DeltaT - A / 2) * solution.col(i - 1) + 0.5 * (F.col(i) + F.col(i - 1));   // update rhs
	     dof_handler.enforce_constraints(b);
	     solution.col(i) = lin_solver.solve(b);
	 }
	 return 0;
      }

.. image:: heat_neumann.gif
   :width: 400
   :align: center
