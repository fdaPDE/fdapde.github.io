:hide-footer:
:hide-toc:

Example 4: The Stokes problem
=============================

.. abstract:: The complete script

   .. code-block:: cpp
      :linenos:

	 #include <fdaPDE/finite_elements.h>
	 #include <fdaPDE/linear_algebra.h>
	 using namespace fdapde;
	 
	 int main() {
	     // useful typedef and constants definition
	     constexpr int local_dim = 2;
	     using PointT = Eigen::Matrix<double, local_dim, 1>;
	     
	     Triangulation<local_dim, local_dim> unit_square = Triangulation<2, 2>::UnitSquare(60);

	     unit_square.mark_boundary(/* as = */ 0);
	     unit_square.mark_boundary(/* as = */ 1, /* where = */ [](const typename Triangulation<2, 2>::EdgeType& e) {
	         return (e.node(0)[1] == 0 && e.node(1)[1] == 0) ||   // bottom and top side of unit_square
                 (e.node(0)[1] == 1 && e.node(1)[1] == 1);
	     });
	     unit_square.mark_boundary(/* as = */ 2, /* where = */ [](const typename Triangulation<2, 2>::EdgeType& e) {
                 return (e.node(0)[0] == 1 && e.node(1)[0] == 1);   // right side of unit_square
	     });
    
	     // Taylor-Hood stable finite element pair
	     FeSpace Vh(unit_square, P2<2>);   // velocity component
	     FeSpace Qh(unit_square, P1<1>);   // pressure component
	     TrialFunction u(Vh);
	     TestFunction  v(Vh);
	     TestFunction  q(Qh);

	     // bilinear forms
	     Eigen::SparseMatrix<double> A = integral(unit_square)(dot(grad(u), grad(v))).assemble();
	     Eigen::SparseMatrix<double> B = integral(unit_square)(q * div(u)).assemble();
	     // assemble 2x2 mixed elements block system matrix
	     SparseBlockMatrix<double, 2, 2> M {A, B.transpose(), B, ZeroBlk(B.rows(), B.rows())};
	     // forcing term
	     MatrixField<2, 2, 1, decltype([](const PointT&) { return 1; })> f_;
	     Eigen::Matrix<double, Dynamic, 1> F = Eigen::Matrix<double, Dynamic, 1>::Zero(M.rows());
	     F.topRows(A.rows()) =
	         (integral(unit_square)(dot(f_, v)) + integral(unit_square.boundary(/* on = */ 2))(dot(f_, v))).assemble();

	     // enforce boundary conditions
	     auto& dof_handler_ = Vh.dof_handler();
	     ScalarField<2, decltype([](const PointT&) { return 0; })> gx;
	     ScalarField<2, decltype([](const PointT&) { return 0; })> gy;
	     dof_handler_.set_dirichlet_constraint(/* on = */ 1, /* data = */ gx, gy);
	     dof_handler_.enforce_constraints(M.block(0, 0), F.topRows(A.rows()));

	     // solve FEM system
	     Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(M);
	     Eigen::Matrix<double, Dynamic, 1> solution = solver.solve(F);
	     
	     return 0;
	 }

.. image:: stokes_velocity.png
   :width: 400
   :align: center
