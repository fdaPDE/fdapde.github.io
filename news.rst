:hide-toc:
:hide-footer:

Latest - September 2024
=======================

.. toctree::
   :caption: Content
   :hidden:
   :maxdepth: 2
	      
   news/may24
   news/feb24
   news/jan24

Introduction of several core API changes, which significantly alter the usage of the library at the C++ level. From this update on, support for PDEs in strong form is dropped.

..
   there is only one namespace fdapde, dropped core, models, calibration, namespaces
   use flag -DNDEBUG (Eigen produces a lot of additional code with debugging active)

**core**

* **constexpr linear algebra**: (*this is something you won't see outside fdaPDE eheheh*) Introduction of several built-in types to enable for compile-time dense linear algebra, togheter with the possibility to solve dense linear system :math:`Ax = b` at compile-time using LU factorization with partial pivoting.

  You find under the :code:`cexpr` namespace, templates :code:`Matrix<Scalar, Rows, Cols>` and :code:`Vector<Scalar, Rows>` which are the backbone for compile-time dense linear algerba. Here a brief explanation of what this means:
  
  .. code-block:: cpp
     :caption: test.cpp
	    
     #include <fdaPDE/linear_algebra.h>
     using namespace fdapde::cexpr;

     int main() {
         // a 2X2 matrix of int
	 static constexpr Matrix<int, 2, 2> M({1, 2, 3, 4});
	 
	 // using M, you can perform computations at compile time. Suppose you want to sum M with itself

	 constexpr auto A = M + M;

	 // A is an expression template encoding the action of summing M with itself.
	 // The point is that, the matrix addition can be performed by the compiler
	 // during compilation. Don't you believe it? Let's query the compiler by writing
         // a static_assert which for sure is falsified

         static_assert(A(0, 0) == 0);
	 return 0;
     }
	    
  We know :code:`A(0, 0) = 2`, and indeed if we compile the program above we expect the compilation to fail:

  .. code-block:: text

       > g++ -o test test.cpp -I.. -O2 -std=c++20
	    
         test.cpp: In function ‘int main()’:
         test.cpp:57:27: error: static assertion failed
            57 |     static_assert(A(0, 0) == 0);
               |                   ~~~~~~~~^~~~
         test.cpp:57:27: note: the comparison reduces to ‘(2 == 0)’
                                                           ^
                                    here the value of A(0, 0) produced by the compiler

  If this still seems useless to you, check the produced assembly code (aka, machine code) to see how much the compiler is able to optimize the operation.
  
  .. code-block:: cpp

      #include <fdaPDE/linear_algebra.h>
      using namespace fdapde::cexpr;

      int main() {
          constexpr Matrix<int, 2, 2> M({1, 2, 3, 4});

          constexpr auto A = M + M;
          return A(0, 0);
      }

  .. code-block:: cpp
     :caption: Generated assembly

      Dump of assembler code for function main():
      0x00000000000010b0 <+0>:     mov    $0x2,%eax    // return statment (which indeed returns 0x2 = 2)
      0x00000000000010b5 <+5>:     ret
      End of assembler dump.

  Just one move instruction! The compiler was able to completely optimize out the matrix addition, and to return directly the result (0x2, which is :code:`A(0, 0) = 2`).

  .. note::

     It must be pointer out that Eigen is also able to reach the same assembly code with the :code:`-DNDEBUG` flag activated. For instance, this Eigen code

     .. code-block:: cpp

	#include <Eigen/Dense>

	int main() {
	    SMatrix<2, 2, double> M;
            M << 1, 2, 3, 4;

	    auto A = M + M;
            return A(0, 0);
	}

     produces the same assembly code above (the compiler is smart enought to completely optimize the matrix addition). This is true for any arithmetic operation between *static-sized* Eigen types. Nonethless, **Eigen types cannot be used in a constexpr-context**, and this poses several limitations, for instance, to evaluate mathematical expressions at compile-time (see next).

  .. note::

    Another limitation of using Eigen (for complie-time linear algebra) is the impossibility to store the result of a computation done at compile-time, and later use it at compile-time.

    .. code-block:: cpp

	#include <fdaPDE/linear_algebra.h>
	using namespace fdapde;

	// since C++20 lambdas have a constexpr call operator. We can define a linear system Ax = b inside a lambda and get it solved
	// at compile time. The solution can be bound to a (static) constexpr variable
	constexpr cexpr::Matrix<double, 3, 1> x = []() {
	   cexpr::Matrix<double, 3, 3> M;
	   for (int i = 0; i < 3; ++i) {
	       for (int j = 0; j < 3; ++j) { M(i, j) = 3 * i + j + 1; }
	   }
	   cexpr::PartialPivLU<cexpr::Matrix<double, 3, 3>> invM;
	   invM.compute(M);
	   cexpr::Matrix<double, 3, 1> b = invM.solve(cexpr::Matrix<double, 3, 1>(1, 1, 1));
	   return b; // solution is [-3 5 -2]
	}();

	int main() {
	    return x[0] + x[0];
	}

	// generated assembly

	Dump of assembler code for function main():
        0x00000000000010d0 <+0>:     mov    $0xfffffffa,%eax    // returns -6
        0x00000000000010d5 <+5>:     ret
        End of assembler dump.

    Again, the compiler was able to completely optimize out all the computations and just return what asked. This is not possible with Eigen, as the nearest thing we can do without constexpr is the following

    .. code-block:: cpp

	#include <Eigen/Dense>

	// solve the linear system and bound the solution to a (static) variable
	Eigen::Matrix<double, 3, 1> x = []() {
	    Eigen::Matrix<double, 3, 3> M;
	    for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) { M(i, j) = 3 * i + j + 1; }
	    }
	    Eigen::PartialPivLU<Eigen::Matrix<double, 3, 3>> invM;
	    invM.compute(M);
	    Eigen::Matrix<double, 3, 1> b = invM.solve(Eigen::Matrix<double, 3, 1>(1, 1, 1));
	    return b;
	}();

	int main() {
	    return x[0] + x[0];
	}

	// generated assembly

	Dump of assembler code for function main():
	0x00000000000010d0 <+0>:     movsd  0x30a8(%rip),%xmm0   // load data from global variable x
	0x00000000000010d8 <+8>:     addsd  %xmm0,%xmm0          // execute addition x[0] + x[0]
	0x00000000000010dc <+12>:    cvttsd2si %xmm0,%eax        // round result from double to integer and return
	0x00000000000010e0 <+16>:    ret
	End of assembler dump.
     
    The compiler, while was still able to solve the linear system at compile-time (since no function-call is executed, this is lambda expressions black magic), is not able to directly return the result, as it is no capable to inspect an Eigen type during compilation. In my opinion, there is no point in having code executed at run-time for a computation which is exactly known at compile-time. Therefore, fdaPDE was able to produce better optimized code in this situation.

    Is not difficult to start noticing the advantages of constexpr computations (if you are in the position to do so).

    .. code-block:: cpp

       // assume you just want to compute the sum of the elements of x
       int main() {
           double sum = 0;
	   for(int i = 0; i < x.rows(); ++x) { sum += x[i]; } // sum is 0
	   return sum;
       }

       // assembly generated by fdaPDE

       Dump of assembler code for function main():
       0x0000000000001110 <+0>:     xor    %eax,%eax // returns 0 (just a nice machine trick to return 0, xor a register with itself)
       0x0000000000001112 <+2>:     ret
       End of assembler dump.

       // assembly generated by Eigen

       Dump of assembler code for function main():
       0x0000000000001110 <+0>:     pxor   %xmm0,%xmm0            // zero xmm0 register
       0x0000000000001114 <+4>:     addsd  0x3064(%rip),%xmm0     // load x[0] in xmm0
       0x000000000000111c <+12>:    lea    0x3085(%rip),%rdx      
       0x0000000000001123 <+19>:    lea    -0x20(%rdx),%rax       
       0x0000000000001127 <+23>:    addsd  (%rax),%xmm0           // execute for loop <---------------------
       0x000000000000112b <+27>:    add    $0x10,%rax             //                                        |
       0x000000000000112f <+31>:    addsd  -0x8(%rax),%xmm0       // sum += x[i]                            |
       0x0000000000001134 <+36>:    cmp    %rdx,%rax              //                                        |
       0x0000000000001137 <+39>:    jne    0x1127 <main()+23>     // ---------------------------------------
       0x0000000000001139 <+41>:    cvttsd2si %xmm0,%eax          // return computed sum
       0x000000000000113d <+45>:    ret
       End of assembler dump.
       
    While fdaPDE was completely able to avoid any for loop at run-time (as it perfectly knows how to compute the for-loop at compile time), Eigen cannot, and must execute code (therefore, waste time) to produce a result. You will further notice the advantages of such data-types when involved in the much more involved math expressions. 

  Below a summary of the API exposed for constexpr dense linear algebra at the time of this update:
    
  .. list-table:: **constexpr dense linear algebra API**
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
    
     
* **field expressions revised**: since writing mathmetical expressions is now a core part of the library API, the whole expression template mechanism has been reimplemented from scratch.

  .. tip::
     
     One notable addition to the fields API are *constexpr-enabled* expressions, i.e., under some circumstances expressions can be evaluated at compile-time.

* **new finite element module**:
  
* **shapefile reader**:


