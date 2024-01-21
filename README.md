These are the source files for the paper "The Lorenz system as a gradient-like system".

theorem_3.jl
------------
This proves that the Lorenz system is gradient-like between given values of rhomin and rhomax, using polynomials of specified degree for V and s.
Tested with Julia 1.9.3.
Dependency versions:
* SumOfSquares.jl 0.7.3
* IntervalMatrices.jl 0.10.0
* IntervalArithmetic.jl 0.22.5
* MosekTools.jl 0.15.1

and with Mosek 10.0.

lorenz.jl
---------
This produces Figure 1.

map.jl
------
This produces Figure 2.
