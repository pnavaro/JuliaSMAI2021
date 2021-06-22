#md # ---

#md # # Why use Julia language!
#md #
#md # - **You develop in the same language in which you optimize.**
#md # - Packaging system is very efficient (5858 registered packages)
#md # - PyPi (311,500 projects) R (17739 packages)
#md # - It is very easy to create a package (easier than R and Python)
#md # - It is very easy to access to a GPU device.
#md # - Nice interface for Linear Algebra and Differential Equations
#md # - Easy access to BLAS, LAPACK and scientific computing librairies.
#md # - Julia talks to all major Languages - mostly without overhead!

#md # ---

#md # # What's bad
#md #
#md # - It is still hard to build shared library or executable from Julia code.
#md # - Compilation times lattency. Using [Revise.jl](https://github.com/timholy/Revise.jl) help a lot.
#md # - Plotting takes time (5 seconds for the first plot)
#md # - OpenMP is better than the Julia multithreading library but it is progressing.
#md # - Does not work well with vectorized code, you need to do a lot of inplace computation to avoid memory allocations and use explicit views to avoid copy. There are some packages like [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl).
#md 
#md # [What's Bad About Julia by Jeff Bezanson](https://www.youtube.com/watch?v=TPuJsgyu87U)

#md # ---

#md # ## From zero to Julia!
#md #
#md # https://techytok.com/from-zero-to-julia/
#md #
#md # ## Python-Julia benchmarks by Thierry Dumont
#md #
#md # https://github.com/Thierry-Dumont/BenchmarksPythonJuliaAndCo/wiki
#md #
#md # ## Mailing List
#md # - https://listes.services.cnrs.fr/wws/info/julia
#md #
