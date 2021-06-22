ENV["GKSwstype"]="100"

using  Literate
using  Plots
import Remark
using Remark, FileWatching

# files =  filter( f -> startswith(f, "0"), readdir("src")) |> collect
 
files = [ "01.Introduction.jl",
          "02.RungeKuttaMethods.jl",
          "03.PoissonEquation.jl",
          "04.HOODESolver.jl",
          "05.Conclusion.jl"]

run(pipeline(`cat src/$files`; stdout="slides/src/index.jl" ))

slideshowdir = Remark.slideshow("slides",
                                options = Dict("ratio" => "16:9"),
                                title = "Differential equations with Julia")

# Open presentation in default browser.
# Remark.open(slideshowdir)

Literate.notebook("src/index.jl", execute=false)

cp("slides/build", "docs")
