ENV["GKSwstype"]="100"

using  Literate
using  Plots
import Remark

files =  filter( f -> startswith(f, "0"), readdir("src")) |> collect

for file in files
    Literate.notebook("src/$file", "notebooks",  execute=true)
end

run(pipeline(`cat src/$files`; stdout="slides.jl" ))
slides_path = joinpath("docs")
mkpath(slides_path)
s = Remark.slideshow("slides.jl", slides_path)
