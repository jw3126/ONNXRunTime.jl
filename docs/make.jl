using ONNXRunTime
using Documenter

DocMeta.setdocmeta!(ONNXRunTime, :DocTestSetup, :(using ONNXRunTime); recursive=true)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"), force=true)

makedocs(;
    modules=[ONNXRunTime],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/jw3126/ONNXRunTime.jl/blob/{commit}{path}#{line}",
    sitename="ONNXRunTime.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jw3126.github.io/ONNXRunTime.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Explanation" => "explanation.md",
    ],
)

deploydocs(;
    repo="github.com/jw3126/ONNXRunTime.jl",
    devbranch = "main",
)
