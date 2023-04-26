using Documenter, HDC

const src = joinpath("..", "src")

makedocs(
    sitename = "HDC.jl",
    format = Documenter.HTML(),
    doctest = true,
    src = [src],
)