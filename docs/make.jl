using Documenter
using SimpleSequencer

makedocs(
    modules = [SimpleSequencer]
)

deploydocs(
    deps = Deps.pip("mkdocs", "mkdocs-material", "python-markdown-math"),
    julia = "nightly",
    osname = "linux",
    repo = "github.com/PainterQubits/SimpleSequencer.jl.git"
)
