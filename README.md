# Explainable Data Decompositions with Disc

This repository provides a Julia library that implements the Disc algorithm. 
In Boolean data, Disc discovers groups with significantly differing distributions, which are modeled in terms of the maximum entropy distribution over higher-order feature interactions (i.e., patterns). Disc leverages the Desc algorithm to describe groups in terms of concise set of characteristic and informative patterns which highlight commonalities and differences between groups.

The code is a from-scratch implementation of algorithms described in the [paper](https://doi.org/10.1609/aaai.v34i04.5780).

```
Dalleiger, S. and Vreeken, J. 2020. Explainable Data Decompositions. Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3709–3716. https://doi.org/10.1609/aaai.v34i04.5780
```

Please consider [citing](CITATION.bib) the paper.

[Contributions](CONTRIBUTING.md) are welcome.

## Installation

To install the library from the REPL:
```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/sdall/disc.git")
```

To install the library from the command line:
```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/sdall/disc.git")'
```

To set up the command line interface (CLI) located in `bin/disc.jl`:

1. Clone the repository:
```sh
git clone https://github.com/sdall/disc
```
2. Install the required dependencies including the library:
```sh
julia -e 'using Pkg; Pkg.add(path="./disc"); Pkg.add.(["Comonicon", "CSV", "GZip", "JSON"])'
```

## Usage

A typical usage of the library is, for example:
```julia-repl
julia> using Disc: disc_heuristic, disc_greedy, patterns
julia> y, p = disc_heuristic(X; alpha=0.01)
julia> patterns.(p)

julia> y, p = disc_greedy(X)
julia> patterns.(p)
```
📝 For conciseness, Disc does not use patterns to characterize groups, which are well-characterized by their singletons. 

💡 Disc determines the decomposition either _greedily_ or _heuristically_ (faster). 

For more information, see the documentation:
```julia-repl
help?> disc_heuristic
help?> disc_greedy
```

A typical usage of the command line interface is:
```sh
chmod +x bin/disc.jl
bin/disc.jl dataset.dat.gz dataset.labels.gz > output.json
```
The output contains a list of `patterns` per group and `labels` designating the group of each data point.
For further information regarding usage or input format, please see the complete list of CLI options:
```sh
bin/disc.jl --help
```
