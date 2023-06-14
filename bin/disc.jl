#!/usr/bin/env -S julia -O3 --threads=auto --check-bounds=no 

using Disc, Comonicon, JSON, GZip, CSV

read_sets(fp; offset=0) = [BitSet([offset + parse(Int, s) for s in split(t, " ") if s != ""]) for t in readlines(fp)]
read_sets(f::String) = read_sets((endswith(f, ".gz") ? GZip : Base).open(f))
function normalize_sets(D)
    I = unique(i for t in D for i in t)
    if sort(I) != collect(eachindex(I))
        tr = Dict(e => i for (i, e) in enumerate(I))
        [BitSet([tr[e] for e in t]) for t in D], Dict(i => e for (e, i) in tr)
    else
        D, nothing
    end
end

"""    
# Introduction

This is a command line interface for the __Disc__ algorithm, which decomposes Boolean datasets into significantly differently distributed parts using maximum entropy distributions, BIC, and statistical significance testing. 
_Returns:_ A JSON document containing meta information, a list of (per-group) `patterns` and the assigned class-`labels`.
_Example:_ `bin/disc.jl data.tsv --min_group_extent=4 > result.json`
_Method:_ Disc either determines decisions heuristically (faster) or greedily (if `--greedy` is set). 

# Arguments

- `x`: Boolean 01 data matrix as a headerless _.tsv_, or as a sparse list-of-sets [_.dat_], where each row `i` is a space-separated list of indices `j`, such that `X[i, j] != 0`. (optionally gzipped [_.gz_]).".

# Options
- `--output`: Write results as JSON to output ('-' = stdout).
- `--alpha`: Set the initial statistical significance level before Bonferroni correction.
- `--split_on_singletons`: Allows partitioning data using singletons (potentially impactful on runtime and quality).
- `--min_group_size`: Require a minimal number of elements per group.
- `--min_support`: Require a minimal support of each pattern.
- `--max_discoveries`: Terminate the algorithm after `max_discoveries` discoveries.
- `--max_factor_size`: Constraint the maximum number of patterns that each factor of the relaxed maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by `MAX_MAXENT_FACTOR_SIZE=12`.
- `--max_factor_width`: Constraint the maximum number of singletons that each factor can model.

# Flags

- `--verbose`: Reports progress on partitioning data to stdout (consider using `--output`).
- `--greedy`: Greedily partitions the data. Determining the partitioning greedily is more precise but also slower than than doing so heuristically.

"""
@main function disc(x;
                    output::String            = "-",
                    greedy::Bool              = false,
                    alpha::Float64            = 0.05,
                    split_on_singletons::Bool = false,
                    min_group_size::Int64     = 2,
                    min_support::Int64        = 2,
                    max_factor_size::Int64    = 8,
                    max_factor_width::Int64   = 50,
                    max_discoveries::Int64    = typemax(Int64),
                    verbose::Bool             = false)
    X, vocab = if any(e -> endswith(x, e), (".dat", ".dat.gz"))
        read_sets(x) |> normalize_sets
    else
        CSV.read(opts["x"], CSV.Tables.matrix; header=0, types=Bool) |> Disc._convert_dataset, nothing
    end

    cbprogress(t...) = @info "Partitioned Data ($(length(t[2])) groups, BIC = $(t[3]))"
    cbcandidates(t...) = @info "Evaluated Candidate ($((t[1], t[2])), Score = $(t[3]))"

    kw = (alpha=alpha, split_on_singletons=split_on_singletons, min_group_size=min_group_size, min_support=min_support,
          max_discoveries=max_discoveries, max_factor_size=max_factor_size, max_factor_width=max_factor_width,
          callback_progress=verbose ? cbprogress : nothing, callback_candidates=verbose ? cbcandidates : nothing)

    y, p = greedy ? Disc.disc_greedy(X; kw...) : Disc.disc_heuristic(X; kw...)

    out = isnothing(vocab) ? e -> Set(e) : e -> Set(vocab[i] for i in e)
    S = p isa Vector ? [[out(e) for e in Disc.patterns(q)] for q in p] : [out(e) for e in Disc.patterns(p)]
    jd = Dict("labels" => y, "patterns" => S, "input" => x) |> JSON.json
    write(output == "-" ? stdout : open(output, 'w'), jd * "\n")
end
