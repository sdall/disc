
using Distributions: ccdf, Chisq

function desc(X::Vector{SetType}, A, B, C, D; min_support=2, max_factor_size=8, max_factor_width=50, args...) where {SetType}
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    n    = size(X, 1)
    L    = Lattice{Candidate{SetType}}(X, x -> x.support)
    p    = MaxEnt{SetType,Float64}([s.support / n for s in L.singletons])
    S    = Candidate{SetType}[]
    cost = log(n) / 2

    isforbidden(x) = isforbidden_ts!(p, x.set, max_factor_size, max_factor_width, A)
    discover_patterns!(L,
                       x -> if x.support <= min_support || isforbidden(x)
                           0.0
                       else
                           x.support * (log(x.support / n) - log_expectation_ts!(p, x.set, A, B, C)) - cost
                       end,
                       isforbidden,
                       x -> if insert_pattern!(p, x.support / n, x.set, max_factor_size, max_factor_width, D)
                           push!(S, x)
                           true
                       else
                           false
                       end; args...)
    p, S, L.singletons
end

function characterize(Dist::Type, mask, patternset, singletons, A, B, C, D; max_factor_size=8, max_factor_width=50,
                      enable_model_selection=false, min_support=2, smoothing_constant=0.01)
    n = length(mask)
    c = smoothing_constant
    th = enable_model_selection ? log(n) / 2 : 0.0

    fr(s) = (intersection_size(s.rows, mask) + c) / (n + c)
    pr = Dist(map(fr, singletons))

    for x in patternset
        isforbidden!(pr, x.set, max_factor_size, max_factor_width, A) && continue

        cx = intersection_size(x.rows, mask)
        cx < min_support && continue

        E = expectation!(pr, copy!(A, x.set), B, C)
        E = (E * n + c) / (n + c)
        q = (cx + c) / (n + c)

        if cx * (log(q) - log(E)) > th
            insert_pattern!(pr, q, x.set, max_factor_size, max_factor_width, D)
        end
    end
    pr
end

function reassign_rows(X, P, A::Vector{BitSet}, B::Vector{BitSet}, C)
    ThreadsX.map(X) do x
        argmax(k -> log_expectation_ts!(P[k], x, A, B, C), eachindex(P))
    end
end

function reassign(y, X, P, patternset, singletons, max_iteration, A, B, C, D; SetType=eltype(A), kwargs...)
    Q = deepcopy(P)
    z = deepcopy(y)

    for _ in 1:max_iteration
        z0 = copy(z)

        z = reassign_rows(X, Q, A, B, C)
        z == z0 && break

        Z = unique(z) |> sort
        Q = [q for (i, q) in enumerate(Q) if i in Z]
        z = [findfirst(==(i), Z) for i in z]

        for k in unique(z)
            if (z0 .== k) != (z .== k)
                id = Threads.threadid()
                group = SetType(findall(==(k), z))
                Q[k] = characterize(eltype(P), group, patternset, singletons, A[id], B[id], C[id], D[id]; kwargs...)
            end
        end
    end
    z, Q
end

function jsdivergence(P, Q, S, x_index, A, B, C; th=1e-18, eps=1e-15)
    kl(q, p) = q < th ? 0.0 : q * log(q / p)
    function js(q, p)
        m = (p + q) / 2
        m < th ? 0.0 : 2(kl(p, m) + kl(q, m))
    end
    jc(q, p) = js(clamp(q, eps, 1 - eps), clamp(p, eps, 1 - eps))
    E(x, pr) = expectation!(pr, copy!(A, x.set), B, C)
    Ediv(i) = i == x_index ? 0.0 : jc(E(S[i], P), E(S[i], Q))
    sum(Ediv, eachindex(S); init=0.0)
end

function jsdivergence(P::BitSet, Q::BitSet, S, x_index; th=1e-18, eps=1e-15)
    kl(q, p) = q < th ? 0.0 : q * log(q / p)
    function js(q, p)
        m = (p + q) / 2
        m < th ? 0.0 : 2(kl(p, m) + kl(q, m))
    end
    function jc(x)
        js(clamp(intersection_size(x.rows, P) / length(P), eps, 1 - eps), clamp(intersection_size(x.rows, Q) / length(Q), eps, 1 - eps))
    end
    sum(eachindex(S); init=0.0) do i
        i == x_index ? 0.0 : jc(S[i])
    end
end

menendez(P, Q, S, x_index, A, B, C) = ccdf(Chisq(length(S) - 1), 2 * length(S) * jsdivergence(P, Q, S, x_index, A, B, C))
menendez(P::BitSet, Q::BitSet, S, x_index) = @show ccdf(Chisq(length(S) - 1), 2 * length(S) * jsdivergence(P, Q, S, x_index))
disc_model_cost(ndatapoints::Int, nelements::Int, ngroups::Int) = (ngroups * nelements) * log(ndatapoints) / 2

function prepare_candidates(Y, masks, splitset, rejected, min_group_size)
    ThreadsX.collect((j, x)
                     for (j, c) in enumerate(Y), x in eachindex(splitset)
                     if !in((c, x), rejected) &&
                            (c = intersection_size(splitset[x].rows, masks[j])) > min_group_size &&
                            (c < length(masks[j]) - min_group_size))
end
function prepare_candidates(SetType::Type, y, splitset, rejected, min_group_size)
    Y = unique(y)
    masks = [SetType(findall(==(c), y)) for c in Y]
    candidates = prepare_candidates(Y, masks, splitset, rejected, min_group_size)
    masks, candidates
end

function iterate_heuristically!(rejected, dist, f_old, individual_likelihoods, y::Vector{Int}, X,
                                splitset::Vector{Candidate{SetType}}, patternset::Vector{Candidate{SetType}},
                                singletons::Vector{Candidate{SetType}}, A::Vector{SetType}, B::Vector{SetType}, C, D;
                                maximize_expectation=true,
                                adjusted_alpha=0.05,
                                max_factor_size=8,
                                max_factor_width=20,
                                min_group_size=2,
                                callback=nothing) where {SetType}
    @assert length(individual_likelihoods) == length(dist)

    S, I = patternset, singletons
    masks, candidates = prepare_candidates(SetType, y, splitset, rejected, min_group_size)
    rejected_ = Vector{Tuple{Int64,Int64}}() # = [Vector{Tuple{Int64, Int64}}() for _ in 1:Threads.nthreads()]

    function characterize_(m; tid=Threads.threadid())
        characterize(eltype(dist), m, S, I, A[tid], B[tid], C[tid], D[tid];
                     max_factor_size=max_factor_size, max_factor_width=max_factor_width)
    end
    ell(m, p) =
        ThreadsX.sum([i for i in m]) do k
            -log_expectation_ts!(p, X[k], A, B, C)
        end

    L = sum(individual_likelihoods)
    M = disc_model_cost(length(X), length(S), length(dist) + 1)
    init = (0, nothing, f_old)

    next = mapreduce((c, d) -> c[3] < d[3] ? c : d, eachindex(candidates); init=init) do candidate_index
        j, i = candidates[candidate_index]
        m1, m2 = intersect(splitset[i].rows, masks[j]), setdiff(masks[j], splitset[i].rows)
        p1, p2 = characterize_(m1), characterize_(m2)
        tid = Threads.threadid()

        if menendez(p1, p2, S, i < length(S) ? i : nothing, A[tid], B[tid], C[tid]) >= adjusted_alpha
            push!(rejected_, (j, i))
            return init
        end

        fn = L - individual_likelihoods[j] + ell(m1, p1) + ell(m2, p2) + M

        !isnothing(callback) && callback(j, i, fn, (p1, p2))

        if fn < f_old
            candidate_index, (p1, p2), fn
        else
            push!(rejected_, (j, i))
            init
        end
    end

    candidate_index, ps, fn = next

    if fn < f_old
        j, i = candidates[candidate_index]
        push!(rejected_, (j, i))

        m1, m2 = intersect(splitset[i].rows, masks[j]), setdiff(masks[j], splitset[i].rows)
        p = push!(deepcopy(dist), ps[2])
        p[j] = ps[1]

        l2 = length(dist) + 1
        z = copy(y)
        # l1 = Y[j] ; for k in m1 z[k] = l1 end
        for k in m2
            z[k] = l2
        end

        ll = if maximize_expectation
            z, p = reassign(z, X, p, S, I, 5, A, B, C, D; max_factor_size=max_factor_size, max_factor_width=max_factor_width)
            map(unique(z)) do i
                ThreadsX.sum(eachindex(z)) do k
                    (z[k] == i) ? -log_expectation_ts!(p[i], X[k], A, B, C) : 0.0
                end
            end
        else
            ll = push!(copy(individual_likelihoods), ell(m2, ps[2]))
            ll[j] = ell(m1, ps[1])
            ll
        end

        z, p, sum(ll) + M, rejected_, ll
    else
        y, dist, f_old, rejected_, individual_likelihoods
    end
end

function iterate_greedily!(rejected, dist, f_old, y, X, splitset, patternset,
                           singletons, A::Vector{SetType}, B::Vector{SetType}, C, D;
                           adjusted_alpha   = 0.05,
                           max_factor_size  = 8,
                           max_factor_width = 20,
                           min_group_size   = 2,
                           callback         = nothing) where {SetType}
    S, I = patternset, singletons
    masks, candidates = prepare_candidates(SetType, y, splitset, rejected, min_group_size)
    rejected_ = Vector{Tuple{Int64,Int64}}()

    function characterize_(m; tid=Threads.threadid())
        characterize(eltype(dist), m, S, I, A[tid], B[tid], C[tid], D[tid];
                     max_factor_size=max_factor_size, max_factor_width=max_factor_width)
    end

    init = (Vector{Int}(), Vector{Int}(), f_old)
    next = mapreduce((c, d) -> c[3] < d[3] ? c : d, eachindex(candidates); init=init) do candidate_index
        tid = Threads.threadid()
        j, i = candidates[candidate_index]

        m1, m2 = intersect(splitset[i].rows, masks[j]), setdiff(masks[j], splitset[i].rows)
        p1, p2 = characterize_(m1), characterize_(m2)

        if menendez(p1, p2, S, i < length(S) ? i : nothing, A[tid], B[tid], C[tid]) >= adjusted_alpha
            push!(rejected_, (j, i))
            return init
        end

        p = push!(deepcopy(dist), p2)
        p[j] = p1

        z, p = reassign(y, X, p, S, I, 2, A, B, C, D;
                        max_factor_size=max_factor_size, max_factor_width=max_factor_width)

        fn = ThreadsX.sum(zip(X, z); init=0.0) do (x, j)
            -log_expectation_ts!(p[j], x, A, B, C)
        end

        !isnothing(callback) && callback(j, i, fn, (p1, p2))

        fn >= f_old && push!(rejected_, (j, i))

        z, p, fn
    end
    if next[3] < f_old
        next..., rejected_
    else
        y, dist, f_old, rejected_
    end
end

function create_disc_context(::Type{SetType}, max_factor_width) where {SetType}
    [SetType() for _ in 1:Threads.nthreads()],
    [SetType() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:Threads.nthreads()],
    [[MaxEntContext{SetType}() for _ in 1:(max_factor_width + 1)] for _ in 1:Threads.nthreads()]
end

"""
    disc_heuristic(x; kwargs...) 
    
Heuristically discovers and characterizes significantly diverging groups in the data using the maximum entropy distribution, statistical testing, and a Bayesian information criteria.

See [`disc_greedy`](@ref).

# Arguments

- `x::Vector{SetType}`: Input dataset.

# Options

- `alpha::Float64`: Set the initial statistical significance level before Bonferroni adjustment.
- `split_on_singletons::Bool`: Allows to partition data using singletons (potentially impactful on time and quality).
- `maximize_expectation::Bool`: Iteratively reassigns data points to their optimal group.
- `min_group_size::Integer`: Require a minimum size for each group.
- `min_support::Integer`: Require a minimum support for each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by [`MAX_MAXENT_FACTOR_SIZE`](@ref)=12.
- `max_factor_width::Integer`: Constraint the maximum number of singletons that each factor can model.
- `max_expansions::Integer`: Limit the number of search-space node-expansions per iteration. 
- `max_discoveries::Integer`: Terminate the algorithm after `max_discoveries` discoveries.
- `callback_candidates::Function`: Calls this function for each split candidate. 
- `callback_progress::Function`: Calls this function for the best split. 

# Returns

- A vector of class-labels assigned to each data-point in `x`.
- Per-group factorized maximum entropy distributions [`MaxEnt`](@ref), which contain patterns, singletons, and estimated coefficients.

Note: Extract patterns (discoveries) via [`patterns`](@ref) or the per-group patterns via `patterns.`.

# Example

```julia-repl
julia> using Disc: disc_heuristic, patterns
julia> y, p = disc_heuristic(X; min_group_size = 2)
julia> patterns.(p)
```

"""
function disc_heuristic(X::Vector{SetType};
                        alpha                = 0.05,
                        split_on_singletons  = false,
                        maximize_expectation = true,
                        min_support          = 2,
                        min_group_size       = min_support,
                        max_factor_size      = 7,
                        max_factor_width     = 30,
                        max_discoveries      = typemax(Int),
                        callback_candidates  = nothing,
                        callback_progress    = nothing) where {SetType}
    A, B, C, D = create_disc_context(SetType, max_factor_width)

    p, S, I = desc(X, A, B, C, D[1]; max_expansions=10, max_discoveries=max_discoveries,
                   min_support=min_support, max_factor_size=max_factor_size, max_factor_width=max_factor_width)

    p = [p]
    y = ones(Int, size(X, 1))
    split_candidates = split_on_singletons ? vcat(S, I) : S
    rejected = Set{Tuple{Int,Int}}()
    fwer_alpha = alpha

    ll0 = ThreadsX.sum(zip(X, y); init=0.0) do (x, j)
        -log_expectation_ts!(p[j], x, A, B, C)
    end
    bic = ll0 + disc_model_cost(length(X), length(S), length(p))
    individual_likelihoods = [ll0]

    while true
        adjusted_alpha = fwer_alpha / (length(p) * length(split_candidates))
        fwer_alpha -= adjusted_alpha

        z, q, bic_i, rss, il = iterate_heuristically!(rejected, p, bic, individual_likelihoods, y, X, split_candidates,
                                                      S, I, A, B, C, D;
                                                      adjusted_alpha       = adjusted_alpha,
                                                      min_group_size       = min_group_size,
                                                      max_factor_size      = max_factor_size,
                                                      max_factor_width     = max_factor_width,
                                                      maximize_expectation = maximize_expectation,
                                                      callback             = callback_candidates)
        if length(unique(y)) != length(unique(z))
            y, p, bic, individual_likelihoods = z, q, bic_i, il
            for r in rss
                push!(rejected, r)
            end
            !isnothing(callback_progress) && callback_progress(y, p, bic, individual_likelihoods)
        else
            break
        end
    end
    y, p
end

"""
    disc_greedy(x; kwargs...) 
    
Greedily discovers and characterizes significantly diverging groups in the data using the maximum entropy distribution, statistical testing, and a Bayesian information criteria.

See [`disc_heuristic`](@ref).

# Arguments

- `x::Vector{SetType}`: Input dataset.

# Options

- `alpha::Float64`: Set the initial statistical significance level before Bonferroni adjustment.
- `split_on_singletons::Bool`: Allows to partition data using singletons (potentially impactful on time and quality).
- `min_group_size::Integer`: Require a minimum size for each group.
- `min_support::Integer`: Require a minimum support for each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by [`MAX_MAXENT_FACTOR_SIZE`](@ref)=12.
- `max_factor_width::Integer`: Constraint the maximum number of singletons that each factor can model.
- `max_expansions::Integer`: Limit the number of search-space node-expansions per iteration. 
- `max_discoveries::Integer`: Terminate the algorithm after `max_discoveries` discoveries.
- `callback_candidates::Function`: Calls this function for each split candidate. 
- `callback_progress::Function`: Calls this function for the best split. 

# Returns

- A vector of class-labels assigned to each data-point in `x`.
- Per-group factorized maximum entropy distributions [`MaxEnt`](@ref), which contain patterns, singletons, and estimated coefficients.

Note: Extract patterns (discoveries) via [`patterns`](@ref) or the per-group patterns via `patterns.`.

# Example

```julia-repl
julia> using Disc: disc_greedy, patterns
julia> y, p = disc_greedy(X; min_group_size = 2)
julia> patterns.(p)
```

"""
function disc_greedy(X::Vector{SetType};
                     alpha               = 0.05,
                     split_on_singletons = false,
                     min_support         = 2,
                     min_group_size      = min_support,
                     max_factor_size     = 7,
                     max_factor_width    = 30,
                     max_discoveries     = typemax(Int),
                     callback_candidates = nothing,
                     callback_progress   = nothing) where {SetType}
    A, B, C, D = create_disc_context(SetType, max_factor_width)

    p, S, I = desc(X, A, B, C, D[1]; max_expansions=10, max_discoveries=max_discoveries,
                   min_support=min_support, max_factor_size=max_factor_size, max_factor_width=max_factor_width)

    p = [p]
    y = ones(Int, size(X, 1))
    split_candidates = split_on_singletons ? vcat(S, I) : S
    rejected = Set{Tuple{Int,Int}}()
    fwer_alpha = alpha

    ll0 = ThreadsX.sum(zip(X, y); init=0.0) do (x, j)
        -log_expectation_ts!(p[j], x, A, B, C)
    end
    bic = ll0 + disc_model_cost(length(X), length(S), length(p))

    while true
        adjusted_alpha = fwer_alpha / (length(p) * length(split_candidates))
        fwer_alpha -= adjusted_alpha

        z, q, bic_i, rss = iterate_greedily!(rejected, p, bic, y, X, split_candidates, S, I, A, B, C, D;
                                             adjusted_alpha   = adjusted_alpha,
                                             min_group_size   = min_group_size,
                                             max_factor_size  = max_factor_size,
                                             max_factor_width = max_factor_width,
                                             callback         = callback_candidates)
        if bic_i < bic
            y, p, bic = z, q, bic_i
            for r in rss
                push!(rejected, r)
            end
            !isnothing(callback_progress) && callback_progress(y, p, bic)
        else
            break
        end
    end
    y, p
end
