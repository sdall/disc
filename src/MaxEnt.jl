const MAX_MAXENT_FACTOR_SIZE = 12

Base.@kwdef mutable struct MaxEntFactor{SetType,FloatType}
    range::SetType                         = SetType()
    theta0::FloatType                      = one(FloatType)
    singleton::Vector{Int64}               = []
    singleton_theta::Vector{FloatType}     = []
    singleton_frequency::Vector{FloatType} = []
    set::Vector{SetType}                   = []
    theta::Vector{FloatType}               = []
    frequency::Vector{FloatType}           = []
end

Base.insert!(f::MaxEntFactor{S,T}, q, x::S) where {S,T} =
    if !any(y -> y == x, f.set)
        union!(f.range, copy(x))
        push!(f.set, copy(x))
        push!(f.theta, copy(q))
        push!(f.frequency, copy(q))
    end

Base.insert!(f::MaxEntFactor{S,T}, q, x::Integer) where {S,T} =
    if !any(y -> y == x, f.singleton)
        push!(f.range, copy(x))
        push!(f.singleton, copy(x))
        push!(f.singleton_theta, copy(q))
        push!(f.singleton_frequency, copy(q))
    end

function log_probability(f::MaxEntFactor, x)
    prob = log(f.theta0)
    @fastmath @inbounds for i in eachindex(f.set)
        if issubset(f.set[i], x)
            prob += log(f.theta[i])
        end
    end
    @fastmath @inbounds for i in eachindex(f.singleton)
        if issubset(f.singleton[i], x)
            prob += log(f.singleton_theta[i])
        end
    end
    prob
end

@fastmath @inbounds function probability(f::MaxEntFactor, x)
    prob = f.theta0
    for i in eachindex(f.set)
        if issubset(f.set[i], x)
            prob *= f.theta[i]
        end
    end
    for i in eachindex(f.singleton)
        if issubset(f.singleton[i], x)
            prob *= f.singleton_theta[i]
        end
    end
    prob
end

function next_permutation(v::Int64)
    t = (v | (v - 1)) + 1
    t | ((div((t & -t), (v & -v)) >> 1) - 1)
end

function permute_level(n::Int64, k::Int64, fn)
    i = Int64(0)
    @fastmath @simd for z in 0:(k - 1)
        i |= (1 << z)
    end
    max = i << (n - k)
    while i != max
        fn(i)
        i = next_permutation(i)
    end
    fn(i) # max
end

function permute_all(fn, n::Int64)
    i = Int64(0)
    @fastmath @simd for z in 0:(n - 1)
        i |= (1 << z)
    end
    fn(i)
    @fastmath @simd for s in (n - 1):-1:1
        permute_level(n, s, fn)
    end
    fn(0)
end

@inline function foreachbit(fn, i)
    l = i
    j = 0
    @fastmath @inbounds while l != 0
        Bool(l & 1) && fn(j)
        l >>= 1
        j = j + 1
    end
end

function create_permutation_list()
    perms = [zeros(Int64, 2^l) for l in 1:MAX_MAXENT_FACTOR_SIZE]
    for (n, class) in enumerate(perms)
        j = 1
        permute_all(n) do i
            @inbounds class[j] = i
            j = j + 1
        end
    end
    perms
end

const permutations::Vector{Vector{Int64}} = create_permutation_list()

mutable struct MaxEntContext{S}
    classes::Vector{S}
    values::Vector{Int}
    MaxEntContext{S}(N=2^MAX_MAXENT_FACTOR_SIZE) where {S} = new{S}([S() for _ in 1:N], zeros(Int, N))
end

function equivalence_classes!(ctx::MaxEntContext{S}, width, itemsets, add_itemset=nothing) where {S}
    classes, values = ctx.classes, ctx.values

    len = length(itemsets) + (add_itemset !== nothing)

    if len == 0 || width == 0
        empty!(classes[1])
        values[1] = exp2(width)
        return 0
    end

    getset = ifelse(add_itemset !== nothing,
                    i -> (i <= length(itemsets) ? itemsets[i] : add_itemset),
                    i -> itemsets[i])

    @fastmath @inbounds for (index, p) in enumerate(permutations[len])
        empty!(classes[index])
        foreachbit(p) do i
            union!(classes[index], getset(i + 1))
        end
        k = exp2(width - length(classes[index]))
        values[index] = k

        @inbounds for prev in 1:(index - 1)
            if values[prev] > 0 && issubset(classes[index], classes[prev])
                if length(classes[index]) == length(classes[prev])
                    values[index] = 0
                    break
                else
                    values[index] -= values[prev]
                end
            end
        end
    end
    length(permutations[len])
end

function expectation_known(f::MaxEntFactor{S,T}, x, classes, values, until::Int=length(classes))::T where {S,T}
    @fastmath sum(1:until; init=zero(T)) do i
        @inbounds values[i] != 0 && issubset(x, classes[i]) ? values[i] * probability(f, classes[i]) : zero(T)
    end
end

function expectation_unknown(f::MaxEntFactor, x, ctx::MaxEntContext)
    until = equivalence_classes!(ctx, union_size(f.range, x), f.set, x)
    expectation_known(f, x, ctx.classes, ctx.values, until)
end

@inbounds function expectation(f::MaxEntFactor{S,T}, x, ctx::MaxEntContext)::T where {S,T}
    if (i = findfirst(==(x), f.set)) !== nothing
        f.frequency[i]
    else
        expectation_unknown(f, x, ctx)
    end
end

@fastmath @inbounds function iterative_scaling!(model::MaxEntFactor, ctx, ctx_lim, sctx, sctx_lim;
                                                max_iter=300, sensitivity=1e-4, epsilon=1e-5)
    bad_scaling_factor(a) = isinf(a) || isnan(a) || a <= 0 || (a != a)

    model.theta0 = exp2(-length(model.range) - 0)
    model.theta .= model.frequency .* model.theta0
    model.singleton_theta .= model.singleton_frequency .* model.theta0

    tol = sensitivity * (length(model.singleton) + length(model.set))
    pg  = typemax(model.theta0)
    for _ in 1:max_iter
        g = 0.0
        for i in eachindex(model.singleton)
            q = model.singleton_frequency[i]
            p = expectation_known(model, model.singleton[i], sctx[i].classes, sctx[i].values, sctx_lim[i])
            g += abs(q - p)
            if abs(q - p) < sensitivity || bad_scaling_factor(model.singleton_theta[i] * (q / p))
                continue
            end
            model.singleton_theta[i] *= q / p
        end
        for i in eachindex(model.set)
            q = model.frequency[i]
            p = expectation_known(model, model.set[i], ctx.classes, ctx.values, ctx_lim)
            g += abs(q - p)
            if abs(q - p) < sensitivity || bad_scaling_factor(model.theta[i] * (q / p))
                continue
            end
            model.theta[i] *= q / p
        end
        if g < tol || abs(g - pg) < epsilon
            pg = g
            break
        end
        pg = g
    end
    pg
end

function fit!(f::MaxEntFactor{S,T}, ctx::Vector{MaxEntContext{S}}) where {S,T}
    @assert length(ctx) >= length(f.singleton) + 1
    d = length(f.range)
    u = equivalence_classes!(ctx[end], d, f.set)
    U = ThreadsX.map(i -> equivalence_classes!(ctx[i], d, f.set, f.singleton[i]), eachindex(f.singleton))::Vector{Int64}
    iterative_scaling!(f, ctx[end], u, ctx, U)
end

abstract type MaxEntModel end

mutable struct MaxEnt{S,T} <: MaxEntModel
    factor                  :: Vector{MaxEntFactor{S,T}}
    singleton               :: Vector{Int64}
    singleton_frequency     :: Vector{T}
    singleton_log_frequency :: Vector{T}

    MaxEnt{S,T}(frequencies::Vector) where {S,T} = new{S,T}([], collect(eachindex(frequencies)), frequencies, map(log, frequencies))
end

function create_factor(pr::MaxEnt{S,T}, factors::Vector{Int}, singletons) where {S,T}
    next = MaxEntFactor{S,T}()
    for i in factors
        f = pr.factor[i]
        for j in eachindex(f.set)
            insert!(next, f.frequency[j], copy(f.set[j]))
        end
        for j in eachindex(f.singleton)
            insert!(next, f.singleton_frequency[j], f.singleton[j])
        end
    end
    for i in singletons
        insert!(next, pr.singleton_frequency[i], i)
    end
    next
end

function factorize!(fn, pr::MaxEnt, x)
    for (i, f) in enumerate(pr.factor)
        if intersects(f.range, x)
            fn(i)
            setdiff!(x, f.range)
            isempty(x) && break
        end
    end
end

@inline function factorize!(mapper, reducer, init, pr::MaxEnt, x)
    acc = init
    for (i, f) in enumerate(pr.factor)
        if intersects(f.range, x)
            acc = @inline reducer(acc, mapper(i))
            setdiff!(x, f.range)
            isempty(x) && break
        end
    end
    acc
end

@inline function factorize_skip_singletons!(mapper, reducer, init, pr::MaxEnt, x)
    acc = init
    for (i, f) in enumerate(pr.factor)
        if intersection_size(f.range, x) >= 2
            acc = @inline reducer(acc, mapper(i))
            setdiff!(x, f.range)
            isempty(x) && break
        end
    end
    acc
end

@fastmath @inbounds function log_expectation!(pr::MaxEnt{S,T}, x::S, buffer::S, ctx::MaxEntContext{S}) where {S,T}
    p = factorize_skip_singletons!(+, zero(T), pr, x) do i
        copy!(buffer, x)
        intersect!(buffer, pr.factor[i].range)
        log(expectation(pr.factor[i], buffer, ctx))
    end
    p + sum(i -> pr.singleton_log_frequency[i], x; init=zero(T))
end

@fastmath @inbounds function expectation!(pr::MaxEnt{S,T}, x::S, buffer::S, ctx::MaxEntContext{S}) where {S,T}
    p = factorize_skip_singletons!(*, one(T), pr, x) do i
        copy!(buffer, x)
        intersect!(buffer, pr.factor[i].range)
        expectation(pr.factor[i], buffer, ctx)
    end
    p * prod(i -> pr.singleton_frequency[i], x; init=one(T))
end

@fastmath @inbounds @inline function isallowed!(pr::MaxEnt, t, max_size, max_width)
    s, w = factorize!((u, v) -> u .+ v, (1, 0), pr, t) do i::Int
        length(pr.factor[i].set), length(pr.factor[i].singleton)
    end
    (s < max_size) && (w < max_width)
end

function insert_pattern!(p::MaxEnt{S,T}, frequency, t, max_size, max_width, ctx::Vector{MaxEntContext{S}}) where {S,T}
    remaining_singletons = copy(t)
    factor_selection = Vector{Int}()
    factorize!(p, remaining_singletons) do i
        push!(factor_selection, i)
    end
    next = create_factor(p, factor_selection, remaining_singletons)
    if length(next.range) >= max_width || (length(next.set) + 1) >= max_size
        return false
    end
    insert!(next, frequency, copy(t))
    fit!(next, ctx)
    for i in factor_selection
        empty!(p.factor[i].range)
    end
    filter!(x -> !isempty(x.range), p.factor)
    push!(p.factor, next)
    true
end

patterns(p::MaxEntModel) = [s for f in p.factor for s in f.set]

function log_expectation_ts!(p::MaxEnt{S,T}, x::S, xbuffer, buffer, ctx; tid=Threads.threadid()) where {S,T}
    log_expectation!(p, copy!(xbuffer[tid], x), buffer[tid], ctx[tid])
end
function expectation_ts!(p::MaxEnt{S,T}, x::S, xbuffer, buffer, ctx; tid=Threads.threadid()) where {S,T}
    expectation!(p, copy!(xbuffer[tid], x), buffer[tid], ctx[tid])
end
function isforbidden_ts!(p, x, max_factor_size, max_factor_width, A; tid=Threads.threadid())
    !isallowed!(p, copy!(A[tid], x), max_factor_size, max_factor_width)
end
function isforbidden_ts!(ps::Vector{M}, x, max_factor_size, max_factor_width, A; tid=Threads.threadid()) where {M<:MaxEntModel}
    !all(ps) do p
        isallowed!(p, copy!(A[tid], x), max_factor_size, max_factor_width)
    end
end
isforbidden!(p, x, max_factor_size, max_factor_width, A) = !isallowed!(p, copy!(A, x), max_factor_size, max_factor_width)
function isforbidden!(ps::Vector{M}, x, max_factor_size, max_factor_width, A) where {M<:MaxEntModel}
    !all(ps) do p
        isallowed!(p, copy!(A, x), max_factor_size, max_factor_width)
    end
end
