using SparseArrays: findnz, AbstractSparseMatrix
using ThreadsX: ThreadsX

mutable struct Candidate{T}
    set     :: T
    rows    :: T
    support :: Int
    score   :: Float64

    Candidate{T}(set::T=T(), rows::T=T(), support=0, score=0.0) where {T} = new{T}(set, rows, support, score)
    Candidate{T}(other::Candidate{T}) where {T} = new{T}(T(other.set), T(other.rows), other.support, other.score)
    Candidate{T}(i::Int, maxsize::Int) where {T} = new{T}(T(i), sizehint!(T(), maxsize), 0, 0.0)
end

function Base.empty!(s::Candidate)
    empty!(s.set)
    empty!(s.rows)
    s.support = 0
    s.score   = 0
end

Base.isless(a::Candidate{T}, b::Candidate{T}) where {T} = isless(b.score, a.score)

function join!(c::Candidate{T}, a::Candidate{T}, b::Candidate{T}, min_support, score) where {T}
    c.score = 0
    issubset(b.set, a.set) && return c
    copy!(c.rows, a.rows)
    intersect!(c.rows, b.rows)
    c.support = length(c.rows)
    c.support < min_support && return c
    copy!(c.set, a.set)
    union!(c.set, b.set)
    c.score = score(c)
    return c
end

function init_singletons(::Type{T}, X::AbstractMatrix) where {T}
    singletons = [Candidate{T}(i, size(X, 1)) for i in 1:size(X, 2)]
    for I in CartesianIndices(X)
        i, j = I[1], I[2]
        if X[i, j] != 0
            push!(singletons[j].rows, i)
        end
    end
    for s in singletons
        s.support = length(s.rows)
    end
    singletons
end

function init_singletons(::Type{T}, X::AbstractSparseMatrix) where {T}
    singletons = [Candidate{T}(i, size(X, 1)) for i in 1:size(X, 2)]
    I, J, _ = findnz(X)
    for k in eachindex(I)
        i, j = I[k], J[k]
        push!(singletons[j].rows, i)
    end
    for s in singletons
        s.support = length(s.rows)
    end
    singletons
end

function init_singletons(::Type{T}, X::AbstractVector{S}) where {T, S}
    singletons = [Candidate{BitSet}(i, size(X, 1)) for i in 1:maximum(x -> maximum(x; init=0), X)]
    for (i, x) in enumerate(X), j in x
        push!(singletons[j].rows, i)
    end
    for s in singletons
        s.support = length(s.rows)
    end
    singletons
end

function join_singletons(a::Candidate{T}, b::Candidate{T}, score)::Candidate{T} where {T}
    rows = intersect(a.rows, b.rows)
    c = Candidate{T}(union(a.set, b.set), rows, length(rows), 0)
    c.score = score(c)
    c
end

function init_candidates(I::Vector{Candidate{T}}, score; min_support=2) where {T}
    ThreadsX.collect(join_singletons(I[i], I[j], score)
                     for i in 1:length(I)
                     for j in (i + 1):length(I)
                     if intersection_size(I[i].rows, I[j].rows) >= min_support)::Vector{Candidate{T}}
end

function argminx(C::Vector{T})::Int64 where {T}
    ThreadsX.reduce(eachindex(C); init=isempty(C) ? 0 : 1) do i::Int, j::Int
        C[i]::T < C[j]::T ? i : j
    end
end

mutable struct Lattice{Candidate}
    singletons :: Vector{Candidate}
    candidates :: Vector{Candidate}
    next       :: Vector{Candidate}
    at         :: Int64

    function Lattice{Candidate{SetType}}(X, score; min_support=2) where SetType
        S = init_singletons(SetType, X)
        C = init_candidates(S, score; min_support=min_support)
        N = Candidate{SetType}[Candidate{SetType}() for _ in eachindex(S)]
        new{Candidate{SetType}}(S, C, N, argminx(C))
    end
end

function unordered_popat!(xs, i)
    if i != length(xs)
        xs[i], xs[end] = xs[end], xs[i]
    end
    pop!(xs)
end

top(L::Lattice) = L.candidates[L.at]
function Base.pop!(L::Lattice{T}) where {T}
    x = unordered_popat!(L.candidates, L.at)
    L.at = argminx(L.candidates)
    x
end
has_candidates(L::Lattice) = !isempty(L.candidates)

function popuntil!(L::Lattice)
    x = pop!(L)
    while has_candidates(L) && top(L).set == x.set
        pop!(L)
    end
    x
end

function expand!(L::Lattice{T}, score, max_steps=10; min_support=1, max_depth=nothing) where {T}
    isempty(L.candidates) && return false
    before = length(L.candidates)
    node = top(L)
    for _ in 1:max_steps
        cur::T = node
        cur.score <= 0 && break
        max_depth !== nothing && length(cur.set) >= max_depth && break
        ThreadsX.foreach(eachindex(L.singletons)) do i
            join!(L.next[i], cur, L.singletons[i], min_support, score)
        end
        local before = length(L.candidates)
        append!(L.candidates, T(n) for n in L.next if n.score > 0)
        before == length(L.candidates) && break
        next = minimum(L.next)
        if next <= cur
            node = next
        else
            break
        end
    end
    L.at = argminx(L.candidates)
    before != length(L.candidates)
end

function prune!(L::Lattice, pred)
    filter!(!(pred), L.candidates)
    L.at = argminx(L.candidates)
end

function update!(L::Lattice, s::Candidate{T}, score, pred) where {T}
    ThreadsX.foreach(L.candidates) do t
        if intersects(s.set, t.set)
            t.score = score(t)
        elseif pred(t)
            t.score = 0
        end
    end
    filter!(x -> x.score > 0, L.candidates)
    L.at = argminx(L.candidates)
end

function reevaluate_candidates!(L::Lattice, score)
    ThreadsX.foreach(L.candidates) do t
        t.score = score(t)
    end
    filter!(s -> s.score > 0, L.candidates)
    L.at = argminx(L.candidates)
end

function discover_patterns!(L::Lattice, score, isforbidden, report; maxiter=typemax(UInt64), max_expansions=10,
                            max_discoveries=typemax(UInt64), min_support=2, max_seconds=Inf, max_patience=100)
    reevaluate_candidates!(L, score)
    expand!(L, score, max_expansions; min_support=min_support)
    discoveries = 0
    patience    = max_patience
    isfin       = !isinf(max_seconds)
    t0          = isfin ? time() : Inf
    for _ in 1:maxiter
        (isempty(L.candidates) || top(L).score <= 0) && break
        x = popuntil!(L)
        if report(x)
            update!(L, x, score, isforbidden)
            expand!(L, score, max_expansions; min_support=min_support)
            discoveries += 1
            discoveries >= max_discoveries && break
            patience = min(patience + 1, max_patience)
        else
            patience -= 1
            patience <= 0 && break
        end
        isfin && (time() - t0 > max_seconds) && break
    end
end
