function any_bits(f, s::BitSet, t::BitSet)
    a1, b1 = s.bits, s.offset
    a2, b2 = t.bits, t.offset
    l1, l2 = length(a1), length(a2)
    bdiff  = b2 - b1
    @inbounds for i in max(1, 1 + bdiff):min(l1, l2 + bdiff)
        f(a1[i], a2[i - bdiff]) && return true
    end
    return false
end
intersects(u::BitSet, v::BitSet) = @fastmath any_bits((a, b) -> (a & b) != 0, u, v)
intersects(u::Int, v::BitSet) = in(u, v)
intersects(v::BitSet, u::Int) = in(u, v)
intersects(v::Int, u::Int) = u == v

function reduce_bits(f, r, init, s::BitSet, t::BitSet)
    a1, b1 = s.bits, s.offset
    a2, b2 = t.bits, t.offset
    l1, l2 = length(a1), length(a2)
    bdiff  = b2 - b1
    acc    = init
    @inbounds for i in max(1, 1 + bdiff):min(l1, l2 + bdiff)
        acc = r(acc, f(a1[i], a2[i - bdiff]))
    end
    acc
end
function intersection_size(u::BitSet, v::BitSet)
    @fastmath reduce_bits((a, b) -> count_ones(a & b), +, 0, u, v)
end
union_size(u::BitSet, v::BitSet) = length(u) + length(v) - intersection_size(u, v)
union_size(u::Int, v::BitSet) = length(v) + !(u in v)
union_size(v::BitSet, u::Int) = union_size(u, v)
