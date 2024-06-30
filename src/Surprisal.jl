module Surprisal
using StatsBase
export surprisal_sum, find_tail_indices

function surprisal_sum(ecdf, X; use_abs=true)
	n,m = size(X)
	@assert length(ecdf) == m		# one ecdf for each column
	surpr_vec = zeros(n)
	for i in 1:n
		sum = 0.0
		x = use_abs ? abs.(X[i,:]) : X[i,:]
		for j in 1:m
			sum += -log(1-ecdf[j](x[j]))
		end
		surpr_vec[i] = sum
	end
	return surpr_vec
end

# upper tail prob for cutoff
find_tail_indices(eCDF, data, cutoff) = findall(<(cutoff), 1 .- eCDF.(data))

end # module Surprisal
