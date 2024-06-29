module Data

using Random, Statistics, Distributions, StatsBase
export generate_data, digitize_matrix, pairwise_diffs, pairwise_diffs_top


# returns data with observations in rows and features in columns
# mean_scale = 0 centers all dimensions on 0
function generate_data(n_samples, n_dimensions, n_anomaly_processes, anomaly_ratio;
			mean_scale=0.0, rstate=nothing, eta_normal=1.0, eta_anomaly=1.0)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		println("rstate for generate_data:")
		println(rstate)
	end
	copy!(Random.default_rng(), rstate)

    function random_correlation_matrix(n, eta)
        distn = LKJ(n, eta)
        return rand(distn)
    end

    # Normal data
    normal_corr = random_correlation_matrix(n_dimensions, eta_normal)
    normal_mean = mean_scale * randn(n_dimensions)
    normal_dist = MvNormal(normal_mean, normal_corr)

    # Anomaly data
    anomaly_corrs = [random_correlation_matrix(n_dimensions, eta_anomaly) for _ in 1:n_anomaly_processes]
    anomaly_dists = [MvNormal(mean_scale * randn(n_dimensions), corr) for corr in anomaly_corrs]

    # Generate data
    n_anomaly = round(Int, n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    X_normal = rand(normal_dist, n_normal)
    X_anomaly = hcat([rand(dist, round(Int, n_anomaly / n_anomaly_processes)) for dist in anomaly_dists]...)

    # Ensure X_anomaly has exactly n_anomaly columns
    if size(X_anomaly, 2) < n_anomaly
        X_anomaly = hcat(X_anomaly, rand(anomaly_dists[end], n_anomaly - size(X_anomaly, 2)))
    elseif size(X_anomaly, 2) > n_anomaly
        X_anomaly = X_anomaly[:, 1:n_anomaly]
    end

   	X = hcat(X_normal, X_anomaly)
	y = vcat(falses(n_normal), trues(n_anomaly))

    return X', y, normal_mean, normal_corr, anomaly_corrs
end

# take data matrix calc pairwise diffs based on normal mean and corr matrix
function pairwise_diffs(X, mean, corr)
  n, m = size(X)
  result = zeros(n, Int(m * (m - 1) / 2))
  col_idx = 1
  for i in 1:m-1
    for j in i+1:m
      if corr[i, j] > 0
        result[:, col_idx] = (X[:, i] .- mean[i]) .- (X[:, j] .- mean[j])
      else
        result[:, col_idx] = (X[:, i] .- mean[i]) .+ (X[:, j] .- mean[j])
      end
      col_idx += 1
    end
  end
  return result
end

# pick top correlated pairs, top is number of pairs, 0 is all
function pairwise_diffs_top(X, mean, corr; top=0)
    n, m = size(X)

    if top > 0 
        # Find indices of top absolute correlations (excluding diagonal)
        top_indices = sortperm(abs.(corr - I(m)), rev=true)[1:top]  
        # Convert linear indices to row, col pairs
        rows = ((top_indices .- 1) .% m) .+ 1
        cols = floor.((top_indices .- 1) ./ m) .+ 1
    else
        # Correct way to generate all unique pairs:
        rows = [i for i in 1:m-1 for j in i+1:m]
        cols = [j for i in 1:m-1 for j in i+1:m] 
    end

    result = zeros(n, length(rows)) 
    for k in 1:length(rows)
        i, j = rows[k], cols[k]
        if corr[i, j] > 0
            result[:, k] = (X[:, i] .- mean[i]) .- (X[:, j] .- mean[j])
        else
            result[:, k] = (X[:, i] .- mean[i]) .+ (X[:, j] .- mean[j])
        end
    end

    return result
end

# return indices of below diagonal elements sorted by absolute values
function sorted_below_diagonal_indices(mat)
    n = size(mat, 1)
    indices = [(i, j) for i in 2:n for j in 1:(i-1)] 
    sort!(indices, by=pair -> abs(mat[pair[1], pair[2]]), rev=true)
    return indices
end

# for matrix X and vector y of 0/1 labels, return a new matrix in which all rows
# have an associated 0 label
normal_data(X,y) = X[findall(y .== 0), :]

# for matrix X with obs in rows and features in cols and vector y of 0/1 labels
# select only rows with label == 0, then return mean vec, corr matrix
function normal_mean_corr(X,y)
	Xn = normal_data(X,y)
	means = mean(Xn, dims=1)
	corr = cor(Xn)
	return means, corr
end

# get empirical cdf by col, use_abs for cdf of absolute values
function ecdf_matrix(data; use_abs=true)
  num_cols = size(data, 2)
  eCDF_vector = Vector{ECDF{Vector{Float64}, StatsBase.Weights{Float64, Float64, Vector{Float64}}}}(undef, num_cols)

  for col in 1:num_cols
    eCDF_vector[col] = use_abs ? ecdf(abs.(data[:, col])) : ecdf(data[:, col])
  end

  return eCDF_vector
end

digitize_matrix(X) = X .>= 0

# threshold for digitizing, one threshold for each row
# matrix has features in rows and obs in columns
function digitize_matrix(X, thresholds)
	@assert is_matrix(X)
	@assert size(X)[1] == size(thresholds)[1]
    r, c = size(X)
    bits = BitArray(undef, r, c)
    for i in 1:r
        bits[i,:] .= X[i,:] .>= thresholds[i]
    end
    return bits
end

is_matrix(obj) = isa(obj, AbstractMatrix) && ndims(obj) == 2

function dataMatrix(normal, anomaly)
	@assert is_matrix(normal)	"normal is not a matrix"
	@assert is_matrix(anomaly)	"anomaly is not a matrix"
	@assert size(normal,1) == size(anomaly,1)	"number of rows (features) must be same"
	normal0 = vcat(normal,zeros(size(normal,2))')  
	anomaly1 = vcat(anomaly,ones(size(anomaly,2))')
	return hcat(normal0,anomaly1)
end

end # module Data
