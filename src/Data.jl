module Data

using Random, Statistics, Distributions, StatsBase, LinearAlgebra, Printf
export generate_data, digitize_matrix, pairwise_diffs_top, mean_corr,
		normal_data, anomaly_data, center_data, ecdf_matrix, ecdf, median_p, score_p,
		select_top_mean_columns, vec2matrix, calculate_metrics

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

# pick top correlated pairs, top is number of pairs, 0 is all
function pairwise_diffs_top(X, mean, corr; top=0)
    n, m = size(X)
    
    if top > 0 
        # Create a copy of corr and set upper triangle (including diagonal) to zero
        corr_abs = abs.(corr)
        corr_abs[triu!(trues(size(corr_abs)))] .= 0
        
        # Find indices of top absolute correlations (lower triangle only)
        top_indices = partialsortperm(vec(corr_abs), 1:min(top, div(m*(m-1), 2)), rev=true)
        
        # Convert linear indices to row, col pairs
        rows = (top_indices .- 1) .% m .+ 1
        cols = (top_indices .- 1) .÷ m .+ 1
    else
        # Generate all unique pairs, lower triangle only (i > j to ensure uniqueness)
        top_indices = pairs = [(i, j) for i in 2:m for j in 1:i-1]
        rows = first.(pairs)
        cols = last.(pairs)
    end
    
    # Preallocate the result matrix
    result = zeros(n, length(rows))
    
    # Compute differences
    for k in 1:length(rows)
        i, j = rows[k], cols[k]
        if corr[i, j] > 0
            result[:, k] = (X[:, i] .- mean[i]) .- (X[:, j] .- mean[j])
        else
            result[:, k] = (X[:, i] .- mean[i]) .+ (X[:, j] .- mean[j])
        end
    end
    
    return result, top_indices
end

function select_top_mean_columns(X, means, k)
    # Ensure means is a vector
    means = vec(means)

    # Ensure k is not larger than the number of columns
    k = min(k, size(X, 2))
    
    # Get the indices of columns sorted by absolute mean values in descending order
    sorted_indices = sortperm(abs.(means), rev=true)
    
    # Select the top k indices
    top_k_indices = sorted_indices[1:k]
    
    # Select the corresponding columns from X
    X_selected = X[:, top_k_indices]
    
    # Also return the corresponding means if needed
    means_selected = means[top_k_indices]
    
    return X_selected, means_selected, top_k_indices
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
anomaly_data(X,y) = X[findall(y .== 1), :]

# for matrix X with only normal data (use normal_data()), return mean vec and corr matrix
mean_corr(X) = return mean(X, dims=1), cor(X)

# size(data) = (n,m) for n obs and m variables, size(col_means) = (1,m) a row vector as matrix
function center_data(data, col_means)
  c_means = vec(col_means)
  n, m = size(data)
  centered_data = zeros(n, m) 
  for col in 1:m
    centered_data[:, col] = data[:, col] .- c_means[col]
  end
  return centered_data
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

function median_p(ecdf, X; use_abs=true)
	n,m = size(X)
	@assert length(ecdf) == m		# one ecdf for each column
	median_vec = zeros(n)
	for i in 1:n
		feature_vec = zeros(m)
		x = use_abs ? abs.(X[i,:]) : X[i,:]
		for j in 1:m
			feature_vec[j] = 1-ecdf[j](x[j])
		end
		median_vec[i] = median(feature_vec)
	end
	return median_vec
end

# If a row has >=k p values <= p, then 1, otherwise 0
function score_p(ecdf, X, p, k; use_abs=true)
	n,m = size(X)
	@assert length(ecdf) == m		# one ecdf for each column
	score_vec = zeros(n)
	for i in 1:n
		feature_vec = zeros(m)
		x = use_abs ? abs.(X[i,:]) : X[i,:]
		for j in 1:m
			feature_vec[j] = 1-ecdf[j](x[j]) <= p ? 1 : 0
		end
		score_vec[i] = sum(feature_vec) >= k ? 1 : 0
	end
	return sum(score_vec)/length(score_vec)
end

function calculate_metrics(TP, TN, FP, FN; display=false)
  """
  Calculates precision, accuracy, recall, and F1 score from confusion matrix counts.

  Args:
    TP: True Positives
    TN: True Negatives
    FP: False Positives
    FN: False Negatives

  Returns:
    A tuple containing: (precision, accuracy, recall, f1)
  """

  precision = TP / (TP + FP)
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  recall = TP / (TP + FN)
  f1 = 2 * (precision * recall) / (precision + recall)
  
  if display
  	@printf("%11s%5.3f\n", "Precision: ", precision)
  	@printf("%11s%5.3f\n", "Accuracy: ", accuracy)
  	@printf("%11s%5.3f\n", "Recall: ", recall)
  	@printf("%11s%5.3f\n", "F1 Score: ", f1)
  end

  return precision, accuracy, recall, f1
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

vec2matrix(x) = reshape(x, (length(x), 1))
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
