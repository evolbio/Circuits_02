module Data

using Random, Statistics, Distributions
export generate_data, digitize_matrix


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

    return X', y, normal_corr, anomaly_corrs
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
