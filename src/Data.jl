module Data
using Distributions, MultivariateStats, Random, Plots
export generateAnomaly, pca_test

# See ChaptGPT: Generate Random PCA mapping
# First, creates random data in n dimensions, PCA map to m<n dimensions for m=2.
# Second, creates normal and anomalous cloud of points in m dimensions.
# Third, maps data from m dimensions to n dimensions using inverse PCA 
# Goal: provides n dimensional normal and anomalous data that can be separated at lower dim

function generateAnomaly(n; n_normal=200, n_anomaly=100, plot2d=false,
			dir_normal=[3, 7, 2], dir_anom=[8, 2, 3], rstate=nothing)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		println(rstate)
	end
	copy!(Random.default_rng(), rstate)
	# Generate the points in (n+1) dimensions and drop the last dimension
	dirichlet_params = rand(1:10, n + 1)
	high_dim_plus1 = rand(Dirichlet(dirichlet_params), 2000)
	high_dim = high_dim_plus1[1:n, :]

	# PCA on the high-dimensional points to create the inverse mapping model
	pca_model = fit(PCA, high_dim; maxoutdim=2)

	# Generate normal and anomalous points in 3D (use 3 parameters), then drop the last dimension
	normal_3D = rand(Dirichlet(dir_normal), 200)
	normal_2D = normal_3D[1:2, :]

	anomaly_3D = rand(Dirichlet(dir_anom), 100)
	anomaly_2D = anomaly_3D[1:2, :]

	normal_nD = MultivariateStats.reconstruct(pca_model, normal_2D)
	anomaly_nD = MultivariateStats.reconstruct(pca_model, anomaly_2D)

	pl = nothing
	if plot2d
		pl = scatter(normal_2D[1, :], normal_2D[2, :], color=:blue, label="Normal")
		scatter!(anomaly_2D[1, :], anomaly_2D[2, :], color=:red, label="Anomalous")
		plot!(title="2D Points", xlabel="X1", ylabel="X2")
		display(pl)
	end
	
	return pca_model, normal_nD, anomaly_nD, normal_2D, anomaly_2D, pl
end

pca_test(pca_model, input, target) = target ≈ predict(pca_model, input)

end # module Data
