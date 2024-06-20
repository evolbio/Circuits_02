using Anomaly, Plots, Random

# temporal adjustment of anomaly detection for single sensor
sol,pl = temporal();
savefig(pl,"/Users/steve/Desktop/anomaly.pdf")

# for publication figure
rstate = Xoshiro(0xa242084f5b199bea, 0x53a1df5ce28618fd, 0x89250c8d5127cb4e, 0x6b5f53c46218257b, 0xc5ed5ca8a7cca17e);
sol,pl = temporal(;rstate=rstate);
savefig(pl,"/Users/steve/Desktop/anomaly2.pdf")


# Multisensor system, optimize each sensor and combination of sensors
# for anomaly detection
# Parameters
n = 8
v = 40
mTypical = 100
mAnomaly = 120

# Perform optimization
optimized_params = multisensor(n, mTypical, mAnomaly, v)
println("Optimized Parameters: ", optimized_params)


# Generate artificial anomaly data by random PCA mapping
using Data

pca_model, normal_nD, anomaly_nD, normal_2D, anomaly_2D, pl = 
		generateAnomaly(8; plot2d=true);
pca_test(pca_model, normal_nD, normal_2D)

# fix random state
rstate = Xoshiro(0xa242084f5b199bea, 0x53a1df5ce28618fd, 0x89250c8d5127cb4e, 0x6b5f53c46218257b, 0xc5ed5ca8a7cca17e);
pca_model, normal_nD, anomaly_nD, normal_2D, anomaly_2D, pl = 
		generateAnomaly(8; plot2d=true, rstate=rstate);
