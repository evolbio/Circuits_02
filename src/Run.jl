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


# Generate artificial data and do oneR analysis
using Data, Boost, Surprisal

# OneR analysis
X,y,nc=generate_data(10000,10,0.1;mean_scale=0.2);
results=oneR_analysis(hcat(X,y)')
Boost.plot_all_roc_curves(hcat(X,y)')

# XGBoost analysis
# generate_data(samples, features, anomaly_proc, anomaly_ratio; mean_scale=0.0, η_normal=1.0, η_anomaly=1.0)
X,y,nm,nc=generate_data(100000,20,0.1; mean_scale=0.2);
bst, dtest = xgb_analysis(X,y; trees=100, depth=6);

# Surprisal analysis of means, abs use assumes symmetric distns; n rows as obs, m cols as variables
X,y,nm,nc=generate_data(100000,20,0.1; mean_scale=0.5);
#bst, dtest = xgb_analysis(X,y; trees=100, depth=6);
Xn=normal_data(X,y);							# get normal obs, drop anomalies
means, corr = mean_corr(Xn);					# values for normal obs
Xn0 = center_data(Xn, means);					# centers data in each col to have zero mean
mcdf_abs = ecdf_matrix(Xn0);					# empirical cdf of abs values
surp_vec = surprisal_sum(mcdf_abs, Xn0);		# for each obs by summing surprisals across variables (rows)
ecdf_surprisal_sum = ecdf(surp_vec);
ti = find_tail_indices(ecdf_surprisal_sum, surp_vec, 0.05);		# in upper tail, false positive rate
length(ti)/length(surp_vec) 									# should be false positive rate

Xa = anomaly_data(X,y);
Xa0 = center_data(Xa, means);
surp_vec = surprisal_sum(mcdf_abs, Xa0);
ti = find_tail_indices(ecdf_surprisal_sum, surp_vec, 0.05);
length(ti)/length(surp_vec)

# median of surprisal p values as measure instead of p value for sum of surprisals
median_vec = median_p(mcdf_abs, Xa0);
median_vec_N = median_p(mcdf_abs, Xn0);

# more typical ensemble method based on number of individual features that are significant
score_vec = score_p(mcdf_abs, Xn0, fill(0.05,size(Xn0,2)), 3)
score_vec = score_p(mcdf_abs, Xa0, fill(0.05,size(Xa0,2)), 3)

# analyze pairwise correlations
Xd, t_idx = pairwise_diffs_top(X, means, corr; top=20);
Xdn=normal_data(Xd,y);							# get normal obs, drop anomalies
means, corr = mean_corr(Xdn);					# values for normal obs
Xdn0 = center_data(Xdn, means);					# centers data in each col to have zero mean
mcdf_abs = ecdf_matrix(Xdn0);					# empirical cdf of abs values

Xda = anomaly_data(Xd,y);
Xda0 = center_data(Xda, means);

score_vec = score_p(mcdf_abs, Xdn0, fill(0.05,size(Xdn0,2)), 3)
score_vec = score_p(mcdf_abs, Xda0, fill(0.05,size(Xda0,2)), 3)

# combine top means and top correlation pairs into single ensemble
X,y,nm,nc=generate_data(100000,20,0.1; mean_scale=0.5);
Xn=normal_data(X,y);							# get normal obs, drop anomalies
means, corr = mean_corr(Xn);					# values for normal obs
Xnm,m_means,m_idx=select_top_mean_columns(Xn, means, 9);	# get top 5 cols w/top deviations of mean from 0
Xnm0=center_data(Xnm, m_means);

Xd, t_idx = pairwise_diffs_top(X, means, corr; top=1);
Xdn=normal_data(Xd,y);							# get normal obs, drop anomalies
Xnc=hcat(Xnm0,Xdn);
mcdf_abs = ecdf_matrix(Xnc);					# empirical cdf of abs values for combined matrices

Xda=anomaly_data(Xd,y);
Xa=anomaly_data(X,y);
Xam=Xa[:,m_idx];
Xam0=center_data(Xam, m_means);
Xac=hcat(Xam0,Xda);

bst, dtest = xgb_analysis(vcat(Xnc,Xac),y; trees=100, depth=6);
best_st, best_at, precision, accuracy, recall, f1 = optimize_thresholds(Xnc, Xac, mcdf_abs);

#metr=get_metrics(mcdf_abs, Xnc, Xac, fill(0.05,size(Xnc,2)), 3; display=true);

# OneR ON SINGLE COLUMNS TO GET BEST THRESHOLD FOR EACH COLUMN, THEN SCORE AS 0/1. THEN LOOK AT VARIOUS WAYS TO AGGREGATE 0/1 SCORES FOR EACH COLUMN, FOR EXAMPLE, THRESHOLD ON SUM. BUT ALSO CONSIDER RUNNING DECISION TREE ON 0/1 VALUES OR AUTOENCODE TO GET FEATURES, ETC.


