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
X,y,nm,am,nc=generate_data(10000,10,0.1;mean_scale=0.2);
results=oneR_analysis(hcat(X,y)')
Boost.plot_all_roc_curves(hcat(X,y)')

# XGBoost figure for manuscript
using Anomaly, Random, Plots
rstate=Random.Xoshiro(0xeaf747279f8ff889, 0xe40e689479627f4c, 0x146f8a31fd37d743,
		0x0ac6d49d37d1ad50, 0xa30788b9f260b0eb);
pl,df=plot_f1(; trees=2:20, depth=2:6, show_legend=false, data_size=1e6,
		features=exp2range(2:5), mean_scale=0.05*exp2range(1:5), rstate=rstate, smooth=false);
df_write(df, "/Users/steve/Desktop/df.arrow");
# rename file to df_full.arrow or xgboost.arrow to avoid overwriting
df = df_read("/Users/steve/Desktop/xgboost.arrow");
pl=plot_f1_trends(df; smooth=false)
savefig(pl, "/Users/steve/Desktop/xgboost.pdf")

# Encoder figure showing scatter plot in 2D projection
using Anomaly, Random, Plots
rstate=Random.Xoshiro(0x88eb0947697305c4, 0xde316ffed0db551b, 0x234055d83576e283,
		0xf68b1e95960ec797, 0xa30788b9f260b0eb);
X, y, nm, am, nc = Anomaly.generate_data(100000, 32, 0.1; mean_scale=1.6, rstate=rstate);
f1, pl = encoder(X[:,1:4],y;twoD=true,rstate=rstate,show_results=true,num_epoch=10000)
savefig(pl, "/Users/steve/Desktop/enc_scatter.pdf")


# Encoder figure comparing mean scale and num features for manuscript
using Anomaly, Random, Plots
rstate=Random.Xoshiro(0xeaf747279f8ff889, 0xe40e689479627f4c, 0x146f8a31fd37d743,
		0x0ac6d49d37d1ad50, 0xa30788b9f260b0eb);
df = encoder_loop(;n=2:5, mean_scale=0.05*exp2range(1:5), rstate=rstate, data_size=1e5, num_epoch=10000);

df_write(df, "/Users/steve/Desktop/df.arrow");
# rename file to encoder.arrow to avoid overwriting
df = df_read("/Users/steve/Desktop/encoder.arrow");
pl = plot_encoder(df);
savefig(pl, "/Users/steve/Desktop/encoder.pdf")

# loop through mean_scale values, use this for figure for encoder iterative feature selection
using Anomaly, Random
rstate=Random.Xoshiro(0xeaf747279f8ff889, 0xe40e689479627f4c, 0x146f8a31fd37d743,
		0x0ac6d49d37d1ad50, 0xa30788b9f260b0eb);
# rstate=nothing
# run mean_scale 1:1, 2:2, ..., 5:5 separately and then combine df's w/vcat
df = feature_loop_backward(32, 1; mean_scale=0.05*exp2range(1:5), rstate=rstate,
                                               show_rstate=false, data_size=1e5, num_epoch=5000);

# update plotting of interative feature search
df_write(df, "/Users/steve/Desktop/iter_f.arrow");
#df = df_read("/Users/steve/Desktop/iter_f_1.arrow");
pl = plot_features(df; n_features=[4,8,16,32,1]);
savefig(pl, "/Users/steve/Desktop/iter_f.pdf")

############################################################################################


# XGBoost analysis, good way to approximate maximum information available in data
# generate_data(samples, features, anomaly_proc, anomaly_ratio; mean_scale=0.0, η_normal=1.0, η_anomaly=1.0)

# plot F1, features increase along rows, mean_scale along cols, rising curves for increasing
# depth per tree, defaults: trees=2:20, depth=2:6, show_legend=false, data_size=1e5, 
#		features=exp2range(2:5), mean_scale=0.05*exp2range(1:5)
# NOTE: different random seeds give significantly different magnitudes for F1 because there is only
# one instance of the typical data generator and correlation structure. However the relative trends are
# consistent, and we are only interested in those relative trends.
# Use this seed if consistent underlying data required, otherwise use rstate=nothing

# Various XGBoost plots and tests
using Anomaly, Random, Plots
rstate=Random.Xoshiro(0xeaf747279f8ff889, 0xe40e689479627f4c, 0x146f8a31fd37d743,
		0x0ac6d49d37d1ad50, 0xa30788b9f260b0eb);
pl,df=plot_f1(; trees=2:20, depth=2:6, show_legend=false, data_size=1e5,
		features=exp2range(3:4), mean_scale=0.05*exp2range(4:5), rstate=rstate, smooth=false);
pl,df=plot_f1(; trees=2:20, depth=2:6, show_legend=false, data_size=1e6,
		features=exp2range(2:5), mean_scale=0.05*exp2range(1:5), rstate=rstate, smooth=true);

df_write(df, "/Users/steve/Desktop/df.arrow");
df = df_read("/Users/steve/Desktop/df_full.arrow");

pl=plot_f1(df; smooth=true)
pl=plot_f1_trends(df; smooth=false)

# REDO GENERATING DATA FOR PLOTS: GENERATE ONE MATRIX ONLY; USE SUBSETS OF X FOR CHANGING NUMBER
# OF FEATURES; RESCALE FOR CHANGING MEAN_SCALE VALUES

# various examples
X,y,nm,am,nc=generate_data(100000,20,0.1; mean_scale=0.2);
bst, dtest, stats = xgb_analysis(X,y; trees=100, depth=6);

X,y,nm,am,nc=generate_data(1000000,20,0.1; mean_scale=0.5);
bst, dtest, stats = xgb_analysis(X,y; trees=3, depth=1);

print_tree_stats(bst)
print_all_trees(bst)
make_graphs(bst)


############################################################################################

# Following show the information in various types of OneR analyses, always less than xgboost
# If only information is in individual feature means, does OneR does OK. 
# Can also pick up pairwise correlation info in modified type of OneR (see below). 
# However, comparing to xgboost shows that much information is in broader multivariate info

# Surprisal analysis of means, abs use assumes symmetric distns; n rows as obs, m cols as variables
X,y,nm,am,nc=generate_data(100000,20,0.1; mean_scale=0.5);
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
X,y,nm,am,nc=generate_data(100000,20,0.1; mean_scale=0.5);
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

metr=get_metrics(mcdf_abs, Xnc, Xac, fill(0.05,size(Xnc,2)), 3; display=true);

