module Anomaly
include("Temporal.jl")
using .Temporal
include("MultiSensor.jl")
using .MultiSensor
include("Data.jl")
using .Data
include("Surprisal.jl")
using .Surprisal
include("Boost.jl")
using .Boost
include("Encoder.jl")
using .Encoder
export temporal, multisensor, generate_data, digitize_matrix, pairwise_diffs_top,
		mean_corr, normal_data, anomaly_data, center_data, ecdf_matrix, ecdf, median_p, score_p,
		oneR_analysis, xgb_analysis, surprisal_sum, find_tail_indices, select_top_mean_columns,
		vec2matrix, calculate_metrics, get_metrics, optimize_thresholds, print_all_trees,
		make_graphs, print_tree_stats, plot_f1, exp2range, df_write, df_read, plot_f1_trends,
		adjust_mean_scale, encoder, encoder_loop, plot_encoder, iterative_feature_search,
		feature_loop, plot_features, feature_loop_backward


end # module Anomaly
