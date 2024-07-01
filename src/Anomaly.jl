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
export temporal, multisensor, generate_data, digitize_matrix, pairwise_diffs_top,
		mean_corr, normal_data, anomaly_data, center_data, ecdf_matrix, ecdf, median_p, score_p,
		oneR_analysis, xgb_analysis, surprisal_sum, find_tail_indices, select_top_mean_columns,
		vec2matrix, calculate_metrics, get_metrics, optimize_thresholds


end # module Anomaly
