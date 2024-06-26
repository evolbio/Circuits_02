module Boost
include("Data.jl")
using .Data
using Plots, DataFrames, MLBase, ROCAnalysis
export oneR_analysis, oneR_analysis_old

# Study boosted decision trees
# Use artificial data from Data.jl

using ROCAnalysis
using DataFrames

function oneR_analysis(data_matrix)
    n_features = size(data_matrix, 1) - 1
    labels = data_matrix[end, :]
    int_labels = Int.(labels)
    results = DataFrame(feature = Int[], AUC = Float64[], F1 = Float64[], Best_Threshold = Float64[])
    
    for i in 1:n_features
        feature = data_matrix[i, :]
        
        # Generate thresholds based on feature values
        thresholds = sort(unique(feature))
        
        # Find the best threshold for maximizing F1 score
        best_f1 = 0.0
        best_threshold = 0.0
        
        for threshold in thresholds
            predicted_labels = feature .>= threshold
            f1 = f1score2(int_labels, predicted_labels)
            
            if f1 > best_f1
                best_f1 = f1
                best_threshold = threshold
            end
        end
        
        # Compute ROC curve using true labels and predicted scores
        predicted_scores = feature
        roc_curve = ROCAnalysis.roc(labels, predicted_scores)
        
        # Compute AUC using the ROC curve
        auc = AUC(roc_curve)
        
        # Store results
        push!(results, (i, auc, best_f1, best_threshold))
    end
    
    return results
end

# Helper function to compute F1 score
function f1score2(true_labels, predicted_labels)
    tp = sum((true_labels .== 1) .& (predicted_labels .== 1))
    fp = sum((true_labels .== 0) .& (predicted_labels .== 1))
    fn = sum((true_labels .== 1) .& (predicted_labels .== 0))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return 2 * (precision * recall) / (precision + recall)
end

# Function to perform OneR analysis and find the best threshold for F1 score
function oneR_analysis_old(data_matrix)
    n_features = size(data_matrix, 1) - 1
    labels = data_matrix[end, :]  # Keep labels as Float64 for ROCAnalysis
    int_labels = Int.(data_matrix[end, :])  # Convert labels to Int for MLBase
    results = DataFrame(feature = Int[], AUC = Float64[], F1 = Float64[], Best_Threshold = Float64[])

    for i in 1:n_features
        feature = data_matrix[i, :]

        # Compute AUC by evaluating the ROC curve with ROCAnalysis
        roc_curve = ROCAnalysis.roc(labels, feature)
        auc = ROCAnalysis.auc(roc_curve)

        # Generate and sort thresholds
        thresholds = sort(unique(feature))

        # Compute ROC instances for multiple thresholds with MLBase
        roc_instances = MLBase.roc(int_labels, feature, thresholds)

        # Find the best threshold for maximizing F1 score
        best_f1 = 0.0
        best_threshold = 0.0

        for (j, roc_num) in enumerate(roc_instances)
            f1 = f1score(roc_num)
            if f1 > best_f1
                best_f1 = f1
                best_threshold = thresholds[j]
            end
        end

        # Store results
        push!(results, (i, auc, best_f1, best_threshold))
    end

    return results
end

# Function to compute TPR and FPR from ROCNums
function tpr_fpr(roc_nums)
    tpr = recall(roc_nums)
    fpr = false_positive_rate(roc_nums)
    return tpr, fpr
end

# Function to plot ROC curves for all features
function plot_all_roc_curves(data_matrix)
    labels = data_matrix[end, :]

    # Create a grid layout for 8 ROC curves
    plot_layout = @layout [grid(4, 2)]

    # Create a plot object with the specified layout
    p = plot(layout = plot_layout, size = (800, 1200))

    for i in 1:8
        feature = data_matrix[i, :]

        # Compute ROC curve
        roc_instances = MLBase.roc(Int.(labels), feature, 100)

        # Extract true positive and false positive rates
        tpr = [recall(roc_num) for roc_num in roc_instances]
        fpr = [false_positive_rate(roc_num) for roc_num in roc_instances]

        # Plot ROC curve for the current feature
        plot!(p, fpr, tpr, label="Feature $i", xlabel="False Positive Rate", ylabel="True Positive Rate", title="Feature $i", subplot=i)
        plot!(p, [0, 1], [0, 1], linestyle=:dash, label="Random Classifier", subplot=i)
    end

    display(p)
end

function plot_roc_curves(data_matrix, results)
    n_features = size(data_matrix, 1) - 1
    labels = data_matrix[end, :]
    
    # Create a grid of subplots for ROC curves
    plot_grid = plot(layout=(2, 4), legend=false, size=(800, 400))
    
    for i in 1:n_features
        feature = data_matrix[i, :]
        auc = results[i, :AUC]
        
        # Compute ROC curve using true labels and predicted scores
        predicted_scores = feature
        roc_curve = ROCAnalysis.roc(labels, predicted_scores)
        
        # Extract false positive rates and true positive rates from the ROC curve
        fprs = roc_curve.fps
        tprs = roc_curve.tps
        
        # Plot the ROC curve in the corresponding subplot
        plot!(plot_grid[i], fprs, tprs,
              title="Feature $i (AUC = $(round(auc, digits=3)))",
              xlabel="False Positive Rate", ylabel="True Positive Rate")
    end
    
    # Display the plot grid
    display(plot_grid)
end

end # module Boost
