module Boost
include("Data.jl")
using .Data
using Plots, DataFrames, MLBase, ROCAnalysis, XGBoost, MLDataUtils, Random, Statistics,
		Base.Threads, JSON3, CairoMakie, Graphs, GraphMakie, NetworkLayout
export oneR_analysis, xgb_analysis, print_all_trees, make_graphs

# samples in rows, features in cols
function xgb_analysis(X, y; rstate=nothing, trees=100, depth=6)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		println(rstate)
	end
	copy!(Random.default_rng(), rstate)
	
	# Create a DataFrame for X to preserve feature names
	df = DataFrame(X, :auto)
	
	# Shuffle and split the data
	df, y = shuffleobs((df, y))
	(train_df, train_y), (test_df, test_y) = splitobs((df, y), at = 0.7)
	
	# Convert labels to Float64 for training
	train_y = Float64.(train_y)
	
	# Convert DataFrames to DMatrix
	dtrain = DMatrix(train_df, label=train_y)
	dtest = DMatrix(test_df, label=test_y)
	
	# Create and train the model
	# params as here: https://xgboost.readthedocs.io/en/stable/parameter.html
	bst = xgboost(dtrain, num_round=trees, max_depth=depth, eta=0.3, objective="binary:logistic",
					eval_metric="logloss")
	
	# Make predictions on the test set
	y_pred = XGBoost.predict(bst, dtest)
	y_pred_binary = y_pred .> 0.5
	
	# Calculate metrics
	accuracy = mean(y_pred_binary .== test_y)
	tp = sum(y_pred_binary .& test_y)
	fp = sum(y_pred_binary .& .!test_y)
	fn = sum(.!y_pred_binary .& test_y)
	recall_value = tp / (tp + fn)
	precision_value = tp / (tp + fp)
	f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
	
	# Print results
	println("Accuracy: ", round(accuracy, digits=4))
	println("Recall: ", round(recall_value, digits=4))
	println("Precision: ", round(precision_value, digits=4))
	println("F1 Score: ", round(f1_score, digits=4))
	
	# Display feature importance
	feature_importance = importance(bst)
	display(feature_importance)
	return bst, dtest
end

# takes data with features in rows and observations in columns
# last row has labels for normal (0) or anomaly (1)
function oneR_analysis(data_matrix)
    n_features = size(data_matrix, 1) - 1
    labels = data_matrix[end, :]
    int_labels = Int.(labels)
    results = DataFrame(feature = Int[], AUC = Float64[], F1 = Float64[], Best_Threshold = Float64[])
    
    Threads.@threads for i in 1:n_features
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

# Function to compute TPR and FPR from ROCNums
function tpr_fpr(roc_nums)
    tpr = recall(roc_nums)
    fpr = false_positive_rate(roc_nums)
    return tpr, fpr
end

# Function to plot ROC curves for all features
# Features in rows, obs in columns, last row as normal/anomaly labels as 0/1
function plot_all_roc_curves(data_matrix)
    labels = data_matrix[end, :]
    features = size(data_matrix)[1]-1

    # Create a grid layout for 8 ROC curves
    plot_layout = @layout [grid(round(Int,features/2), 2)]

    # Create a plot object with the specified layout
    p = plot(layout = plot_layout, size = (800, 1200))

    for i in 1:features
        feature = data_matrix[i, :]

        # Compute ROC curve
        roc_instances = MLBase.roc(Int.(labels), feature, 100)

        # Extract true positive and false positive rates
        tpr = [recall(roc_num) for roc_num in roc_instances]
        fpr = [false_positive_rate(roc_num) for roc_num in roc_instances]

        # Plot ROC curve for the current feature
        plot!(p, fpr, tpr, label="Feature $i", xlabel="False Positive Rate",
        		ylabel="True Positive Rate", title="Feature $i", subplot=i)
        plot!(p, [0, 1], [0, 1], linestyle=:dash, label="Random Classifier", subplot=i)
    end

    display(p)
end


# print trees from xgboost in text format
# Function to print a tree in a readable text format
function print_tree(tree, indent="")
    if haskey(tree, "children")
        println(indent, "Split: ", tree["split"], " < ", tree["split_condition"])
        println(indent, "Yes ->")
        print_tree(tree["children"][1], indent * "  ")
        println(indent, "No ->")
        print_tree(tree["children"][2], indent * "  ")
    else
        println(indent, "Leaf: ", tree["leaf"])
    end
end

function print_all_trees(model_result)
	model_dump = XGBoost.dump(model_result, fmap="", with_stats=true)
	for (i, tree) in enumerate(model_dump)
    	println("Tree $i:")
    	print_tree(tree)
    	println()
	end
end

# Function to convert XGBoost tree to a graphic format

function make_graphs(model_result; show_plot=true)
    model_dump = XGBoost.dump(model_result, fmap="", with_stats=true)
    
    n_trees = length(model_dump)
    n_cols = min(5, n_trees)  # Maximum 5 columns
    n_rows = ceil(Int, n_trees / n_cols)
    
    if show_plot
        # Create a figure with a grid of subplots for display
        fig = Figure(size=(200 * n_cols, 200 * n_rows))  # Adjust size as needed
        
        for (i, tree) in enumerate(model_dump)
            row = (i - 1) ÷ n_cols + 1
            col = (i - 1) % n_cols + 1
            ax = Axis(fig[row, col], title="Tree $i")
            plot_xgb_tree(tree, ax)
        end
        
        # Display the figure
        Makie.display(fig)
    else
        # Save individual PDF files for each tree
        for (i, tree) in enumerate(model_dump)
            fig = Figure()
            ax = Axis(fig[1, 1], title="Tree $i")
            plot_xgb_tree(tree, ax)
            save("xgboost_tree_$i.pdf", fig)
            println("Tree $i saved as xgboost_tree_$i.pdf")
        end
    end
end

function xgb_tree_to_graph(tree::JSON3.Object)
    g = SimpleDiGraph()
    properties = Any[]
    
    function walk_tree!(node, parent_id=nothing)
        add_vertex!(g)
        current_id = nv(g)
        
        if parent_id !== nothing
            add_edge!(g, parent_id, current_id)
        end
        
        if haskey(node, :split)
            label = "$(node.split) < $(round(node.split_condition, digits=2))\n$(round(node.gain, digits=2))"
            push!(properties, (:Node, label))
            
            walk_tree!(node.children[1], current_id)
            walk_tree!(node.children[2], current_id)
        else
            label = "$(round(node.leaf, digits=2))"  # Just the number, without "Leaf:"
            push!(properties, (:Leaf, label))
        end
    end
    
    walk_tree!(tree)
    return g, properties
end

Makie.@recipe(PlotXGBoostTree, tree) do scene
    Attributes(
        nodecolormap = :viridis,
        textcolor = :black,
        leafcolor = :green,
        nodecolor = :white,
    )
end

function Makie.plot!(plt::PlotXGBoostTree)
    tree = plt[:tree][]
    graph, properties = xgb_tree_to_graph(tree)
    
    labels = [p[2] for p in properties]
    node_colors = [p[1] == :Leaf ? :transparent : plt[:nodecolor][] for p in properties]
    label_colors = [p[1] == :Leaf ? plt[:leafcolor][] : plt[:textcolor][] for p in properties]
    
    node_sizes = [p[1] == :Leaf ? 0 : 40 for p in properties]
    
    graphplot!(plt, graph;
        layout=Buchheim(),
        nlabels=labels,
        node_color=node_colors,
        nlabels_color=label_colors,
        node_size=node_sizes,
        nlabels_align=(:center, :center),
        nlabels_textsize=10,
        nlabels_distance=0,
    )
end

function plot_xgb_tree(tree::JSON3.Object, ax)
    plotxgboosttree!(ax, tree)
    hidedecorations!(ax)
    hidespines!(ax)
end

end # module Boost
