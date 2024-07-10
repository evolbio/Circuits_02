module Boost
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
using Plots, DataFrames, MLBase, ROCAnalysis, XGBoost, MLDataUtils, Random, Statistics,
		Base.Threads, JSON3, CairoMakie, Graphs, GraphMakie, NetworkLayout,
		DataInterpolations, RegularizationTools, Measures
export oneR_analysis, xgb_analysis, print_all_trees, print_tree_stats, make_graphs,
		plot_f1, exp2range, plot_f1_trends

# samples in rows, features in cols
function xgb_analysis(X, y; rstate=nothing, trees=100, depth=6, show_info=true)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_info println(rstate) end
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
	kw = show_info ? Dict() : Dict(:watchlist => [])
	bst = xgboost(dtrain, num_round=trees, max_depth=depth, eta=0.3, objective="binary:logistic",
					eval_metric="logloss"; kw...)
	
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
	
	if show_info
		# Print results
		println("Accuracy: ", round(accuracy, digits=4))
		println("Recall: ", round(recall_value, digits=4))
		println("Precision: ", round(precision_value, digits=4))
		println("F1 Score: ", round(f1_score, digits=4))
		
		# Display feature importance
		feature_importance = importance(bst)
		display(feature_importance)
	end
	return bst, dtest, (accuracy, recall_value, precision_value, f1_score)
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

# Function to calculate the depth of a tree
function get_tree_depth(tree)
    if haskey(tree, "children")
        return 1 + maximum(get_tree_depth.(tree["children"]))
    else
        return 0
    end
end

# Function to count the number of nodes in a tree
function count_nodes(tree)
    if haskey(tree, "children")
        return 1 + sum(count_nodes.(tree["children"]))
    else
        return 1
    end
end

# Function to count the number of leaves in a tree
function count_leaves(tree)
    if haskey(tree, "children")
        return sum(count_leaves.(tree["children"]))
    else
        return 1
    end
end

# Function to extract basic stats from a tree
function get_tree_stats(tree)
    num_nodes = count_nodes(tree)
    num_leaves = count_leaves(tree)
    depth = get_tree_depth(tree)
    return num_nodes, num_leaves, depth
end

# Iterate over each tree in the model dump
function print_tree_stats(model_result)
	model_dump = XGBoost.dump(model_result, fmap="", with_stats=true)
	for (i, tree) in enumerate(model_dump)
		num_nodes, num_leaves, depth = get_tree_stats(tree)
		println("Tree $i:")
		println("  Depth: $depth")
		println("  Number of nodes: $num_nodes")
		println("  Number of leaves: $num_leaves")
		println()
	end
end

exp2range(r::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}) = exp2.(r)
exp2range(r::UnitRange{Int64}) = Int.(exp2.(r))

# plot F1 for range of num_trees and tree_depth
# generate data once, then use increasing number of feature columns, and rescale mean deviations
# using one matrix fixes typical data pattern for all analyses, otherwise newly created data matrix
# would have differing typical pattern and would make comparisons difficult
function plot_f1(; trees=2:20, depth=2:6, show_legend=false, data_size=1e5,
		features=exp2range(2:5), mean_scale=0.05*exp2range(1:5), smooth=false, rstate=nothing)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_info println(rstate) end
	end
	copy!(Random.default_rng(), rstate)
	
	# use mean_scale[1] for base data matrix, rescale below for each increase in mean_scale
	X,y,nm,am,nc=generate_data(Int(data_size),features[end],0.1; mean_scale=mean_scale[1], show_rstate=false)
	
	df = DataFrame(trees=Int[], depth=Int[], features=Int[], scale=Float64[], F1=Float64[])

	top = Int(floor(sqrt(maximum(trees))))
	xt = exp2range(1:top)
	pl_size=(length(features)*325,length(mean_scale)*260)
	pl = Plots.plot(xscale=:log2, xticks=(xt,string.(xt)), legend=show_legend ? :topleft : :none,
			legendtitle="depth", xlabel="Number of trees", ylabel="F1 score", ylim=(0,:auto),
			layout=(length(mean_scale),length(features)), size=pl_size)
	s = 1
	for m in mean_scale
		for f in features
			d = m/mean_scale[1]		# multiplier for scaling relative to smallest scale value
			Xf = adjust_mean_scale(X,y,d,f,nm,am)
			println("features = ", f, "; mean_scale = ", m)
			default(;lw=2)
			top = Int(floor(sqrt(maximum(trees))))
			xt = exp2range(1:top)
			for i in 1:length(depth)
				f1=zeros(length(trees));
				for j in 1:length(trees)
					bst, dtest, stats = xgb_analysis(Xf,y; trees=trees[j], depth=depth[i], show_info=false);
					f1[j] = stats[4] === NaN ? -1e-5 : stats[4]
					push!(df, Dict(:trees => trees[j], :depth=>depth[i], :features=>f, :scale=>m, :F1=>f1[j]))
				end
				if smooth && f1[1] > 0
					Plots.plot!(RegularizationSmooth(f1,Float64.(trees),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				elseif smooth
					tr = trees[f1 .>= 0]
					f1 = f1[f1 .>= 0]
					Plots.plot!(RegularizationSmooth(f1,Float64.(tr),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				else
					Plots.plot!(trees,f1,label=depth[i],subplot=s)
				end
				display(pl)
			end
			s += 1
		end
	end
	display(pl)
	return pl, df
end

# plot F1 for range of num_trees and tree_depth
function plot_f1_varying_data(; trees=2:20, depth=2:6, show_legend=false, data_size=1e5,
		features=exp2range(2:5), mean_scale=0.05*exp2range(1:5), smooth=false, rstate=nothing)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_info println(rstate) end
	end
	copy!(Random.default_rng(), rstate)
	
	df = DataFrame(trees=Int[], depth=Int[], features=Int[], scale=Float64[], F1=Float64[])

	top = Int(floor(sqrt(maximum(trees))))
	xt = exp2range(1:top)
	pl_size=(length(features)*325,length(mean_scale)*260)
	pl = Plots.plot(xscale=:log2, xticks=(xt,string.(xt)), legend=show_legend ? :topleft : :none,
			legendtitle="depth", xlabel="Number of trees", ylabel="F1 score",
			layout=(length(mean_scale),length(features)), size=pl_size)
	s = 1
	for m in mean_scale
		for f in features
			X,y,nm,am,nc=generate_data(Int(data_size),f,0.1; mean_scale=m, show_rstate=false)
			println("features = ", f, "; mean_scale = ", m)
			default(;lw=2)
			top = Int(floor(sqrt(maximum(trees))))
			xt = exp2range(1:top)
			for i in 1:length(depth)
				f1=zeros(length(trees));
				for j in 1:length(trees)
					bst, dtest, stats = xgb_analysis(X,y; trees=trees[j], depth=depth[i], show_info=false);
					f1[j]=stats[4]
					push!(df, Dict(:trees => trees[j], :depth=>depth[i], :features=>f, :scale=>m, :F1=>f1[j]))
				end
				if smooth && f1[1] > 0
					Plots.plot!(RegularizationSmooth(f1,Float64.(trees),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				elseif smooth
					tr = trees[f1 .>= 0]
					f1 = f1[f1 .>= 0]
					Plots.plot!(RegularizationSmooth(f1,Float64.(tr),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				else
					Plots.plot!(trees,f1,label=depth[i],subplot=s)
				end
				display(pl)
			end
			s += 1
		end
	end
	display(pl)
	return pl, df
end

exp2_1(x) = 2^x - 1

# take a dataframe for F1 and make a plot
function plot_f1(df::DataFrame; smooth=true)
	features = unique(df.features)
	mean_scale = unique(df.scale)
	trees = unique(df.trees)
	depth = unique(df.depth)
	top = Int(floor(sqrt(maximum(trees))))
	xt = exp2range(1:top)
	yt = exp2_1.(0.0:0.2:1.0)
	pl_size=(length(features)*325,length(mean_scale)*260)
	pl = Plots.plot(xscale=:log2, xticks=(xt,string.(xt)), yticks=(yt,string.(0.0:0.2:1.0)),
			legend=:none, ylimits=(0,1),
			legendtitle="depth", xlabel="Number of trees", ylabel="F1 score",
			layout=(length(mean_scale),length(features)), size=pl_size)
	s = 1
	for m in mean_scale
		for f in features
			println("features = ", f, "; mean_scale = ", m)
			default(;lw=2)
			for i in 1:length(depth)
				f1=zeros(length(trees));
				for j in 1:length(trees)
					curr_val = (trees=trees[j], depth = depth[i], features = f, scale = m)
					f1[j]=filter(row -> all(row[col] == val for (col, val) in pairs(curr_val)), df)[1,:F1]
				end
				f1 .= exp2_1.(f1)
				if smooth && f1[1] > 0
					Plots.plot!(RegularizationSmooth(f1,Float64.(trees),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				elseif smooth
					tr = trees[f1 .> 0]
					f1 = f1[f1 .> 0]
					Plots.plot!(RegularizationSmooth(f1,Float64.(tr),4;λ=1e0,alg = :fixed),label=depth[i],subplot=s)
				else
					Plots.plot!(trees,f1,label=depth[i],subplot=s)
				end
				display(pl)
			end
			s += 1
		end
	end
	display(pl)
	return pl
end

function plot_f1_trends(df::DataFrame; smooth=false)
	tree_vec = [4, 8, 16]
	depth_vec = [2, 4, 6]
	features = unique(df.features)
	mean_scale = unique(df.scale)
	xt = (mean_scale, string.(mean_scale))
	yt = 0.0:0.2:1.0
	pl_size=(length(depth_vec)*325*1.0,length(tree_vec)*260*1.0)
	pl = Plots.plot(yticks=(yt,string.(0.0:0.2:1.0)), xticks=xt, xscale=:log2,
			legend=:none, ylimits=(-0.02,1.02), xlabel="", ylabel="", grid=:none,
			size=pl_size, layout=(length(tree_vec),length(depth_vec)))
	default(;lw=2.5)
	s = 1
	for tr in tree_vec
		for dp in depth_vec
			curr_val = (trees=tr, depth = dp)
			df2=filter(row -> all(row[col] == val for (col, val) in pairs(curr_val)), df)
			select!(df2, [:features, :scale, :F1])
			c = 1
			for f in features
				f1=zeros(length(mean_scale))
				for i in 1:length(mean_scale)
					curr_val = (features = f, scale = mean_scale[i])
					f1[i] = filter(row -> all(row[col] == val for (col, val) in pairs(curr_val)), df2)[1,:F1]
				end
				f1 .= max.(f1,0)
				ms = mean_scale[f1 .>= 0]
				f1 = f1[f1 .>= 0]
				if smooth && length(f1)>3
					#Plots.plot!(RegularizationSmooth(f1,Float64.(ms),3;λ=1e-1,alg = :fixed))
					Plots.plot!(AkimaInterpolation(f1,Float64.(ms)), subplot=s, color=mma[c])
				else
					Plots.plot!(ms,f1,subplot=s, color=mma[c])
				end
				display(pl)
				c += 1
			end
			s += 1
		end
	end
	annotate!(pl,(0.5,-0.23),Plots.text("Mean scale",18),subplot=8, bottom_margin=1cm)
	annotate!(pl,(-0.23,0.5),Plots.text("F1 score",18,rotation=90),subplot=4, left_margin=1cm)
	a = collect('a':'z')
	s = 1
	for i in 1:length(tree_vec)
		for j in 1:length(depth_vec)
			annotate!(pl,(0.07,0.98),Plots.text("("*a[s]*")",11),subplot=s)
			annotate!(pl,(0.38,0.98),Plots.text(
				"("*string(tree_vec[i])*","*string(depth_vec[j])*")",11,:center),subplot=s)
			s+=1
		end
	end
	annotate!(pl,(0.38,0.88),Plots.text("(trees,depth)",11,:center),subplot=1)

	
	# make legend for line types
	s=4
	y_base=0.35
	x = fill([0.13,0.21],length(features))
	y=[[k,k] for k in y_base:0.1:(y_base+0.1*(length(features)-1))]
	for i in 1:length(x)
		Plots.plot!(x[i],y[i],subplot=s,color=mma[i])
		annotate!(pl,(0.38,y[i][1]),Plots.text(features[i],9,:right),subplot=s)
	end
	annotate!(pl,(0.37,0.02+y_base+0.1*length(features)),Plots.text("features",11,:right),subplot=s)
	lf = 0.11
	rt = 0.33
	bt = y_base - 0.07
	tp = y_base + 0.50
	Plots.plot!(pl,[lf,rt,rt,lf,lf],[bt,bt,tp,tp,bt],color=:black,lw=1,subplot=s)
	
# 	# label rows and cols of plots
# 	annotate!(pl,(0.02,1.08),Plots.text("tree depth =",12,:left),subplot=1,top_margin=15pt)
# 	for i in 1:length(depth_vec)
# 		annotate!(pl,(0.5,1.08),Plots.text(string(depth_vec[i]),12,:center),subplot=i)
# 	end
# 	annotate!(pl,(1.06,0.90),Plots.text("trees =",12,:left,rotation=270),subplot=3,right_margin=13pt)
# 	for i in 1:length(tree_vec)
# 		annotate!(pl,(1.06,0.5),Plots.text(string(tree_vec[i]),12,:center,rotation=270),subplot=i*3)
# 	end

	display(pl)
	return pl
end

# plot F1 for range of num_trees and tree_depth
function plot_f1_single(; trees=2:20, depth=2:6, show_legend=false, data_size=1e5, features=32, mean_scale=0.5)
	X,y,nm,am,nc=generate_data(Int(data_size),features,0.1; mean_scale=mean_scale)
	default(;lw=2)
	top = Int(floor(sqrt(maximum(trees))))
	xt = exp2range(1:top)
	pl = Plots.plot(xscale=:log2, xticks=(xt,string.(xt)), legend=show_legend ? :topleft : :none,
			legendtitle="depth", xlabel="Number of trees", ylabel="F1 score")
	for i in 1:length(depth)
		f1=zeros(length(trees));
		for j in 1:length(trees)
			bst, dtest, stats = xgb_analysis(X,y; trees=trees[j], depth=depth[i], show_info=false);
			f1[j]=stats[4]
		end
		Plots.plot!(trees,f1,label=depth[i])
	end
	display(pl)
	return pl
end

end # module Boost
