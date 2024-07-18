module Encoder

using Anomaly, Plots, Lux, Random, Optimisers, Zygote, Statistics, LinearAlgebra, Printf,
		DataFrames, ProgressMeter, Base.Threads
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export encoder, encoder_loop, plot_encoder, iterative_feature_search, feature_loop, plot_features

function encoder_loop(;n=2:5, mean_scale=0.05*exp2range(1:5), twoD=false, rstate=nothing,
						data_size=1e5, show_rstate=true, num_epoch)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_rstate
			println("rstate for encoder:")
			println(rstate)
		end
	end
	copy!(Random.default_rng(), rstate)

	df = DataFrame(features=Int[], scale=Float64[], F1=Float64[])
	X,y,nm,am,nc=generate_data(Int(data_size),2^n[end],0.1; mean_scale=mean_scale[1], 
							rstate=rstate, show_rstate=false)
	for m in mean_scale
		for nn in n
			f = 2^nn
			d = m/mean_scale[1]		# multiplier for scaling relative to smallest scale value
			Xf = adjust_mean_scale(X,y,d,f,nm,am)
			@printf("features = %2d, mean_scale = %3.1f\n", f, m)
			f1, _ = encoder(Xf,y; n=nn, mean_scale=m, rstate=rstate, num_epoch=num_epoch)
			push!(df, Dict(:features=>f, :scale=>m, :F1=>f1))
		end
	end
	return df
end

# if multiple obs for a feature and mean_scale combination, then plot the average F1 for those vals
function plot_encoder(df::DataFrame)
	features = unique(df.features)
	mean_scale = unique(df.scale)
	yt = collect(0.0:0.2:1.0)
	pl_size=(325,260)
	pl = Plots.plot(xscale=:log2, xticks=(mean_scale,string.(mean_scale)), yticks=(yt,string.(0.0:0.2:1.0)),
			legend=:none, ylimits=(0,1),
			xlabel="Mean scale", ylabel="F1 score", size=pl_size)
	default(;lw=2)
	for j in 1:length(features)
		f1=zeros(length(mean_scale));
		for i in 1:length(mean_scale)
			curr_val = (scale=mean_scale[i], features = features[j])
			f1[i]=mean(filter(row -> all(row[col] == val for (col, val) in pairs(curr_val)), df)[:,:F1])
		end
		plot!(mean_scale,f1,color=mma[j])
	end
	annotate!(pl,(0.29,0.18),text("4",9,:center))
	#annotate!(pl,(0.25,0.325),text("8",9,:center))
	annotate!(pl,(0.24,0.42),text("8",9,:center))
	annotate!(pl,(0.18,0.715),text("16",9,:center))
	annotate!(pl,(0.15,0.94),text("32",9,:center))

	display(pl)
	return pl
end

function encoder(X=nothing, y=nothing; n=4, twoD=false, mean_scale=0.8, rstate=nothing,
					show_rstate=true, show_results=false, num_epoch=3000)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_rstate
			println("rstate for encoder:")
			println(rstate)
		end
	end
	copy!(Random.default_rng(), rstate)

	if X === nothing
		input_dim = 2^n
		X, y, nm, am, nc = Anomaly.generate_data(100000, input_dim, 0.1; mean_scale=mean_scale, rstate=rstate)
	else
		input_dim = size(X, 2)			# before taking transpose, cols are features
		n = ceil(Int, log2(input_dim))  # Adjust n based on actual input dimension
	end
	
	@assert !twoD || input_dim > 1	"Cannot have twoD with 1D input"
	m = twoD ? n-1 : n
	output_dim = 2^(n-m)
	
	X = X'  # Transpose X to match the expected input shape
	X_train, y_train, X_test, y_test = split_data(X, y)

	rng = Random.default_rng()
	model = create_encoder(input_dim, output_dim)
	ps, st = Lux.setup(rng, model)
	
	# Initialize logistic regression parameters
	w = randn(rng, output_dim)
	b = randn(rng)
		
	# Training loop
	optimizer = Optimisers.Adam(0.001)
	opt_state = Optimisers.setup(optimizer, ps)
	
	for epoch in 1:num_epoch
		grad_ps, grad_w, grad_b =
			Zygote.gradient((ps, w, b) -> new_loss(ps, w, b, st, model, X_train, y_train), ps, w, b)
		opt_state, ps = Optimisers.update(opt_state, ps, grad_ps)
		w -= 0.001 * grad_w
		b -= 0.001 * grad_b
		
		if epoch % 100 == 0 && show_results
			train_loss = new_loss(ps, w, b, st, model, X_train, y_train)
			test_loss = new_loss(ps, w, b, st, model, X_test, y_test)
			@printf("Epoch %4s, Train Loss: %5.3e, Test Loss: %5.3e\n", epoch, train_loss, test_loss)
		end
	end
	
	encoded_test, _ = model(X_test, ps, st)
	if output_dim == 1
		accuracy, precision, recall, f1, threshold, confusion_matrix, direction = evaluate_1d(encoded_test, y_test)
		p = visualize_encoded_data(encoded_test, y_test, nothing, threshold, nothing, direction)
	else
		accuracy, precision, recall, f1, threshold, confusion_matrix, separation_direction,
			typical_mean, direction = evaluate_2d(encoded_test, y_test)
		p = visualize_encoded_data(encoded_test, y_test, separation_direction, threshold, typical_mean, direction)
	end
	
	if show_results
		# Evaluate and visualize
		display(p)
		
		println("\nTest Set Evaluation:")
		println("Confusion Matrix:")
		println(confusion_matrix)
		println("Accuracy: ", accuracy)
		println("Precision: ", precision)
		println("Recall: ", recall)
		println("F1 Score: ", f1)
		println("Best Threshold: ", threshold)
		println("Direction: ", direction)
	end
	return f1, p
end

function create_encoder(input_dim::Int, output_dim::Int)
    layers = []
    
    if input_dim == 1
        if output_dim == 1
            push!(layers, Dense(1 => 1, tanh))
        else
            throw(ArgumentError("Cannot increase dimensionality from 1 to $output_dim"))
        end
    elseif input_dim == output_dim
        # If input_dim equals output_dim, use a single layer
        push!(layers, Dense(input_dim => output_dim, tanh))
    else
        n = floor(Int, log2(input_dim))
        
        if 2^n == input_dim && input_dim > output_dim
            # If input_dim is a power of 2 and greater than output_dim, start reducing
            push!(layers, Dense(input_dim => input_dim ÷ 2, tanh))
            n -= 1
        elseif 2^n < input_dim
            # If input_dim is not a power of 2, reduce to the nearest lower power of 2
            push!(layers, Dense(input_dim => 2^n, tanh))
        end
        
        # Subsequent layers: reduce by factor of 2 each time
        while 2^n > output_dim
            push!(layers, Dense(2^n => 2^(n-1), tanh))
            n -= 1
        end
        
        # Output layer
        if 2^n != output_dim
            push!(layers, Dense(2^n => output_dim))
        end
    end
    
    return Chain(layers...)
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function focal_loss(y_true, y_pred; γ=2.0, α=0.25)
    y_pred = clamp.(y_pred, 1e-7, 1 - 1e-7)
    bce = -y_true .* log.(y_pred) - (1 .- y_true) .* log.(1 .- y_pred)
    fl = α * (1 .- y_pred).^γ .* y_true .* log.(y_pred) +
         (1 - α) * y_pred.^γ .* (1 .- y_true) .* log.(1 .- y_pred)
    return mean(-fl)
end

function new_loss(ps, w, b, st, model, X, y)
    encoded, _ = model(X, ps, st)
    y_pred = sigmoid.(w' * encoded .+ b)
    return focal_loss(y, vec(y_pred))
end

function evaluate_1d(encoded, y)
    encoded_vec = vec(encoded)
    sorted_indices = sortperm(encoded_vec)
    sorted_encoded = encoded_vec[sorted_indices]
    sorted_y = y[sorted_indices]
    
    # Determine the direction of separation
    typical_mean = mean(encoded_vec[y .== 0])
    anomalous_mean = mean(encoded_vec[y .== 1])
    direction = sign(anomalous_mean - typical_mean)
    
    best_threshold = 0
    best_f1 = 0
    best_confusion_matrix = zeros(Int, 2, 2)
    
    for i in 1:length(sorted_encoded)-1
        if sorted_y[i] != sorted_y[i+1]
            threshold = (sorted_encoded[i] + sorted_encoded[i+1]) / 2
            if direction > 0
                y_pred = encoded_vec .>= threshold
            else
                y_pred = encoded_vec .<= threshold
            end
            tp = sum((y .== 1) .& (y_pred .== 1))
            tn = sum((y .== 0) .& (y_pred .== 0))
            fp = sum((y .== 0) .& (y_pred .== 1))
            fn = sum((y .== 1) .& (y_pred .== 0))
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            if f1 > best_f1
                best_f1 = f1
                best_threshold = threshold
                best_confusion_matrix = [tn fp; fn tp]
            end
        end
    end
    
    accuracy = sum(diag(best_confusion_matrix)) / sum(best_confusion_matrix)
    precision = best_confusion_matrix[2,2] / sum(best_confusion_matrix[:, 2])
    recall = best_confusion_matrix[2,2] / sum(best_confusion_matrix[2, :])
    
    return accuracy, precision, recall, best_f1, best_threshold, best_confusion_matrix, direction
end

function evaluate_2d(encoded, y)
    typical_mean = mean(encoded[:, y .== 0], dims=2)
    anomalous_mean = mean(encoded[:, y .== 1], dims=2)
    
    separation_direction = anomalous_mean - typical_mean
    separation_direction /= norm(separation_direction)
    
    projected = (encoded .- typical_mean)' * separation_direction
    
    accuracy, precision, recall, f1, threshold, confusion_matrix, direction = evaluate_1d(projected, y)
    
    return accuracy, precision, recall, f1, threshold, confusion_matrix, separation_direction, typical_mean, direction
end

function visualize_encoded_data(encoded, y, separation_direction=nothing, threshold=nothing, typical_mean=nothing, direction=1)
    if size(encoded, 1) == 1
        encoded_vec = vec(encoded)
        p = histogram(encoded_vec[y .== 0], bins=100, alpha=0.5, label="Typical", color=mma[1], normalize=:probability)
        histogram!(p, encoded_vec[y .== 1], bins=100, alpha=0.5, label="Anomalous", color=mma[2], normalize=:probability)
        if !isnothing(threshold)
            vline!([threshold], label="Decision Boundary", color=mma[3], linewidth=2)
        end
        xlabel!(p, "Encoded Value")
        ylabel!(p, "Probability")
        if direction > 0
            annotate!([(minimum(encoded_vec), 0.9, ("Typical", 10, :left, :blue)),
                       (maximum(encoded_vec), 0.9, ("Anomalous", 10, :right, :red))])
        else
            annotate!([(maximum(encoded_vec), 0.9, ("Typical", 10, :right, :blue)),
                       (minimum(encoded_vec), 0.9, ("Anomalous", 10, :left, :red))])
        end
    else
        p = scatter(encoded[1, y .== 0], encoded[2, y .== 0], 
                    label="Typical", color=mma[1], alpha=0.99, 
                    markersize=3, markerstrokewidth=0)
        scatter!(p, encoded[1, y .== 1], encoded[2, y .== 1], 
                 label="Anomalous", color=mma[2], alpha=0.7, 
                 markersize=3, markerstrokewidth=0)
        if !isnothing(separation_direction) && !isnothing(threshold) && !isnothing(typical_mean)
            perpendicular_direction = [-separation_direction[2], separation_direction[1]]
            boundary_point = typical_mean + threshold * separation_direction
            
            x_min, x_max = minimum(encoded[1, :]), maximum(encoded[1, :])
            y_min, y_max = minimum(encoded[2, :]), maximum(encoded[2, :])
            
            margin = 0.1
            x_range = (x_max - x_min) * (1 + margin)
            y_range = (y_max - y_min) * (1 + margin)
            
            t_min = -max(x_range, y_range)
            t_max = max(x_range, y_range)
            t_range = range(t_min, t_max, length=100)
            
            boundary_points = [boundary_point .+ t * perpendicular_direction for t in t_range]
            boundary_x = [p[1] for p in boundary_points]
            boundary_y = [p[2] for p in boundary_points]
            
            plot!(p, boundary_x, boundary_y, label="Decision Boundary", color=mma[3], linewidth=2)
        end
        xlabel!(p, "Encoded Dimension 1")
        ylabel!(p, "Encoded Dimension 2")
    end
    return p
end

# Split data into training and test sets
function split_data(X, y, train_ratio=0.7)
    n = size(X, 2)
    indices = randperm(n)
    train_size = round(Int, train_ratio * n)
    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]
    return X[:, train_indices], y[train_indices], X[:, test_indices], y[test_indices]
end

### Code for iterative feature search, find best subset of features to reduce dimensionality

function feature_loop(orig_features, max_features; mean_scale=0.05*exp2range(1:5), rstate=nothing,
						show_rstate=true, data_size=1e5, num_epoch=3000)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_rstate
			println("rstate for encoder:")
			println(rstate)
		end
	end
	copy!(Random.default_rng(), rstate)

	df = nothing
	X,y,nm,am,nc=generate_data(Int(data_size),orig_features,0.1; mean_scale=mean_scale[1], 
							rstate=rstate, show_rstate=false)
	for m in mean_scale
		d = m/mean_scale[1]		# multiplier for scaling relative to smallest scale value
		Xf = adjust_mean_scale(X,y,d,orig_features,nm,am)
		@printf("mean_scale = %3.1f\n", m)
		new_df = iterative_feature_search(Xf, y, m; max_features=max_features, rstate=rstate,
			num_epoch=num_epoch, show_progress=false)
		df === nothing ? df = new_df : df = vcat(df, new_df)
	end
	return df
end

function iterative_feature_search(X, y, mean_scale; max_features=size(X, 2)÷2, rstate=nothing,
			num_epoch=3000, show_progress=true)
    best_features = Int[]
    results = DataFrame(mean_scale=Float64[], features=Int[], selected_features=Vector{Int}[], F1=Float64[])
    if show_progress p = Progress(max_features, 1, "Searching features: ") end # progress bar

    for f in 1:max_features
        local_best_f1 = Atomic{Float64}(0.0)
        local_best_feature = Atomic{Int}(0)
        
        @printf("Working on feature number %2d\n", f)

        Threads.@threads for i in 1:size(X, 2)
            if i ∉ best_features
                current_features = sort([best_features; i])
                X_subset = X[:, current_features]
                
                f1, _ = encoder(X_subset, y; rstate=rstate, show_rstate=false, num_epoch=num_epoch)

                # Atomic operations to update best score and feature
                while true
                    old_f1 = local_best_f1[]
                    if f1 <= old_f1
                        break
                    end
                    if atomic_cas!(local_best_f1, old_f1, f1) === old_f1
                        atomic_xchg!(local_best_feature, i)
                        break
                    end
                end
            end
        end

        new_best_feature = local_best_feature[]
        new_best_f1 = local_best_f1[]

        # Always add the best feature for this iteration
        push!(best_features, new_best_feature)

        push!(results, (mean_scale, f, copy(best_features), new_best_f1))
        if show_progress next!(p) end # update progress bar
    end

    return results
end

function plot_features(df; n_features=[2, 4, 8])
	features = unique(df.features)
	mean_scale = unique(df.mean_scale)
	yt = collect(0.0:0.2:1.0)
	pl_size=(325,260)
	pl = Plots.plot(xscale=:log2, xticks=(mean_scale,string.(mean_scale)), yticks=(yt,string.(0.0:0.2:1.0)),
			legend=:none, ylimits=(0,1),
			xlabel="Mean scale", ylabel="F1 score", size=pl_size)
	default(;lw=2)
	i = 1
	for f in n_features
		df_tmp = df[df.features .== f, :]
		plot!(df_tmp.mean_scale, df_tmp.F1, color=mma[i])
		i += 1
	end
	display(pl)
	return pl
end

end # module Encoder
