module Encoder

using Anomaly, Plots, Lux, Random, Optimisers, Zygote, Statistics, LinearAlgebra, Printf,
		DataFrames
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export encoder, encoder_loop, plot_encoder

function encoder_loop(;n=2:5, mean_scale=0.05*exp2range(1:5), twoD=false, rstate=nothing,
						data_size=1e5, show_rstate=true)
	df = DataFrame(features=Int[], scale=Float64[], F1=Float64[])
	X,y,nm,am,nc=generate_data(Int(data_size),2^n[end],0.1; mean_scale=mean_scale[1], show_rstate=false)
	for m in mean_scale
		for nn in n
			f = 2^nn
			d = m/mean_scale[1]		# multiplier for scaling relative to smallest scale value
			Xf = adjust_mean_scale(X,y,d,f,nm,am)
			@printf("features = %2d, mean_scale = %3.1f\n", f, m)
			f1, _ = encoder(Xf,y; n=nn, mean_scale=m, rstate=rstate)
			push!(df, Dict(:features=>f, :scale=>m, :F1=>f1))
		end
	end
	return df
end

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
			f1[i]=filter(row -> all(row[col] == val for (col, val) in pairs(curr_val)), df)[1,:F1]
		end
		plot!(mean_scale,f1,color=mma[j])
	end
	display(pl)
	return pl
end

function encoder(X=nothing, y=nothing; n=4, twoD=false, mean_scale=0.8, rstate=nothing,
					show_rstate=true, show_results=false, num_epoch=3000)
	@printf("Size X = %d, 2^n = %d", size(X,1), 2^n)
	#@assert X === nothing || size(X,1) == 2^n	"# features must be 2^n"
	if rstate === nothing
		rstate = copy(Random.default_rng())
		if show_rstate
			println("rstate for generate_data:")
			println(rstate)
		end
	end
	copy!(Random.default_rng(), rstate)

	m = twoD ? n-1 : n
	input_dim = 2^n
	output_dim = 2^(n-m)
	hidden_layers = [2^(n-i) for i in 1:m-1]
	
	if X === nothing
		X, y, nm, am, nc = Anomaly.generate_data(100000, 2^n, 0.1; mean_scale=mean_scale, rstate=rstate)
	end
	X = X'  # Transpose X to match the expected input shape
	X_train, y_train, X_test, y_test = split_data(X, y)

	rng = Random.default_rng()
	model = create_encoder(input_dim, output_dim, hidden_layers)
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

function create_encoder(input_dim::Int, output_dim::Int, hidden_layers::Vector{Int})
    layers = []
    push!(layers, Dense(input_dim => hidden_layers[1], tanh))
    for i in 1:length(hidden_layers)-1
        push!(layers, Dense(hidden_layers[i] => hidden_layers[i+1], tanh))
    end
    push!(layers, Dense(hidden_layers[end] => output_dim))
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
        p = histogram(encoded_vec[y .== 0], bins=100, alpha=0.5, label="Typical", color=:blue, normalize=:probability)
        histogram!(p, encoded_vec[y .== 1], bins=100, alpha=0.5, label="Anomalous", color=:red, normalize=:probability)
        if !isnothing(threshold)
            vline!([threshold], label="Decision Boundary", color=:green, linewidth=2)
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
                    label="Typical", color=:blue, alpha=0.6, 
                    markersize=3, markerstrokewidth=0)
        scatter!(p, encoded[1, y .== 1], encoded[2, y .== 1], 
                 label="Anomalous", color=:red, alpha=0.6, 
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
            
            plot!(p, boundary_x, boundary_y, label="Decision Boundary", color=:green, linewidth=2)
        end
        xlabel!(p, "Encoded Dimension 1")
        ylabel!(p, "Encoded Dimension 2")
    end
    title!(p, "Encoded Data Visualization")
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

end # module Encoder
