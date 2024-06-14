module MultiSensor
using Distributions, Optimization, OptimizationOptimisers, ReverseDiff, Enzyme,
	DataInterpolations, OptimizationOptimJL, DataStructures, OptimizationBBO
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export multisensor

# Receptor response for temporal anomaly detection
# Run optimization
function multisensor(n, mTypical, mAnomaly, v)
    initialWeightsDown = rand(n) #fill(1.0, n)
    initialWeightsUp = rand(n) #fill(1.0, n)
    initialTarray = (mTypical + mAnomaly) * rand(n)
    initialCutoff = 0.0
    # initial p values for optimization
    p = vcat(initialWeightsDown, initialWeightsUp, initialTarray, [initialCutoff])
    
    println(p)

    # Optimization using ADAM
    lb = vcat(zeros(3n), -n*ones(1))
    ub = vcat(ones(2n), (mTypical + mAnomaly)*ones(n),n*ones(1))
    optf = OptimizationFunction((p,x) -> loss(p,n,mTypical,mAnomaly,v), AutoEnzyme())
    prob = OptimizationProblem(optf, p, lb=lb, ub=ub)
    result = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=50000)
    #prob = OptimizationProblem(optf, p)
	#result = solve(prob, NelderMead(), maxiters=5000)

    return result
end

# Objective function for optimization
function loss(p, n, mTypical, mAnomaly, v)
    wdown = p[1:n]		# typical
    wup = p[n+1:2n]		# anomaly
    tarray = p[2n+1:3*n]
    cutoff = p[3*n+1]

    prob = pvals(tarray, mTypical, v)
    pmf = getPMF(wdown, wup, prob)
    cmf = getCMF(pmf)
    cdf = getCDF(cmf)
    FPR = 1 - cdf(cutoff)

    prob = pvals(tarray, mAnomaly, v)
    pmf = getPMF(wdown, wup, prob)
    cmf = getCMF(pmf)
    cdf = getCDF(cmf)
    FNR = cdf(cutoff)
    
    #println(FPR+FNR)

    #return FPR + FNR
    return -F1(FNR, FPR; P_actual=0.1, N_actual=0.9)
end

# Function to get PMF, convolution
# For each sensor, update pmf by splitting current prob mass into two new
# components, one component for if sensor reports typical input and one
# for report of anomaly, with weighting down for typical and up for anomaly
function getPMF(wdown, wup, p)
    pmf = Dict(0 => 1.0)  # Initial PMF, all probability at 0

    for i in 1:length(wdown)
        newPMF = Dict{Float64, Float64}()
        for (currentSum, prob) in pmf
        	newSum = currentSum - wdown[i]
            newPMF[newSum] = get(newPMF, newSum, 0.0) + (1 - p[i]) * prob
            newSum = currentSum + wup[i]
            newPMF[newSum] = get(newPMF, newSum, 0.0) + p[i] * prob
        end
        pmf = newPMF
    end

    return pmf
end

# Function to get CMF
function getCMF(pmf)
    sortedKeys = sort(collect(keys(pmf)))
    cmf = Dict(k => sum(pmf[i] for i in sortedKeys if i <= k) for k in sortedKeys)
    return cmf
end

# Function to get CDF
function getCDF(cmf)
    sortedKeys = sort(collect(keys(cmf)))
    sortedValues = [cmf[key] for key in sortedKeys]
    interp = LinearInterpolation(sortedValues,sortedKeys)
    function cdf(x)
        if x < first(sortedKeys)
            return 0.0
        elseif x >= last(sortedKeys)
            return 1.0
        else
            return interp(x)
        end
    end
    return cdf
end

# Calculate F1, which is model accuracy
function F1(FNR, FPR; P_actual=0.5, N_actual=0.5)
    # Calculate TP, FN, FP
    TP = (1 - FNR) * P_actual
    FN = FNR * P_actual
    FP = FPR * N_actual
    
    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    # Calculate F1 score
    F1 = 2 * (precision * recall) / (precision + recall)
    return (F1 === NaN) ? 0.0 : F1
end

# Response function
Response(t, m, v) = 1 - cdf(Normal(m, v), t)
#Response(t, m, v) = 1 - cdf(Frechet(m, v), t)

# pvals function
pvals(tarray, m, v) = [Response(t, m, v) for t in tarray]

function sortDict(orig_dict)
	sorted_keys = sort(collect(keys(orig_dict)))
	sorted_dict = OrderedDict(key => orig_dict[key] for key in sorted_keys)
end

end # module MultiSensor
