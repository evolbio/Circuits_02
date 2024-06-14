module MultiSensor
using Distributions, Optimization, OptimizationOptimisers, ReverseDiff, Enzyme,
	DataInterpolations, OptimizationOptimJL, DataStructures, OptimizationBBO
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export multisensor

# Receptor response for temporal anomaly detection
# Run optimization
function multisensor(n, mTypical, mAnomaly, v)
    initialWeights = rand(n) #fill(1.0, n)
    initialTarray = fill((mTypical + mAnomaly) / 2, n)
    initialTarray = (mTypical + mAnomaly) * rand(n)
    initialCutoff = sum(initialWeights) / 2.0
    # initial p values for optimization
    p = vcat(initialWeights, initialTarray, [initialCutoff])
    
    println(p)

    # Optimization using ADAM
    optf = OptimizationFunction((p,x) -> loss(p,n,mTypical,mAnomaly,v), AutoEnzyme())
    prob = OptimizationProblem(optf, p, lb=zeros(length(p)), ub=vcat(ones(n),
    	(mTypical + mAnomaly)*ones(n),n*ones(1)))
    result = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=20000)
    #prob = OptimizationProblem(optf, p)
	#result = solve(prob, NelderMead(), maxiters=5000)

    return result
end

# Objective function for optimization
function loss(p, n, mTypical, mAnomaly, v)
    wvals = p[1:n]
    tarray = p[n+1:2*n]
    cutoff = p[2*n+1]

    prob = pvals(tarray, mTypical, v)
    pmf = getPMF(wvals, prob)
    cmf = getCMF(pmf)
    cdf = getCDF(cmf)
    FPR = 1 - cdf(cutoff)

    prob = pvals(tarray, mAnomaly, v)
    pmf = getPMF(wvals, prob)
    cmf = getCMF(pmf)
    cdf = getCDF(cmf)
    FNR = cdf(cutoff)
    
    #println(FPR+FNR)

    return FPR + FNR
end

# Function to get PMF
function getPMF(w, p)
    pmf = Dict(0 => 1.0)  # Initial PMF, all probability at 0

    for i in 1:length(w)
        newPMF = Dict{Float64, Float64}()
        for (currentSum, prob) in pmf
            newPMF[currentSum] = get(newPMF, currentSum, 0.0) + (1 - p[i]) * prob
            newSum = currentSum + w[i]
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

# Response function
Response(t, m, v) = 1 - cdf(Normal(m, v), t)

# pvals function
pvals(tarray, m, v) = [Response(t, m, v) for t in tarray]

function sortDict(orig_dict)
	sorted_keys = sort(collect(keys(orig_dict)))
	sorted_dict = OrderedDict(key => orig_dict[key] for key in sorted_keys)
end

end # module MultiSensor
