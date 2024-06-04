module Anomaly
using JumpProcesses, Plots, Random, DifferentialEquations
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export driver

function driver(; base_in=100, m2_frac=0.1, r_scale=25, rstate = nothing)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		println(rstate)
	end
	copy!(Random.default_rng(), rstate)
	m2 = m2_frac*base_in
	sol = dynamics(base_in)
	pl = plot(layout=(2,1),size=(750,600))
	plot!(sol.t,[sol[1,:],sol[2,:]], subplot=1,lw=[1.5 2.25],color=[mma[1] mma[2]],
		label=["input value" "internal average"])
	plot!(sol.t,[sol[4,:],
		[receptor0(sol[2,i],sol[1,i],base_in,m2,2,1) .* r_scale for i in 1:length(sol[1,:])]],
		subplot=2,lw=[2.25 1.8],xlabel="Time", label=["anomaly jumps" "receptor response"],
		color=[mma[1] mma[2]])
	annotate!(pl,(0.05,0.72),text("(a)",10),subplot=1)
	annotate!(pl,(0.05,0.88),text("(b)",10),subplot=2)
	display(pl)
	return sol, pl
end

function dynamics(base_in; tmax=20)
	default(; lw=2)
	jump_size = 0.05  		# Amplitude of the jumps
	rate(u,p,t) = 0.2		# Rate of the Poisson process
	sigma = 0.02			# diffusion coeff for multiplicative noise
	
	function f!(du, u, p, t)
		du[1] = 0.0002(base_in - u[1])
		du[2] = 10.0(u[1] - u[2])
		du[3] = 2.0(1 - u[3])		# placeholder for a dynamics, not used
		du[4] = -60*u[4]
	end
	function g!(du, u, p, t) 
		du[1] = sigma*u[1]			# multiplicative noise
		du[2:4] .= zeros(3)
	end
	affect!(integrator) = (integrator.u[1] *= jmp=1+sign(randn())*jump_size;
		integrator.u[4] = (jmp>1) ? 1 : -1)

	u0 = [base_in,base_in,1.0,0.0]
	tspan = (0.0, tmax)
	
	prob = SDEProblem(f!, g!, u0, tspan, dt=1e-13)
	const_jump = ConstantRateJump(rate, affect!)
	jump_prob = JumpProblem(prob, Direct(), const_jump)
	sol = solve(jump_prob, SRIW1(),reltol=1e-8)
	#display(plot(sol, label=["u"], xlabel="Time", ylabel="Value"))
	return sol
end

hill(u,m,k) = u^k/(m^k+u^k)

receptor(u,m1,m2,k,a) = hill(u,m1,k) - a*hill(u,m2,k)

receptor0(us,u,m1,m2,k,a) = receptor(u,m1,m2,k,a) - receptor(us,m1,m2,k,a)

end # module Anomaly
