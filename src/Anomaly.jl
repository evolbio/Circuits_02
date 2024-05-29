module Anomaly
using JumpProcesses, Plots, Random, DifferentialEquations
export jtest

function jtest()
	default(; lw=2)
	jump_amplitude = 0.2  	# Amplitude of the jumps
	rate(u,p,t) = 0.3		# Rate of the Poisson process
	
	f!(du, u, p, t) = du[1] = 0.0002(100 - u[1])
	g!(du, u, p, t) = du[1] = 0.1*u[1]
	affect!(integrator) = (integrator.u[1] += jmp = 2randn(); integrator.u[2] += jmp)

	u0 = [100.0,0]
	tspan = (0.0, 20.0)
	
	prob = SDEProblem(f!, g!, u0, tspan, dt=1e-8)
	const_jump = ConstantRateJump(rate, affect!)
	jump_prob = JumpProblem(prob, Direct(), const_jump)
	sol = solve(jump_prob, SRIW1())
	display(plot(sol, label=["u"], xlabel="Time", ylabel="Value"))	
end

end # module Anomaly
