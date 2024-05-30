module Anomaly
using JumpProcesses, Plots, Random, DifferentialEquations
export driver

function driver()
	sol = dynamics()
	pl = plot(layout=(2,1),size=(1000,800))
	plot!(sol.t,[sol[1,:],sol[2,:]], subplot=1,lw=[2 3],
		label=["input value" "internal average"])
	plot!(sol.t,[sol[4,:],
		[receptor0(sol[2,i],sol[1,i],100,10,2,1) .* 25 for i in 1:length(sol[1,:])]],
		subplot=2,lw=[3 2.5],xlabel="Time", label=["anomaly jumps" "receptor response"])
	display(pl)
	return sol, pl
end

function dynamics()
	default(; lw=2)
	jump_amplitude = 0.2  	# Amplitude of the jumps
	rate(u,p,t) = 0.2		# Rate of the Poisson process
	
	function f!(du, u, p, t)
		du[1] = 0.0002(100 - u[1])
		du[2] = 5.0(u[1] - u[2])
		du[3] = 2.0(1 - u[3])
		du[4] = -60*u[4]
	end
	function g!(du, u, p, t) 
		du[1] = 0.02*u[1]
		du[2:4] .= zeros(3)
	end
	affect!(integrator) = (integrator.u[1] *= jmp=1+sign(randn())*0.05;
		integrator.u[4] = (jmp>1) ? 1 : -1)

	u0 = [100.0,100.0,1.0,0.0]
	tspan = (0.0, 20.0)
	
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

##########################################################################

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
