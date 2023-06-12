using DelimitedFiles, DifferentialEquations, DiffEqOperators, Dierckx, SparseArrays, Roots, Sundials, LinearAlgebra, LoopVectorization, JLD2
using AdaptiveSparseGrids
using CairoMakie, MathTeXEngine
include("bikt.jl")
using .bikt

const L = 1000.0
const N = 1+2^12
const h = L/(N-1)
const ξ = range(-2*L/4,+2*L/4,length=N); @assert isapprox(h, diff(ξ)[1])
const d = 12
const B = Neumann0BC(h, d)
const δ1 = CenteredDifference(1, d, h, N)
const δ2 = CenteredDifference(2, d, h, N)
const ∇ = sparse((δ1*B))[1]
const Δ = sparse((δ2*B))[1]
const atol = 1e-06
const rtol = 1e-06
const jac_upper = max(length(B.a_l), length(B.a_r), (δ1.stencil_length-1)÷2, (δ2.stencil_length-1)÷2)

const sampled_lock = ReentrantLock();
const resolved_lock = ReentrantLock();

const plotting = parse(Bool, ARGS[1])
const writing  = parse(Bool, ARGS[2])

function H(x, k)
	return 0.5*(1.0+tanh(k*x))
end

function f!(dx, x, p, t)

	u  = @view x[1,:]
	v  = @view x[2,:]
	
	du = @view dx[1,:]
	dv = @view dx[2,:]
	
	@views tau, k, c = p[1:3]
	
	@inbounds begin
		@turbo for n in eachindex(u,v,du,dv)
			du[n] = H(u[n]-1.0, k)*v[n]
			dv[n] = (H(-u[n], k)-v[n])/tau
		end
	end
	mul!(du, Δ, u, 1.0, 1.0) 	# mul!(C, A, B, a, b) = A*B*a + C*b -> du .= (Δ*u)*(ep/Cm) .+ du.*(1.0/(ENa-E1)); correct
	
	return nothing
end

function updateState!(u0::Array{T,2}, pp) where T <: Real
	eh = state(pp)
	for (n,z) in enumerate(ξ)
		u0[:,n] .= eh(z)
	end
	return nothing
end

function Ψ(u)
	spl = Spline1D(ξ, u; k=5)
	spl2 = Spline1D(ξ, derivative(spl, ξ).^2; k=5)
	return sqrt(integrate(spl2, minimum(ξ), maximum(ξ)))
end

function adapt(p, set; 
			lower_bound = [+(2*h), minimum(ξ)+(L/4)+h],
			upper_bound = [+(L/2), maximum(ξ)-(L/4)-h], 
			max_depth = Int(ceil(log2(maximum(upper_bound .- lower_bound) / (h)))))
		
		# set stiffness parameter
		p[:k] = 10.0
		
		# copy dicts
		p̂ = copy(p); p̌ = copy(p);
		
		# determine parameters for the slow front
		p̂[:c] = minimum(c(p̂))
		params!(p̂)
		û = Array{Float64, 2}(undef, 2, N);
		updateState!(û, p̂);
		@show ψ̂ = Ψ(û[1,:])
		
		# determine parameters for the fast front
		p̌[:c] = maximum(c(p̌))
		params!(p̌)
		ǔ = Array{Float64, 2}(undef, 2, N);
		updateState!(ǔ, p̌);
		@show ψ̌ = Ψ(ǔ[1,:])
		
		# set integration parameters
		tspan = (0., round(0.95*abs(ξ[begin])/p̌[:c]));
		u0 = copy(ǔ);
		p = [p̌[:tau], p̌[:k], 0.0]
		
		function plotSol(sol)
			if plotting
				let fig = Figure(fonts = (; regular = texfont(), bold = texfont()))
					ga = fig[1,1] = GridLayout()
					axs = [Axis(ga[1,1], xlabel = "", ylabel = L"$x$"), Axis(ga[2,1], xlabel = L"$t$", ylabel = L"$\phi(t) - \hat{\phi}$")]
					hm1 = heatmap!(axs[1], sol.t, ξ, transpose(sol[1,:,:]), colormap=:viridis, colorrange = (-p̌[:alpha], p̌[:omega]), rasterize=4, highclip = :pink, lowclip = :red)
					Colorbar(ga[0, 1], hm1, label=L"$u_1(t, x)$", vertical=false, flipaxis=true, ticks = ([-p̌[:alpha] , 0.0, 1.0, p̌[:omega]], [L"-\alpha", "0.0", "1.0", L"\omega"]))
					lines!(axs[2], sol.t, [Ψ(sol[1,:,n]) for n in 1:size(sol,3)] .- ψ̂, label=L"$\phi(t)-\hat{\phi}$")
					xlims!.(axs, tspan[begin], tspan[end])
					ylims!(axs[1], ξ[begin], ξ[end])
					hidexdecorations!(axs[1], ticks=false, label=false, ticklabels=true)
					save("./bikt_sol.pdf", fig, pt_per_unit=1);
				end
			end
			return nothing
		end
		
		function writeSamples(q, psi)
			if writing
				open("./sampled_$(set).dat","a") do io
					writedlm(io, [q[1] q[2] q[3] psi]);
					@info "Appended $([q[1] q[2] q[3] psi]) to './sampled_$(set).dat'."
				end
			end
			return nothing
		end
		
		function X(q; H = sign) # H = x->0.5*(1.0+tanh(100.0*x))
			xs, th, A = q
			out = zeros(Float64,2,N)
			out[1,:] .= (A/4.0) .* (1.0.+H.(ξ.-th.+xs/2.0)) .* (1.0.-H.(ξ.-th.-xs/2.0))
			return out
		end
		
		function f(q; pltt = false, lck = sampled_lock)
			u = u0 .+ X(q)
			prob = ODEProblem(f!, u, tspan, p)
			if pltt
				saveat=range(tspan[begin], tspan[end]; length=257)
			else
				saveat=[tspan[end]]
			end
			sol = solve(prob, CVODE_BDF(linear_solver=:Band, jac_upper=jac_upper, jac_lower=jac_upper); maxiters=Int(1e8), abstol=atol, reltol=rtol, saveat=saveat)
			psi = Ψ(sol[1,:,end])
			lock(lck) do
				if pltt
					plotSol(sol);
				else
					writeSamples(q, psi);
				end
			end
			return psi
		end
		
		@assert f([0.0, 0.0, 0.0]; pltt=true) > ψ̂
		
		function plotResolved(set)
			if plotting
				let dat = readdlm("./resolved_$(set).dat");
					inds = findall(.~isnan.(dat[:,3]))
					cind = findall(isnan.(dat[:,3]))
					dfs = [maximum(dat[inds,n])-minimum(dat[inds,n]) for n in size(dat,2)];
					tri = all(dfs .> 0.0)
					fig = Figure(fonts = (; regular = texfont(), bold = texfont()))
					ga = fig[1,1] = GridLayout()
					axs = Axis(ga[1,1], xlabel = L"$x_s$", ylabel = L"\theta");
					if tri
						tr1 = tricontourf!(axs, dat[inds,1], dat[inds,2], dat[inds,3], colormap=:Oranges, colorrange=(-1000.0,0.0))
						Colorbar(ga[1, 2], tr1, label=L"$U_s$")
					end
					scatter!(dat[inds,1], dat[inds,2], color=dat[inds,3], colormap=:Oranges, strokewidth=1, markersize=3, strokecolor=:black)
					scatter!(dat[cind,1], dat[cind,2], strokewidth=1, markersize=3, strokecolor=:black)
					xlims!(axs, [lower_bound[1], upper_bound[1]])
					ylims!(axs, [lower_bound[2], upper_bound[2]])
					save("./resolved_$(set).pdf", fig, pt_per_unit=1);
				end
			end
			return nothing
		end	

		function writeResolved(xs, th, Us, set)
			if writing
				open("./resolved_$(set).dat","a") do io
					writedlm(io, [xs th Us]);
					@info "Appended $([xs th Us]) to './resolved_$(set).dat'."
				end
			end
			return nothing
		end
		
		function g(q; Us_lims=(-1.0e4, -0.0), method=Roots.A42(), lck = resolved_lock, Us_fail = +1.0)
			xs, th = q
			function ff(A::Float64)
				qq = (xs, th, A)
				tmp = f(qq)-ψ̂		# call model
				return tmp
			end
			
			while sign(ff(minimum(Us_lims))) == sign(ψ̌ -ψ̂) && abs(minimum(Us_lims)-maximum(Us_lims)) > rtol
				Us_lims = (minimum(Us_lims)/2.0, maximum(Us_lims))
			end
			if abs(minimum(Us_lims)-maximum(Us_lims)) <= rtol
				Us = NaN
			else
				zp = ZeroProblem(ff, Us_lims)
				Us = solve(zp, method; verbose=true, xatol=atol, xrtol=rtol)
			end
			lock(resolved_lock) do
				writeResolved(xs, th, Us, set);
			end
			if !isnan(Us)
				f([xs, th, Us]; pltt=true);
				try
					plotResolved(set);
				catch err
					@show err
				end
				return Us
			else
				return Us_fail
			end
		end

		Us = AdaptiveSparseGrid(g, 
						lower_bound, upper_bound,
						max_depth = max_depth,	# The maximum depth of basis elements in 
										# each dimension
						tol = 1.0e+0)		# Add nodes when 
										# min(abs(alpha/f(x)), abs(alpha)) < tol
		@save "fun_$(set).jld2" Us
		
	return nothing
end

function main()
	ps 	= [ 	Dict(:alpha => 1.0, :tau => 8.2),
			Dict(:alpha => 1.0, :tau => 11.0), 
			Dict(:alpha => 0.5625, :tau => 8.2), 
			Dict(:alpha => 0.5625, :tau => 7.7) ]
	sets 	= [ 1,2,3,4 ]
	
	#=
	for (p,set) in zip(ps, sets)
		adapt(p, set; max_depth = 9);
	end
	=#
	
	set = parse(Int, ARGS[3]);
	rm("./sampled_$(set).dat", force=true)
	rm("./resolved_$(set).dat", force=true)
	adapt(ps[set], set; max_depth=9);
	
	return nothing
end

main()
