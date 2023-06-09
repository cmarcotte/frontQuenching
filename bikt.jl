module bikt

using Roots

export state, c, params!

function alpha!(p::Dict{Symbol, T}) where T <: Real
	@views tau = p[:tau]
	alpha = (exp(1 - tau/exp(1))*(1 + (5exp(1))/(2tau)), tau*(1 - 2/sqrt(tau)))
	p[:alpha] = alpha[1] + (alpha[2]-alpha[1])*rand()
	return nothing
end

function c(p::Dict{Symbol, T}) where T <: Real
	@views tau, alpha = p[:tau], p[:alpha]
	f(c) = tau*c*c*log((1.0+alpha)*(1.0+tau*c*c)/tau) + log((1.0+alpha)/alpha)
	C = (exp(0.5)/tau + 0.75*exp(1.5)/(tau^2), 2*(exp(0.5)/tau + 0.75*exp(1.5)/(tau^2)))
	cc = (find_zero(f, C[1]), find_zero(f, C[2]))
	if cc[1] ≈ cc[2]
		cc = find_zeros(f, (min(0.0,minimum(C)),max(1.0, maximum(C))))
	end
	return cc
end

function params!(p::Dict{Symbol, T}) where T <: Real
	@views tau, alpha, c = p[:tau], p[:alpha], p[:c]
	p[:omega] = 1.0 + p[:tau]*c*c*(alpha+1.0)
	p[:x1] = log((1.0+alpha)/alpha)/c
	return p
end

function state(p::Dict{Symbol, T}) where T <: Real
	@views tau, alpha, c = p[:tau], p[:alpha], p[:c]
	E(z) = z <= p[:x1] ? alpha * (exp(c*z) - 1.0) : p[:omega] - exp(-z/tau/c)*((tau*c)^2 / (1.0 + tau*c^2))
	h(z) = z <= 0.0 ? 1.0 : exp(-z/tau/c)
	return (z)->(E(z),h(z))
end

function eigenvalues(p::Dict{Symbol, T}) where T <: Real
	@views tau, alpha, c = p[:tau], p[:alpha], p[:c]
	function f(l)
		v0 = (1 + tau*c*c)/(c*tau)
		v1 = (1+l*tau)/(c*tau)
		if c^2 + 4*l > 0.0
			v2 = (c + sqrt(c^2 + 4*l))/2
         	vc2= (c - sqrt(c^2 + 4*l))/2
		else
			v2 = (c + im*sqrt(-(c*c + 4*l)))/2
         	vc2= (c - im*sqrt(-(c*c + 4*l)))/2
		end
		return alpha*c*(v2-vc2)*exp(v0*x1) - 1.0 + (tau*c*(v1+vc2)) * exp(-(v1+v2-v0)*x1)/((1+l*tau)^2 + tau*c^2, v0, v1, v2, vc2)
	end
	l1 = find_zero((l)->f(l)[1], 0.1, Order0()) 
	return f(l1)
end

function eigenfunctions(p::Dict{Symbol, T}) where T <: Real
	@warn "This is not done!"
	return nothing
end

end
