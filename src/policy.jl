abstract type AbstractNNPolicy  <: Policy end

## NN Policy interface

"""
    getnetwork(policy)
    return the  value network of the policy 
"""
function getnetwork end 

"""
    resetstate!(policy)
reset the hidden states of a policy
"""
function resetstate! end

struct NNPolicy{P <: Union{MDP, POMDP}, Q, A} <: AbstractNNPolicy 
    problem::P
    qnetwork::Q
    action_map::Vector{A}
    n_input_dims::Int64
    sample_prob::Bool
    action_probability::Function
end

function getnetwork(policy::NNPolicy)
    return policy.qnetwork
end

function resetstate!(policy::NNPolicy)
    Flux.reset!(policy.qnetwork)
end

actionmap(p::NNPolicy) = p.action_map

function _action(policy::NNPolicy{P,Q,A}, o::AbstractArray{T, N}) where {P<:Union{MDP,POMDP},Q,A,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        vals = policy.qnetwork(obatch)
        if policy.sample_prob
            aprobs =  dropdims(policy.action_probability(obatch), dims = 2)
            probs = aprobs .* dropdims(vals, dims = 2)
            probs = (sum(probs) == 0) ? aprobs : probs ./ sum(probs)
            policy.action_map[rand(Categorical(probs))]
        else
            return policy.action_map[argmax(vals)]
        end
    else
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function Distributions.logpdf(policy::NNPolicy{P,Q,A}, o, a) where {P<:Union{MDP,POMDP},Q,A}
    @assert policy.sample_prob
    obatch = reshape(o, (size(o)...,1))
    vals = policy.qnetwork(obatch)
    aprobs =  dropdims(policy.action_probability(obatch), dims = 2)
    probs = aprobs .* dropdims(vals, dims = 2)
    probs = (sum(probs) == 0) ? aprobs : probs ./ sum(probs)
    log(probs[findfirst(policy.action_map .== [a])])
end

function _actionvalues(policy::NNPolicy{P,Q,A}, o::AbstractArray{T,N}) where {P<:Union{MDP,POMDP},Q,A,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        return dropdims(policy.qnetwork(obatch), dims=2)
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function _value(policy::NNPolicy{P}, o::AbstractArray{T,N}) where {P<:Union{MDP,POMDP},T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        vals = policy.qnetwork(obatch)
        if policy.sample_prob
            return sum(policy.action_probability(obatch) .* vals)
        else
            return maximum(vals)
        end
    else
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function POMDPs.action(policy::NNPolicy{P}, s) where {P <: MDP}
    _action(policy, convert_s(Array{Float32}, s, policy.problem))
end

function POMDPs.action(policy::NNPolicy{P}, o) where {P <: POMDP}
    _action(policy, convert_o(Array{Float32}, o, policy.problem))
end

function POMDPPolicies.actionvalues(policy::NNPolicy{P}, s) where {P<:MDP}
    _actionvalues(policy, convert_s(Array{Float32}, s, policy.problem))
end

function POMDPPolicies.actionvalues(policy::NNPolicy{P}, o) where {P<:POMDP}
    _actionvalues(policy, convert_o(Array{Float32}, o, policy.problem))
end

function POMDPs.value(policy::NNPolicy{P}, s) where {P <: MDP}
    _value(policy, convert_s(Array{Float32}, s, policy.problem))
end

function POMDPs.value(policy::NNPolicy{P}, o) where {P <: POMDP}
    _value(policy, convert_o(Array{Float32}, o, policy.problem))
end
