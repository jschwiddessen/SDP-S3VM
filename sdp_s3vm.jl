
using Mosek, JuMP, MosekTools, LinearAlgebra, DelimitedFiles, Printf, DataStructures, Gurobi, Statistics, Random, BenchmarkTools, ScikitLearn


# create environment that will be reused of all Gurobi models
if !@isdefined(gurobi_env)
	const gurobi_env = Gurobi.Env()
end


# these are constants
const TOL_BOUND = 1e-5
const TOL_P = 1e-5
const TOL_D = 1e-4


Base.@kwdef struct MyParameters
	minimum_violation::Float64
	gap_prune::Float64
	recompute_bound_constraints::Bool
	rel_tol_cutting_plane_approach::Float64
	max_cutting_plane_iter::Int64
	max_bb_nodes::Int64
	branching_rule::Int64
	time_limit::Float64
	factor_maximum_cuts_dualbound::Float64
	use_bound_tightening::Bool
	use_local_search::Bool
	num_runs_heuristics::Int64
	locally_infeasible_solutions::Int64
	penalty_factor_unlabeled::Float64
	kfold::Int64
	seed::Int64
	perc::Float64
	kernel_selection::String
	kernel_type::String
	gamma::Float64
	penalty_parameter::Float64
	penalty_parameter_min::Float64
	penalty_parameter_max::Float64
	penalty_parameter_num_steps::Int64
	branching_epsilon::Float64
	use_everything_for_cross_validation::Bool
	add_product_constraints::Bool
	use_small_relaxation::Bool
	num_threads::Int64
end


function get_parameters(filename::String)

	# set default values
	minimum_violation::Float64 = -1e-2
	gap_prune::Float64 = 1e-3
	recompute_bound_constraints::Bool = true
	rel_tol_cutting_plane_approach::Float64 = 1e-3
	max_cutting_plane_iter::Int64 = 25
	max_bb_nodes::Int64 = 1e10
	branching_rule::Int64 = 2
	time_limit::Float64 = 1e10
	factor_maximum_cuts_dualbound::Float64 = 5.0
	use_bound_tightening::Bool = true
	use_local_search::Bool = true
	num_runs_heuristics::Int64 = 1
	locally_infeasible_solutions::Int64 = 1
	penalty_factor_unlabeled::Float64 = 0.5
	kfold::Int64 = 10
	seed::Int64 = 12345
	perc::Float64 = 0.3
	kernel_selection::String = "auto"
	kernel_type::String = "linear"
	gamma::Float64 = 0.0
	penalty_parameter::Float64 = 1.0
	penalty_parameter_min::Float64 = 0.01
	penalty_parameter_max::Float64 = 100.0
	penalty_parameter_num_steps::Int64 = 20
	branching_epsilon::Float64 = 0.3
	use_everything_for_cross_validation::Bool = false
	add_product_constraints::Bool = true
	use_small_relaxation::Bool = true
	num_threads::Int64 = 0

	content = read(filename, String)

	for line in split(content, '\n')

		line_modified = split(line, '#', limit=2)[1] # remove comments
        parts = split(line_modified, '=')

        if length(parts) == 2

            key = strip(parts[1])
            value = strip(parts[2])

            if key == "minimum_violation"
                minimum_violation = parse(Float64, value)
            elseif key == "gap_prune"
                gap_prune = parse(Float64, value)
            elseif key == "recompute_bound_constraints"
                recompute_bound_constraints = parse(Bool, value)
			elseif key == "rel_tol_cutting_plane_approach"
                rel_tol_cutting_plane_approach = parse(Float64, value)
            elseif key == "max_cutting_plane_iter"
                max_cutting_plane_iter = parse(Int64, value)
			elseif key == "max_bb_nodes"
                max_bb_nodes = parse(Int64, value)
			elseif key == "branching_rule"
                branching_rule = parse(Int64, value)
			elseif key == "time_limit"
                time_limit = parse(Float64, value)
			elseif key == "factor_maximum_cuts_dualbound"
                factor_maximum_cuts_dualbound = parse(Float64, value)
			elseif key == "use_bound_tightening"
                use_bound_tightening = parse(Bool, value)
			elseif key == "use_local_search"
                use_local_search = parse(Bool, value)
			elseif key == "num_runs_heuristics"
                num_runs_heuristics = parse(Int64, value)
			elseif key == "locally_infeasible_solutions"
                locally_infeasible_solutions = parse(Int64, value)
			elseif key == "penalty_factor_unlabeled"
                penalty_factor_unlabeled = parse(Float64, value)
			elseif key == "kfold"
                kfold = parse(Int64, value)
			elseif key == "seed"
                seed = parse(Int64, value)
			elseif key == "perc"
                perc = parse(Float64, value)
			elseif key == "kernel_selection"
                kernel_selection = value
			elseif key == "kernel_type"
                kernel_type = value
			elseif key == "gamma"
                gamma = parse(Float64, value)
			elseif key == "penalty_parameter"
                penalty_parameter = parse(Float64, value)
			elseif key == "penalty_parameter_min"
                penalty_parameter_min = parse(Float64, value)
			elseif key == "penalty_parameter_max"
                penalty_parameter_max = parse(Float64, value)
			elseif key == "penalty_parameter_num_steps"
                penalty_parameter_num_steps = parse(Int64, value)
			elseif key == "branching_epsilon"
                branching_epsilon = parse(Float64, value)
			elseif key == "use_everything_for_cross_validation"
                use_everything_for_cross_validation = parse(Bool, value)
			elseif key == "add_product_constraints"
                add_product_constraints = parse(Bool, value)
			elseif key == "use_small_relaxation"
                use_small_relaxation = parse(Bool, value)
			elseif key == "num_threads"
                num_threads = parse(Int64, value)
			else
				s::String = @sprintf "Unknown parameter provided: \"%s\" with value \"%s\"\n" key value
				error(s) 
            end

        end
    end

	# check parameters
	if minimum_violation >= 0.0
		error("\"minimum_violation\" must be negative\n")
	end
	if gap_prune < 0.0
		error("\"gap_prune\" must be nonnegative\n")
	end
	if rel_tol_cutting_plane_approach < 0.0
		error("\"rel_tol_cutting_plane_approach\" must be nonnegative\n")
	end
	if max_cutting_plane_iter < 1
		error("\"max_cutting_plane_iter\" must be greater equal than 1\n")
	end
	if max_bb_nodes <= 0
		error("\"max_bb_nodes\" must be greater equal than 1\n")
	end
	if branching_rule > 3 || branching_rule < 0
		error("\"branching_rule\" must be set to 0, 1, 2, or 3\n")
	end
	if time_limit <= 0.0
		error("\"time_limit\" must be positive\n")
	end
	if factor_maximum_cuts_dualbound < 0.0
		error("\"factor_maximum_cuts_dualbound\" must be nonnegative\n")
	end
	if num_runs_heuristics < 0
		error("\"num_runs_heuristics\" must be nonnegative\n")
	end
	if locally_infeasible_solutions != 0 && locally_infeasible_solutions != 1 && locally_infeasible_solutions != 2
		error("\"locally_infeasible_solutions\" must be set to 0, 1, or 2\n")
	end
	if penalty_factor_unlabeled <= 0.0
		error("\"penalty_factor_unlabeled\" must be positive\n")
	end
	if kfold <= 1
		error("\"kfold\" must be at least 2\n")
	end
	if perc < 0.0 || perc > 1.0
		error("\"perc\" must be between 0.0 and 1.0\n")
	end
	if kernel_selection != "auto" && kernel_selection != "user"
		error("\"kernel_selection\" must be set to \"auto\" or \"user\"\n")
	end
	if kernel_type != "linear" && kernel_type != "rbf"
		error("\"kernel_type\" must be set to \"linear\" or \"rbf\"\n")
	end
	if gamma < 0.0
		error("\"gamma\" must be positive (or 0.0 for automatic choice)\n")
	end
	if penalty_parameter <= 0.0
		error("\"penalty_parameter\" must be positive\n")
	end
	if penalty_parameter_min <= 0.0
		error("\"penalty_parameter_min\" must be positive\n")
	end
	if penalty_parameter_max <= 0.0
		error("\"penalty_parameter_max\" must be positive\n")
	end
	if penalty_parameter_min >= penalty_parameter_max
		error("\"penalty_parameter_max\" must be strictly larger than \"penalty_parameter_min\"\n")
	end
	if penalty_parameter_num_steps < 2
		error("\"penalty_parameter_num_steps\" must be at least 2\n")
	end
	if branching_epsilon <= 0.0
		error("\"branching_epsilon\" must be positive\n")
	end
	if num_threads < 0
		error("\"num_threads\" must be nonnegative\n")
	end

	parameters::MyParameters = MyParameters(
		minimum_violation = minimum_violation,
		gap_prune = gap_prune,
		recompute_bound_constraints = recompute_bound_constraints,
		rel_tol_cutting_plane_approach = rel_tol_cutting_plane_approach,
		max_cutting_plane_iter = max_cutting_plane_iter,
		max_bb_nodes = max_bb_nodes,
		branching_rule = branching_rule,
		time_limit = time_limit,
		factor_maximum_cuts_dualbound = factor_maximum_cuts_dualbound,
		use_bound_tightening = use_bound_tightening,
		use_local_search = use_local_search,
		num_runs_heuristics = num_runs_heuristics,
		locally_infeasible_solutions = locally_infeasible_solutions,
		penalty_factor_unlabeled = penalty_factor_unlabeled,
		kfold = kfold,
		seed = seed,
		perc = perc,
		kernel_selection = kernel_selection,
		kernel_type = kernel_type,
		gamma = gamma,
		penalty_parameter = penalty_parameter,
		penalty_parameter_min = penalty_parameter_min,
		penalty_parameter_max = penalty_parameter_max,
		penalty_parameter_num_steps = penalty_parameter_num_steps,
		branching_epsilon = branching_epsilon,
		use_everything_for_cross_validation = use_everything_for_cross_validation,
		add_product_constraints = add_product_constraints,
		use_small_relaxation = use_small_relaxation,
		num_threads = num_threads
	)

	return parameters

end


struct RLT_Cut
    i::Int64 # 1 <= i < j <= n
    j::Int64 # 1 <= i < j <= n
    type::Int64 # 1 <= type <= 4
end


mutable struct Subproblem
    lower_bound::Base.Threads.Atomic{Float64}
	id::Int64
	depth::Int64
    L::Vector{Float64}
    U::Vector{Float64}
    x::Vector{Float64} # best known feasible solution for this subproblem
    UB::Float64 # best known local upper bound
    inherited_cuts::Vector{RLT_Cut} # inherited cuts from parent node
end


struct ReturnValue
	lower_bound::Float64
	x::Vector{Float64}
	X::Matrix{Float64}
	iter::Int64
	cutting_planes::Int64
	local_upper_bound::Float64
	volume_end::Float64
	volume_diff::Float64
	duals::Vector{Float64}
	Z::Matrix{Float64}
	rlt_sum::Vector{Float64}
	final_cuts::Vector{RLT_Cut}
	time_spent::Float64
	subproblem::Subproblem
end


struct Tri_Cut
	i::Int64 # 1 <= i < j < k <= n
	j::Int64 # 1 <= i < j < k <= n
	k::Int64 # 1 <= i < j < k <= n
	type::Int64 # 1 <= type <= 4
end


struct Branching_Decision
	index::Int64
	abs_x::Float64
	diff_x_best::Float64
	rlt_sum::Float64
	sum_Zi::Float64
	sum_Ci::Float64
	error1::Float64
	error2::Float64
	error3::Float64
	error4::Float64
	objective_diff::Float64
end


struct S3vm_instance
	file::String
	n::Int64
	d::Int64 # number of features
	X_data::Matrix{Float64}
	y::Vector{Float64} # possible entries are -1.0, 0.0, and +1.0
	true_labels::Vector{Float64} # entries are -1.0 and +1.0
	penalty_parameters::Vector{Float64}
	kernel_matrix::Matrix{Float64}
	D::Matrix{Float64}
	K::Matrix{Float64}
	Kinv::Matrix{Float64}
	C::Matrix{Float64}
	Cinv::Matrix{Float64}
	C_tilde::Matrix{Float64}
	kernel::String
	gamma::Float64
	upper_bound::Base.Threads.Atomic{Float64}
	x::Vector{Float64}
	labeling::Vector{Float64}
	predictions::Vector{Float64}
	lk::ReentrantLock
end



# returns objective value of labeling AND the corresponding best vector x
# does not check whether the labeling actually is feasible for the original problem (labeled data might be mislabeled)
function evaluate_labeling_qp(C::Matrix{Float64}, y::Vector{Float64}, l::Int64, rhs::Float64)

	n = length(y)

	#@assert typeof(n) == Int64
	#@assert n >= 1
	#@assert typeof(C) == Matrix{Float64}
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert typeof(y) == Vector{Float64}
	#@assert length(y) == n
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0

    model = Model(() -> Gurobi.Optimizer(gurobi_env); add_bridges = false)
    set_silent(model)
    
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "BarQCPConvTol", 1e-9)
    
	@variable(model, x_var[i = 1:n])
    
	@constraint(model, [i = 1:n], y[i] * x_var[i] >= 1.0)
    
	if l != 0
		@constraint(model, sum(x_var[i] for i in l+1:n) == rhs)
	end

    @objective(model, Min, x_var' * C * x_var)
    
	optimize!(model)
	
	if termination_status(model) != MOI.OPTIMAL
		@printf "in evaluate_labeling_qp():\n"
		@show termination_status(model)
	end
	
	##@assert termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.SLOW_PROGRESS

	x::Vector{Float64} = value.(x_var)
	#@assert length(x) == n
	
	for i = 1:n
		#@assert y[i] * x[i] > 1.0 - 1e-10
		if x[i] < 0.0
			x[i] = min(x[i], -1.0)
		end
		if x[i] > 0.0
			x[i] = max(x[i], +1.0)
		end
	end
	
	upper_bound = x' * C * x

    return upper_bound, x

end



function two_opt_local_search!(C::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64}, l::Int64, rhs::Float64)

	n::Int64 = length(x)
	
	#@assert (n, n) == size(C)
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert length(y) == n
	for i = 1:n
		#@assert y[i] == -1.0 || y[i] == 0.0 || y[i] == +1.0
	end
	for i = 1:l
		#@assert y[i] != 0.0
	end
	#@assert all(y .* x .>= zeros(Float64, n))
	#@assert minimum(abs.(x)) >= 1.0
	#@assert abs(1.0 * sum(y[1:l]) * (n - l) / l - rhs) / (1.0 + abs(rhs)) < 1e-6
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
	#@assert length(x) == n
	
	value_start::Float64 = x' * C * x
	
	while true
	
		improvement_found::Bool = false
		
		for i in shuffle(collect(1:n)), j in shuffle(collect(i+1:n))
		#for i in 1:n, j in i+1:n

			# only apply to unlabeled data points
			if y[i] != 0.0 || y[j] != 0.0
				continue
			end

			k::Float64 = x[i] + x[j]

			gi::Float64 = dot(C[:,i], x) - C[i,i] * x[i]
			
			gj::Float64 = dot(C[:,j], x) - C[j,j] * x[j]
		
			gi_tilde::Float64 = gi - C[i,j] * x[j]
			gj_tilde::Float64 = gj - C[i,j] * x[i]

			a::Float64 = C[i,i] + C[j,j] - 2.0 * C[i,j]
			b::Float64 = -2.0 * k * C[i,i] + 2.0 * k * C[i,j] - 2.0 * gi_tilde + 2.0 * gj_tilde
			#c::Float64 = C[i,i] * k * k + 2.0 * k * gi_tilde
		
			g_ij::Vector{Float64} = [gi_tilde; gj_tilde]
		
			C_ij::Matrix{Float64} = [C[i,i] C[i,j]; C[i,j] C[j,j]]
			
			x_ij::Vector{Float64} = [x[i]; x[j]]
			
			# compute current contribution of (x_i,x_j) to the objective
			current_contribution::Float64 = x_ij' * C_ij * x_ij + 2.0 * dot(g_ij, x_ij)

			#contribution(x) = x^2 * (C[i,i] + C[j,j] - 2.0 * C[i,j]) + x * (-2.0 * k * C[i,i] + 2.0 * k * C[i,j] - 2.0 * gi_tilde + 2.0 * gj_tilde) + C[i,i] * k^2 + 2.0 * gi_tilde * k

			##@assert abs(contribution(x[j]) - current_contribution) < 1e-7
			
			best_contribution::Float64 = +Inf
			best_solution::Vector{Float64} = [x[i]; x[j]]
			
			# solve unconstrained problem
			xj::Float64 = - 0.5 * b / a
			if abs(xj) < 1.0
				xj = sign(xj)
			end
			candidate::Vector{Float64} = [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			
			candidate_contribution::Float64 = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end
			
			# test xj = -1
			xj = -1.0
			candidate .= [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			candidate_contribution = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end

			# test xj = +1
			xj = +1.0
			candidate .= [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			candidate_contribution = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end

			# test xj = k - 1.0
			xj = k - 1.0
			if abs(xj) < 1.0
				xj = sign(xj)
			end
			candidate .= [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			candidate_contribution = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end

			# test xj = k + 1.0
			xj = k + 1.0
			if abs(xj) < 1.0
				xj = sign(xj)
			end
			candidate .= [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			candidate_contribution = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end

			# test xj = k / 2
			xj = 0.5 * k
			if abs(xj) < 1.0
				xj = sign(xj)
			end
			candidate .= [k - xj; xj]
			if abs(candidate[1]) < 1.0
				candidate[1] = sign(candidate[1])
			end
			candidate_contribution = candidate' * C_ij * candidate + 2.0 * dot(g_ij, candidate)
			if candidate_contribution < best_contribution && minimum(abs.(candidate)) >= 1.0 && abs(k - sum(candidate)) < 1e-8
				best_contribution = candidate_contribution
				best_solution .= candidate
			end

			#@assert best_contribution < current_contribution + (1e-6) * abs(current_contribution)
			
			if best_contribution < current_contribution
				value_before_update::Float64 = x' * C * x
				x[i] = best_solution[1]
				x[j] = best_solution[2]
				value_after_update::Float64 = x' * C * x
				value_after_update < value_before_update + (1e-6) * abs(value_before_update)
			end

			if best_contribution + (1e-6) * abs(best_contribution) < current_contribution
				improvement_found = true
			end
		
		end # i,j loop
		
		#@assert x' * C * x < value_start + (1e-6) * abs(value_start)
		#@assert minimum(abs.(x)) >= 1.0
		#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
		#@assert all(y .* x .>= zeros(Float64, n))
		#@assert length(x) == n
		
		if !improvement_found
			break # if we have not found any improvement, we can break and do not need to solve the QP again
		else

			value_before::Float64 = x' * C * x
			# solve convex QP to get the best x
			_, best_x::Vector{Float64} = evaluate_labeling_qp(C, sign.(x), l, rhs)
			x .= best_x
			value_after::Float64 = x' * C * x

			#@assert minimum(abs.(x)) >= 1.0
			#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
			#@assert all(y .* x .>= zeros(Float64, n))
			#@assert length(x) == n
			#@assert value_after < value_before + (1e-6) * abs(value_before)

			if value_after + (1e-6) * abs(value_after) > value_before
				break
			end

		end
		
	end # while loop
	
	#@assert x' * C * x < value_start + (1e-6) * abs(value_start)
	#@assert minimum(abs.(x)) >= 1.0
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
	#@assert all(y .* x .>= zeros(Float64, n))
	#@assert length(x) == n
	
	# solve convex QP to get best x
	_, best_x = evaluate_labeling_qp(C, sign.(x), l, rhs)
	x .= best_x
	
	#@assert x' * C * x < value_start + (1e-6) * abs(value_start)
	#@assert minimum(abs.(x)) >= 1.0
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
	#@assert all(y .* x .>= zeros(Float64, n))
	#@assert length(x) == n

end



function improve_x!(C::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64}, l::Int64, rhs::Float64)

	n::Int64 = length(x)
	#@assert (n, n) == size(C)
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert length(x) == n
	#@assert length(y) == n
	#@assert minimum(abs.(x)) >= 1.0
	for i = 1:n
		#@assert y[i] == -1.0 || y[i] == 0.0 || y[i] == +1.0
	end
	#@assert minimum(abs.(x)) >= 1.0
	#@assert abs(1.0 * sum(y[1:l]) * (n - l) / l - rhs) < 1e-8
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
		
	value_before = x' * C * x
	two_opt_local_search!(C, x, y, l, rhs)
	#@assert x' * C * x < value_before + 1e-6
	#@assert minimum(abs.(x)) >= 1.0
	for i = 1:n
		#@assert y[i] * x[i] >= 0.0
	end
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6
	
	# final checks
	for i = 1:n
		#@assert y[i] * x[i] >= 0.0
	end
	#@assert minimum(abs.(x)) >= 1.0
	#@assert abs(sum(x[l+1:n]) - rhs) / (1.0 + abs(rhs)) < 1e-6

end


function heuristic_labeling(instance::S3vm_instance, use_local_search::Bool, labeling::Vector{Float64})

	n::Int64 = instance.n
	#@assert minimum(abs.(labeling)) == 1.0
	#@assert maximum(abs.(labeling)) == 1.0
	for i = 1:n
		#@assert instance.y[i] * labeling[i] >= 0.0
	end

	l::Int64 = count(x -> x != 0.0, instance.y)
	rhs::Float64 = 1.0 * sum(instance.y[1:l]) * (n - l) / l

	upper_bound::Float64, x::Vector{Float64} = evaluate_labeling_qp(instance.C, labeling, l, rhs)

	if use_local_search
		improve_x!(instance.C, x, instance.y, l, rhs)
	end

	try_to_update_global_upper_bound(n, x' * instance.C * x, x, sign.(x), instance)

end




# returns whether a better solution was found and the value of the best solution found
function run_heuristic(
	C::Matrix{Float64},
	x_sdp::Vector{Float64},
	X_sdp::Matrix{Float64},
	y::Vector{Float64},
	subproblem::Union{Subproblem,Nothing},
	instance::S3vm_instance,
	num_runs::Int64,
	use_local_search::Bool)

	n::Int64 = length(x_sdp)

	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert length(x_sdp) == n
	#@assert issymmetric(X_sdp)
	#@assert size(X_sdp,1) == n
	#@assert size(X_sdp,2) == n
	#@assert length(y) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	for i = 1:n
		#@assert y[i] == -1.0 || y[i] == 0.0 || y[i] == +1.0
	end
	
	success::Bool = false
	best_upper_bound::Float64 = +Inf
	best_x::Vector{Float64} = ones(Float64,n)

	l::Int64 = count(x -> x != 0.0, instance.y)
	rhs::Float64 = 1.0 * sum(instance.y[1:l]) * (n - l) / l
	
	for run = 1:num_runs
	
		# construct labeling vector using SDP solution
		x::Vector{Float64} = sign.(x_sdp)
		upper_bound::Float64 = x' * C * x
		
		if run % 2 == 1
			# solve convex QP to get a primal feasible solution and an upper bound
			upper_bound, x = evaluate_labeling_qp(C, x, l , rhs)
		end

		#@assert abs(x' * C * x - upper_bound) / abs(upper_bound) < 1e-8
		#@assert minimum(abs.(x)) >= 1.0
		
		# try to improve the primal feasible solution
		if use_local_search
			improve_x!(C, x, y, l, rhs)
		end
		
		#@assert x' * C * x < upper_bound + 1e-6
		#@assert minimum(abs.(x)) >= 1.0
		
		# solve the convex QP again to get the final upper bound
		labeling::Vector{Float64} = sign.(x)
		upper_bound, x = evaluate_labeling_qp(C, labeling, l, rhs)
		
		# check that the labeling vector is actually feasible
		for i = 1:n
			#@assert y[i] * x[i] >= 0.0
		end
		
		if try_to_update_global_upper_bound(n, upper_bound, x, labeling, instance)
			success = true
		end
		
		if upper_bound < best_upper_bound
			best_upper_bound = upper_bound
			best_x = deepcopy(x)
		end
		
	end
	
	if !isnothing(subproblem)
		feasible::Bool = true
		for i = 1:n
			if subproblem.L[i] > 0.0 && best_x[i] < 0.0 || subproblem.U[i] < 0.0 && best_x[i] > 0.0
				feasible = false
			end
		end
		if feasible && best_x' * C * best_x < subproblem.UB
			subproblem.x .= best_x
			subproblem.UB = best_x' * C * best_x
		end
	end
	
	return success, best_upper_bound
	
end


function print_accuracy(instance::S3vm_instance)

	num_labeled::Int64 = 0
	num_unlabeled::Int64 = 0

	correct_labeled_direct::Int64 = 0
	correct_labeled_prediction::Int64 = 0
	correct_unlabeled_direct::Int64 = 0
	correct_unlabeled_prediction::Int64 = 0

	n::Int64 = instance.n

	for i = 1:n

		if instance.y[i] == 0.0
			num_unlabeled += 1
			if instance.labeling[i] == instance.true_labels[i]
				correct_unlabeled_direct += 1
			end
			if instance.predictions[i] == instance.true_labels[i]
				correct_unlabeled_prediction += 1
			end
		else
			num_labeled += 1
			if instance.labeling[i] == instance.true_labels[i]
				correct_labeled_direct += 1
			end
			if instance.predictions[i] == instance.true_labels[i]
				correct_labeled_prediction += 1
			end
		end

	end

	@printf "accuracy on labeled data points using the direct approach: %.2f%%\n" 100.0 * correct_labeled_direct / num_labeled
	@printf "accuracy on labeled data points using the prediction approach: %.2f%%\n" 100.0 * correct_labeled_prediction / num_labeled
	@printf "accuracy on unlabeled data points using the direct approach: %.2f%%\n" 100.0 * correct_unlabeled_direct / num_unlabeled
	@printf "accuracy on unlabeled data points using the prediction approach: %.2f%%\n" 100.0 * correct_unlabeled_prediction / num_unlabeled

	return 1.0 * correct_labeled_direct / num_labeled, 1.0 * correct_labeled_prediction / num_labeled, 1.0 * correct_unlabeled_direct / num_unlabeled, 1.0 * correct_unlabeled_prediction / num_unlabeled

end	


function compute_predictions(instance::S3vm_instance)

	n::Int64 = instance.n

	#@assert sign.(instance.x) == instance.labeling

	@printf "To compute the predictions, we do the following:\n"
	@printf "Let v be the solution found for the nonconvex problem.\n"
	@printf "We compute the corresponding labeling sign(v) and with this labeling we solve the normal convex QP without the balancing constraint.\n"
	@printf "Then we compute \"alpha\" by using the solution of this problem.\n"
	@printf "\"alpha\" is then used to make predictions.\n"

	_, x = evaluate_labeling_qp(instance.C, instance.labeling, 0, 0.0)

	#alpha::Vector{Float64} = inv(instance.K .* (instance.labeling * instance.labeling')) * (instance.x .* instance.labeling)
	alpha::Vector{Float64} = inv(instance.K .* (instance.labeling * instance.labeling')) * (x .* instance.labeling)
	#@assert length(alpha) == n
	if minimum(alpha) < -1e-6
		@printf "\n"
		@show minimum(alpha)
		@printf "We project alpha but there is an issue here!\n\n"
	end
	
	for i = 1:n
		##@assert alpha[i] > -1e-6
		alpha[i] = max(0.0, alpha[i])
	end
	
	# compute predictions
	for i = 1:n
	
		value = 0.0
		for j = 1:n
			value += instance.labeling[j] * alpha[j] * instance.kernel_matrix[i,j]
		end
		
		instance.predictions[i] = sign(value)
		
	end
	
end


function compute_w(instance::S3vm_instance)

	#@assert instance.kernel == "linear"

	n::Int64 = instance.n

	#@assert sign.(instance.x) == instance.labeling

	_, x = evaluate_labeling_qp(instance.C, instance.labeling, 0, 0.0)

	alpha::Vector{Float64} = inv(instance.K .* (instance.labeling * instance.labeling')) * (x .* instance.labeling)
	#@assert length(alpha) == n
	for i = 1:n
		#@assert alpha[i] > -1e-6
		alpha[i] = max(0.0, alpha[i])
	end
	
	w::Vector{Float64} = zeros(Float64,instance.d)

	for i = 1:n
		w += instance.labeling[i] * alpha[i] * instance.X_data[i,:]	
	end

	return w
	
end


# updates global_upper_bound
# "upper_bound" must be computed be calling function
# WARNING: be careful when a parallel B&B code is used!!!
function try_to_update_global_upper_bound(
	n::Int64,
	new_upper_bound::Float64,
	new_x::Vector{Float64},
	new_labeling::Vector{Float64},
	instance::S3vm_instance)

	#@assert n >= 1
	#@assert n == instance.n
	#@assert new_upper_bound > 0.0
	#@assert length(new_x) == n
	#@assert minimum(abs.(new_x)) >= 1.0
	#@assert length(new_labeling) == n
	#@assert minimum(abs.(new_labeling)) == 1.0
	#@assert maximum(abs.(new_labeling)) == 1.0
	#@assert new_labeling == sign.(new_x)
	#@assert abs(new_upper_bound - new_x' * instance.C * new_x) < 1e-8
	for i = 1:n
		#@assert instance.y[i] == 0.0 || new_labeling[i] == instance.y[i]
	end

	success::Bool = false

	if new_upper_bound < instance.upper_bound[]

		lock(instance.lk) do

			# TODO we assume here that new_x is the global minimum of the convex QP with respect to the labeling
			same_labeling::Bool = (new_labeling == instance.labeling)

			if new_upper_bound < instance.upper_bound[] && !same_labeling

				Threads.atomic_xchg!(instance.upper_bound, new_upper_bound)

				instance.x .= new_x
				instance.labeling .= new_labeling
			
				@printf "new upper bound : %.8f\n" new_upper_bound
				
				compute_predictions(instance)
				print_accuracy(instance)
				
				success = true # found better solution
				
			end

		end
	
	end
	
	return success # returns whether a better upper bound was found

end



# can be called at any time
# updates L_i and U_i by projecting on (-inf,-1] and [1,inf)
function project_bound_constraints!(L::Vector{Float64}, U::Vector{Float64})

    n::Int64 = length(L)

	#@assert length(L) == n
	#@assert length(U) == n
	#@assert TOL_BOUND > 0.0

    for i = 1:n

		#@assert L[i] < U[i] + TOL_BOUND

        if L[i] > -1.0 + TOL_BOUND
            L[i] = max(L[i], 1.0)
        end

        if U[i] < 1.0 - TOL_BOUND
            U[i] = min(U[i], -1.0)
        end

		#@assert L[i] < U[i] + TOL_BOUND

    end

end




function compute_number_of_labeled_data_points(L::Vector{Float64}, U::Vector{Float64})

    n::Int64 = length(L)

	#@assert length(L) == n
	#@assert length(U) == n

    count::Int64 = 0

    for i = 1:n
    
        if L[i] > 0.0 || U[i] < 0.0
            count += 1
        end
        
        #@assert L[i] < U[i] + TOL_BOUND # otherwise, the subproblem is infeasible
        
    end

    return count
end



function read_data(file::String, seed::Int64, perc::Float64)

	#@assert 0.0 <= perc && perc <= 1.0

	# read input
	@printf "We read instance \"%s\".\n" basename(file)
    data = readdlm(file)
    
	# shuffle randomly
	@printf "We now shuffle all data points (rows) using the specified seed %d.\n" seed
	Random.seed!(seed) # always needs to be set again to ensure reproducibility
    data = data[shuffle(1:end), :]
	
    #@assert size(data,1) >= 2
    #@assert size(data,2) >= 3
    
    num_features::Int64 = size(data, 2) - 1
    n::Int64 = size(data,1)
    X_data::Matrix{Float64} = data[:, 1:num_features]
    y::Vector{Float64} = data[:, num_features + 1]

    # check that the data is standardized, i.e., each column (feature) has mean 0 and standard deviation 1
    for i = 1:num_features
    	#@assert abs(mean(X_data[:,i])) < 1e-8
    	#@assert abs(std(X_data[:,i]; corrected = false) - 1.0) < 1e-8 || abs(std(X_data[:,i]; corrected = true) - 1.0) < 1e-8
		if abs(mean(X_data[:,i])) >= 1e-8 || (abs(std(X_data[:,i]; corrected = false) - 1.0) >= 1e-8 && abs(std(X_data[:,i]; corrected = true) - 1.0)) >= 1e-8
			error("The data is not standardized.\n")
		end
    end

	sum_before::Float64 = sum(sum(X_data))
	norm_before::Float64 = norm(X_data)
	
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0
	
	size_class_minus_one::Int64 = length(findall(<(0), y))
	size_class_plus_one::Int64 = length(findall(>(0), y))

	num_select_minus_one::Int64 = floor(size_class_minus_one * perc)
	num_select_plus_one::Int64 = floor(size_class_plus_one * perc)

	@printf "The instance has %d data points and %d features.\n" n num_features
	@printf "There are %d data points with label \"-1\" (%.2f%%).\n" size_class_minus_one 100.0 * size_class_minus_one / n
	@printf "There are %d data points with label \"+1\" (%.2f%%).\n" size_class_plus_one 100.0 * size_class_plus_one / n

	@printf "We are asked to create an S3VM instance with %.2f%% labeled data points.\n" 100.0 * perc
	@printf "We select the first %d data points with label \"-1\" to be labeled (that's %.2f%% of all data points with label \"-1\").\n" num_select_minus_one 100.0 * num_select_minus_one / size_class_minus_one
	@printf "We select the first %d data points with label \"+1\" to be labeled (that's %.2f%% of all data points with label \"+1\").\n" num_select_plus_one 100.0 * num_select_plus_one / size_class_plus_one
	
	for i = 1:num_select_minus_one
		if y[i] != -1.0
			for j = i + 1:n
				if y[j] == -1.0
					# swap i and j
					temp = X_data[i,:]
					X_data[i,:] = X_data[j,:]
					X_data[j,:] = temp
					temp = y[i]
					y[i] = y[j]
					y[j] = temp
					break
				end
			end
		end
	end
	
	for i = num_select_minus_one + 1:num_select_minus_one + num_select_plus_one
		if y[i] != 1.0
			for j = i + 1:n
				if y[j] == 1.0
					# swap i and j
					temp = X_data[i,:]
					X_data[i,:] = X_data[j,:]
					X_data[j,:] = temp
					temp = y[i]
					y[i] = y[j]
					y[j] = temp
					break
				end
			end
		end
	end
	
	for i = 1:num_select_minus_one
		#@assert y[i] == -1.0
	end
	
	for i = num_select_minus_one + 1:num_select_minus_one + num_select_plus_one
		#@assert y[i] == +1.0
	end

	num_labeled::Int64 = num_select_minus_one + num_select_plus_one
	@printf "The first %d data points are now the labeled ones.\n" num_labeled
	data = [X_data y]
	@printf "We shuffle the labeled data points.\n"
	data = [data[shuffle(1:num_select_minus_one + num_select_plus_one), :]; data[num_select_minus_one + num_select_plus_one + 1:n, :]]
	@printf "We shuffle the unlabeled data points.\n"
	data = [data[1:num_select_minus_one + num_select_plus_one, :]; data[shuffle(num_select_minus_one + num_select_plus_one + 1:n), :]]
    X_data = data[:,1:num_features]
    y = data[:,num_features + 1]

	#@assert length(findall(<(0), y)) == size_class_minus_one
	#@assert length(findall(>(0), y)) == size_class_plus_one
	#@assert abs(norm(X_data) - norm_before) / (1.0 + norm_before) < 1e-8
	#@assert abs(sum(sum(X_data)) - sum_before) / (1.0 + sum_before) < 1e-8
	
	return n, X_data, y, num_labeled
	
end


function compute_kernel_matrix(n::Int64, X_data::Matrix{Float64}, kernel_type::String, gamma::Union{Nothing,Float64})

	#@assert n >= 1
	#@assert size(X_data,1) == n
	#@assert kernel_type == "linear" || kernel_type == "rbf"
	#@assert kernel_type == "linear" || kernel_type == "rbf" && gamma > 0.0
	
	kernel_matrix::Matrix{Float64} = zeros(n,n)
	
	if kernel_type == "linear"
	
		kernel_matrix = X_data * X_data'
		
	elseif kernel_type == "rbf"
	
		for i = 1:n
		
			kernel_matrix[i,i] = 1.0
			
			for j = i + 1:n
			
				kernel_matrix[i,j] = exp(- gamma * norm(X_data[i,:] - X_data[j,:])^2)
				kernel_matrix[j,i] = kernel_matrix[i,j]
			
			end
		end
		
	else
		#@assert false # no other kernel is implemented
	end
	
	#@assert issymmetric(kernel_matrix)
	#@assert eigmin(kernel_matrix) > -1e-8
	
	return kernel_matrix
	
end



function compute_and_print_accuracy_using_labeled_points(instance::S3vm_instance)

	n::Int64 = instance.n
	d::Int64 = instance.d

	X_data::Matrix{Float64} = deepcopy(instance.X_data)
	y::Vector{Float64} = deepcopy(instance.y)
	penalty_parameters = deepcopy(instance.penalty_parameters)
	permutation::Vector{Int64} = 1:n

	# sort such that the labeled data points are the first ones
	for i = 1:n
		if y[i] == 0.0
			for j = i + 1:n
				if y[j] != 0.0
					# swap i and j
					temp = X_data[i,:]
					X_data[i,:] = X_data[j,:]
					X_data[j,:] = temp
					temp = y[i]
					y[i] = y[j]
					y[j] = temp
					temp = penalty_parameters[i]
					penalty_parameters[i] = penalty_parameters[j]
					penalty_parameters[j] = temp
					temp = permutation[i]
					permutation[i] = permutation[j]
					permutation[j] = temp
					break
				end
			end
		end
	end

	num_labeled::Int64 = 0
	for i = 1:n
		if y[i] != 0.0
			num_labeled += 1
		end
	end

	# check that everything went fine
	for i = 1:num_labeled
		#@assert y[i] != 0.0
	end
	for i = num_labeled + 1:n
		#@assert y[i] == 0.0
	end

	# linear kernel
	kernel_matrix = compute_kernel_matrix(n, X_data, "linear", 0.0)
	K = kernel_matrix[1:num_labeled,1:num_labeled] + diagm(0.5 ./ penalty_parameters[1:num_labeled])
	#@assert issymmetric(K)
	#@assert eigmin(K) > 0.0
	Kinv = inv(K)
	Kinv = 0.5 * (Kinv + Kinv')
	C = 0.5 * Kinv
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0

	#l = count(x -> x != 0.0, instance.y)
	#rhs = 1.0 * sum(instance.y[1:l]) * (n - l) / l

	yy = y[1:num_labeled]
	_, sol = evaluate_labeling_qp(C, yy, 0, 0.0)
	
	alpha::Vector{Float64} = inv(K .* (yy * yy')) * (sol .* yy)
	for i = 1:num_labeled
		#@assert alpha[i] > -1e-6
		alpha[i] = max(0.0, alpha[i])
	end

	labeling_linear_permuted::Vector{Float64} = [y[1:num_labeled]; zeros(Float64, n)]
	correct_linear::Int64 = 0

	for i = num_labeled + 1:n
		#@assert y[i] == 0.0

		value = 0.0
		for j = 1:num_labeled
			value += y[j] * alpha[j] * kernel_matrix[i,j]
		end

		prediction = sign(value)
		labeling_linear_permuted[i] = prediction
		if prediction == instance.true_labels[i]
			correct_linear += 1
		end
	end

	accuracy_linear::Float64 = correct_linear / (n - num_labeled)
	@printf "accuracy of linear kernel: %.2f%%\n" 100.0 * accuracy_linear
	
	# rbf kernel
	kernel_matrix = compute_kernel_matrix(n, X_data, "rbf", 1.0 / d)
	K = kernel_matrix[1:num_labeled,1:num_labeled] + diagm(0.5 ./ penalty_parameters[1:num_labeled])
	#@assert issymmetric(K)
	#@assert eigmin(K) > 0.0
	Kinv = inv(K)
	Kinv = 0.5 * (Kinv + Kinv')
	C = 0.5 * Kinv
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0

	yy = y[1:num_labeled]
	_, sol = evaluate_labeling_qp(C, yy, 0, 0.0)
	
	alpha = inv(K .* (yy * yy')) * (sol .* yy)
	for i = 1:num_labeled
		#@assert alpha[i] > -1e-6
		alpha[i] = max(0.0, alpha[i])
	end

	labeling_rbf_permuted::Vector{Float64} = [y[1:num_labeled]; zeros(Float64, n)]
	correct_rbf::Int64 = 0
	for i = num_labeled + 1:n
		#@assert y[i] == 0.0

		value = 0.0
		for j = 1:num_labeled
			value += y[j] * alpha[j] * kernel_matrix[i,j]
		end

		prediction = sign(value)
		labeling_rbf_permuted[i] = prediction
		if prediction == instance.true_labels[i]
			correct_rbf += 1
		end
	end

	accuracy_rbf::Float64 = correct_rbf / (n - num_labeled)
	@printf "accuracy of rbf kernel (gamma = %.17e): %.2f%%\n\n" 1.0 / d 100.0 * accuracy_rbf

	# reorder all labels using "permutation"
	labeling_linear::Vector{Float64} = Vector{Float64}(undef, n)
	labeling_rbf::Vector{Float64} = Vector{Float64}(undef, n)
	for i = 1:n
		labeling_linear[permutation[i]] = labeling_linear_permuted[i]
		labeling_rbf[permutation[i]] = labeling_rbf_permuted[i] 
	end

	# check some labels
	for i = 1:n
		#@assert y[i] * labeling_linear[i] >= 0.0
		#@assert y[i] * labeling_rbf[i] >= 0.0
	end

	return accuracy_linear, accuracy_rbf, labeling_linear, labeling_rbf

end


function get_initial_L_and_U(
	n::Int64,
	C::Matrix{Float64},
	Cinv::Matrix{Float64},
	C_tilde::Matrix{Float64},
	y::Vector{Float64},
	upper_bound::Float64)

	#@assert n >= 1
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert size(Cinv,1) == n
	#@assert size(Cinv,2) == n
	#@assert issymmetric(Cinv)
	#@assert eigmin(Cinv) > 0.0
	#@assert norm(C * Cinv - I(n)) < 1e-8
	#@assert norm(Cinv * C - I(n)) < 1e-8
	#@assert size(C_tilde,1) == n + 1
	#@assert size(C_tilde,2) == n + 1
	#@assert issymmetric(C_tilde)
	#@assert eigmin(C_tilde) >= 0.0
	#@assert eigmin(C_tilde[1:n,1:n]) > 0.0
	#@assert length(y) == n
	for i = 1:n
		#@assert y[i] == -1.0 || y[i] == 0.0 || y[i] == +1.0
	end
	
	L = Vector{Float64}(undef, n)
	U = Vector{Float64}(undef, n)
	
	for i = 1:n
	
		if y[i] == -1.0
			L[i] = -Inf
			U[i] = -1.0
		end
		
		if y[i] == 0.0
			L[i] = -Inf
			U[i] = +Inf
		end
		
		if y[i] == +1.0
			L[i] = +1.0
			U[i] = +Inf
		end
		
	end
	
	tstart = time()

	l = count(x -> x != 0.0, y)
	rhs = 1.0 * sum(y[1:l]) * (n - l) / l
	
	compute_L_and_U!(n, C, Cinv, C_tilde, L, U, "qp", upper_bound, l, rhs)
	
	@printf "\ntime for solving QPs : %.2f seconds\n" time() - tstart
	
	@printf "initial volume : %.4f\n\n" sum(U - L)
	
	return L, U

end


# sense == 0: minimization / L[i] is computed
# sense == 1: maximization / U[i] is computed
function compute_bound_constraint_qp!(
	n::Int64,
	C::Matrix{Float64},
	Cinv::Matrix{Float64},
	L::Vector{Float64},
	U::Vector{Float64},
	upper_bound::Float64,
	sense::Int64,
	index::Int64,
	l::Int64,
	rhs::Float64)
	
	#@assert n >= 1
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert length(L) == n
	#@assert length(U) == n
	#@assert upper_bound > 0.0
	#@assert TOL_BOUND > 0.0
	for i = 1:n
		#@assert L[i] < U[i] + TOL_BOUND
	end
	#@assert sense == 0 || sense == 1
	#@assert 1 <= index && index <= n
	
	# do not compute a bound when the result is already known
	if sense == 0
		#@assert L[index] < 0.0 # otherwise, the lower bound will be +1.0
	else
		#@assert U[index] > 0.0 # otherwise, the lower bound will be -1.0
	end
	
	model = Model(() -> Gurobi.Optimizer(gurobi_env); add_bridges = false)
	#model = Model(Mosek.Optimizer)
	#model = Model(ECOS.Optimizer)
	#model = Model(SCS.Optimizer)

	set_silent(model)

	# Gurobi parameters
	set_optimizer_attribute(model, "BarQCPConvTol", 1e-8)
	set_optimizer_attribute(model, "Threads", 1)
	set_optimizer_attribute(model, "QCPDual", 1)

	# Mosek parameters
	#set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
	
	@variable(model, L[i] <= x_var[i = 1:n] <= U[i])

	@constraint(model, sum(x_var[i] for i in l+1:n) == rhs)

	# bound objective value
	bound_objective = @constraint(model, x_var' * C * x_var <= upper_bound)

	# avoid degeneracy: make sure that the constraint x' * C * x <= UB always is active at the optimum
	if sense == 0 && has_lower_bound(x_var[index])
		delete_lower_bound(x_var[index])
		#@assert !has_lower_bound(x_var[index])
	elseif has_upper_bound(x_var[index])
		delete_upper_bound(x_var[index])
		#@assert !has_upper_bound(x_var[index])
	end

	# set objective
	if sense == 0
		@objective(model, Min, x_var[index])
	else
		@objective(model, Max, x_var[index])
	end

	optimize!(model)
	
	if termination_status(model) != MOI.OPTIMAL
		@printf "in compute_bound_constraint_qp!():\n"
		@show termination_status(model)
	end

	##@assert termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.SLOW_PROGRESS

	x::Vector{Float64} = value.(x_var)
	#@assert length(x) == n
	for i = 1:n
		#@assert i == index || x[i] >= L[i] - 1e-5
		#@assert i == index || x[i] <= U[i] + 1e-5
		#@assert i == index || (has_lower_bound(x_var[i]) && L[i] > -Inf) || (!has_lower_bound(x_var[i]) && L[i] == -Inf)
		#@assert i == index || (has_upper_bound(x_var[i]) && U[i] < +Inf) || (!has_upper_bound(x_var[i]) && U[i] == +Inf)
		if has_lower_bound(x_var[i])
			#@assert dual(LowerBoundRef(x_var[i])) >= -1e-5
		end
		if has_upper_bound(x_var[i])
			#@assert -dual(UpperBoundRef(x_var[i])) >= -1e-5
		end
	end

	signum::Float64 = (sense == 0) ? -1.0 : 1.0 # needed to construct a dual feasible solution

	p::Vector{Float64} = zeros(Float64,n)
	dual_bound_objective::Float64 = -dual(bound_objective)
	#@assert dual_bound_objective >= -1e-5
	dual_bound_objective = max(0.0, dual_bound_objective)
	r::Float64 = signum * dual_bound_objective * upper_bound

	# the dual variable corresponding to x^T * C * x <= UB must actually be positive!
	#@assert dual_bound_objective > 1e-3

	p[index] = 1.0
	for i = 1:n
		if has_lower_bound(x_var[i])
			value::Float64 = dual(LowerBoundRef(x_var[i]))
			#@assert value >= -1e-5
			value = max(0.0, value)
			p[i] += signum * value
			r -= signum * L[i] * value
		end
		if has_upper_bound(x_var[i])
			value = -dual(UpperBoundRef(x_var[i]))
			#@assert value >= -1e-5
			value = max(0.0, value)
			p[i] -= signum * value
			r += signum * U[i] * value
		end
	end

	# TODO balancing constraint needs to be added to "my_dual_objective_value"
	# my_dual_objective_value::Float64 = signum * 0.25 / dual_bound_objective * p' * Cinv * p + r

	# if abs(my_dual_objective_value - objective_value(model)) / (1.0 + abs(my_dual_objective_value) + abs(objective_value(model))) >= 1e-3
	# 	@printf "WARNING: QCQP solver might have failed during bound constraint computation.\n"
	# 	@show objective_value(model)
	# 	@show my_dual_objective_value
	# 	@show dual_bound_objective
	# end

	return objective_value(model)
    
end


# this function is called as part of the preprocessing or whenever the global upper bound is updated
# TODO IMPORTANT: if this causes problems, then it is probably because updating L[i] and U[i] is not atomic!
function compute_L_and_U!(
	n::Int64,
	C::Matrix{Float64},
	Cinv::Matrix{Float64},
	C_tilde::Matrix{Float64},
	L::Vector{Float64},
	U::Vector{Float64},
	method::String,
	upper_bound::Float64,
	l::Int64,
	rhs::Float64)
	
	#@assert n >= 1
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert size(Cinv,1) == n
	#@assert size(Cinv,2) == n
	#@assert issymmetric(Cinv)
	#@assert eigmin(Cinv) > 0.0
	#@assert norm(C * Cinv - I(n)) < 1e-8
	#@assert norm(Cinv * C - I(n)) < 1e-8
	#@assert size(C_tilde,1) == n + 1
	#@assert size(C_tilde,2) == n + 1
	#@assert issymmetric(C_tilde)
	#@assert eigmin(C_tilde) >= 0.0
	#@assert eigmin(C_tilde[1:n,1:n]) > 0.0
	#@assert length(L) == n
	#@assert length(U) == n
	#@assert TOL_BOUND > 0.0
	for i = 1:n
		#@assert L[i] < U[i] + TOL_BOUND
	end
	#@assert method == "qp" || method == "sdp" || method == "sdp_rlt"

	L_copy::Vector{Float64} = deepcopy(L)
	U_copy::Vector{Float64} = deepcopy(U)
	
	#for loop_index::Int64 in 1:2 * n
	Threads.@threads for loop_index::Int64 in 1:2 * n
		
		i::Int64 = div(loop_index + 1, 2)
		
		#if mod(i, 10) == 0
		#	# call garbage collector
		#	GC.gc(true); GC.gc(false)
		#end
		
		sense::Int64 = mod(loop_index, 2)
		#@assert sense == 0 || sense == 1
			
		if sense == 0 && L[i] >= 1.0 || sense == 1 && U[i] <= -1.0
			continue # skip bound computation
		end
		
		if sense == 0
		
			if method == "qp"
				L[i] = max(L[i], - TOL_BOUND + compute_bound_constraint_qp!(n, C, Cinv, L_copy, U_copy, upper_bound, sense, i, l, rhs))
			#elseif method == "sdp"
			#	L[i] = max(L[i], - TOL_BOUND + compute_bound_constraint_sdp!(n, C_tilde, L_copy, U_copy, upper_bound, sense, i))
			#elseif method == "sdp_rlt"
			#	L[i] = max(L[i], - TOL_BOUND + compute_bound_constraint_sdp_rlt!(n, C_tilde, L_copy, U_copy, upper_bound, sense, i))
			else
				#@assert false
			end
			
	        if L[i] > -1.0 + TOL_BOUND
		        L[i] = max(L[i], 1.0)
		    end
		    
		else
		
			if method == "qp"
				U[i] = min(U[i], TOL_BOUND + compute_bound_constraint_qp!(n, C, Cinv, L_copy, U_copy, upper_bound, sense, i, l, rhs))
			#elseif method == "sdp"
			#	U[i] = min(U[i], TOL_BOUND + compute_bound_constraint_sdp!(n, C_tilde, L_copy, U_copy, upper_bound, sense, i))
			#elseif method == "sdp_rlt"
			#	U[i] = min(U[i], TOL_BOUND + compute_bound_constraint_sdp_rlt!(n, C_tilde, L_copy, U_copy, upper_bound, sense, i))
			#else
				#@assert false
			end
			
	        if U[i] < 1.0 - TOL_BOUND
		        U[i] = min(U[i], -1.0)
		    end
		    
		end
			
	end # loop_index
	
	for i = 1:n
		#@assert L[i] < U[i] + TOL_BOUND
	end
		
end


function solve_basic_sdp_relaxation(n::Int64, C_tilde::Matrix{Float64}, y::Vector{Float64})

	#@assert n >= 1
	#@assert size(C_tilde,1) == n + 1
	#@assert size(C_tilde,2) == n + 1
	#@assert length(y) == n
	#@assert issymmetric(C_tilde)
	#@assert eigmin(C_tilde) >= 0.0
	#@assert eigmin(C_tilde[1:n,1:n]) > 0.0
	for i = 1:n
		#@assert y[i] == -1.0 || y[i] == 0.0 || y[i] == +1.0
	end
	
    model = Model(Mosek.Optimizer; add_bridges = false)
	set_silent(model)
        
    @variable(model, X_var[1:n+1, 1:n+1], PSD)
    
    for i = 1:n
    	if y[i] != 0.0
    		@constraint(model, y[i] * X_var[n+1,i] >= 1.0)
    	else
    		@constraint(model, X_var[i,i] >= 1.0)
    	end
    end

    @constraint(model, X_var[n+1,n+1] == 1.0)
    
    @objective(model, Min, LinearAlgebra.dot(C_tilde, X_var))
    
	optimize!(model)

	#@assert termination_status(model) == MOI.OPTIMAL
	# TODO we could add more checks here but we this is not needed
	
	X = value.(X_var)
	
    return X[n+1, 1:n], X[1:n, 1:n]

end


function complicated_branching_decision(
	C::Matrix{Float64},
	L::Vector{Float64},
	U::Vector{Float64},
	x::Vector{Float64},
	X::Matrix{Float64},
	duals::Vector{Float64},
	v::Vector{Float64},
	Z::Matrix{Float64},
	rlt_sum::Vector{Float64},
	l::Int64,
	rhs::Float64)

	n::Int64 = length(x)
	#@assert (n,n) == size(C)
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert length(L) == n
	#@assert length(U) == n
	for i = 1:n
		#@assert L[i] < U[i] + TOL_BOUND
	end
	#@assert length(x) == n
	#@assert (n,n) == size(X)
	#@assert issymmetric(X)
	#@assert eigmin(X) > -1e-8
	#@assert length(duals) == n
	#@assert minimum(duals) >= 0.0
	#@assert length(v) == n
	#@assert minimum(abs.(v)) >= 1.0
	#@assert (n+1,n+1) == size(Z)
	#@assert issymmetric(Z)
	#@assert eigmin(Z) > -1e-8
	#@assert length(rlt_sum) == n
	#@assert minimum(rlt_sum) >= 0.0

	y::Vector{Float64} = sign.(x)
	_, best_sol::Vector{Float64} = evaluate_labeling_qp(C, y, l, rhs)
	#@assert length(best_sol) == n
	#@assert minimum(abs.(best_sol)) >= 1.0

	branching_decisions::Vector{Branching_Decision} = Branching_Decision[]
	
	for i = 1:n

		if L[i] < 0.0 && U[i] > 0.0 && abs(duals[i]) > 1e-4 && abs(abs(best_sol[i]) - 1.0) < 1e-4 # unlabeled data point and X_ii >= 1 is active and could be a support vector
		
			error1 = 0.0
			error2 = 0.0
			error3 = 0.0
			error4 = 0.0
			for j = 1:n
				error1 += C[i,j] * (x[i] * x[j] - X[i,j])
				error2 += x[i] * x[j] - X[i,j]
				error3 += abs(C[i,j] * (x[i] * x[j] - X[i,j]))
				error4 += abs(x[i] * x[j] - X[i,j])
			end
			
			contribution_sdp = 0.0
			contribution_best = 0.0
			for j = 1:n
				contribution_sdp += C[i,j] * X[i,j]
				contribution_best += C[i,j] * v[i] * v[j]
			end
					
			decision = Branching_Decision(
				i, # index::Int64
				abs(x[i]), # abs_x::Float64
				abs(x[i] - v[i]), # diff_x_best::Float64
				rlt_sum[i], # rlt_sum::Float64
				sum(abs.(Z[i,:])), # sum_Zi::Float64
				sum(abs.(C[i,:])), # sum_Ci::Float64
				abs(error1), # error1::Float64
				abs(error2), # error2::Float64
				abs(error3), # error3::Float64
				abs(error4), # error4::Float64
				abs(contribution_best - contribution_sdp) # objective_diff::Float64
			)

			#branching_decisions[i] = decision
			push!(branching_decisions, decision)
			
		end
		
	end # i loop
	
	scores = Vector{Int64}(undef, n)
	for i = 1:n
		scores[i] = 0
	end
	
	# sort ascending with respect to "abs_x"
	sort!(branching_decisions, by = v -> v.abs_x, rev=false)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "diff_x_best"
	sort!(branching_decisions, by = v -> v.diff_x_best, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "rlt_sum"
	sort!(branching_decisions, by = v -> v.rlt_sum, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "sum_Zi"
	sort!(branching_decisions, by = v -> v.sum_Zi, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "sum_Ci"
	sort!(branching_decisions, by = v -> v.sum_Ci, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "error1"
	sort!(branching_decisions, by = v -> v.error1, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "error2"
	sort!(branching_decisions, by = v -> v.error2, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "error3"
	sort!(branching_decisions, by = v -> v.error3, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "error4"
	sort!(branching_decisions, by = v -> v.error4, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	# sort descending with respect to "objective_diff"
	sort!(branching_decisions, by = v -> v.objective_diff, rev=true)
	for pos = 1:length(branching_decisions)
		scores[branching_decisions[pos].index] += pos
	end
	
	best_score = Int64(typemax(Int64))
	branching_variable = Int64(0)
	
	for i = 1:n
		if scores[i] != 0 && scores[i] < best_score
			best_score = scores[i]
			branching_variable = i
		end
	end
	
	if branching_variable == 0
	
		# choose most fractional variable as backup decision
		most_frac = + Inf
		for i = 1:n
			if L[i] < 0.0 && U[i] > 0.0 && abs(x[i]) < most_frac
				branching_variable = i
				most_frac = abs(x[i])
			end
		end
	
	end

	return branching_variable

end


function get_branching_variable(
	n::Int64,
	C::Matrix{Float64},
	L::Vector{Float64},
	U::Vector{Float64},
	x::Vector{Float64},
	X::Matrix{Float64},
	duals::Vector{Float64},
	v::Vector{Float64},
	Z::Matrix{Float64},
	rlt_sum::Vector{Float64},
	branching_rule::Int64,
	branching_epsilon::Float64,
	l::Int64,
	rhs::Float64)

	#@assert n >= 1
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	length(L) == n
	length(U) == n
	length(x) == n
	#@assert size(X,1) == n
	#@assert size(X,2) == n
	#@assert issymmetric(X)
	#@assert eigmin(X) > -1e-8
	#@assert length(duals) == n
	#@assert minimum(duals) >= 0.0
	#@assert length(v) == n
	#@assert minimum(abs.(v)) >= 1.0
	#@assert size(Z,1) == n + 1
	#@assert size(Z,2) == n + 1
	#@assert issymmetric(Z)
	#@assert eigmin(Z) > -1e-8
	#@assert length(rlt_sum) == n
	#@assert minimum(rlt_sum) >= 0.0
	for i = 1:n
		#@assert L[i] < U[i] + TOL_BOUND # otherwise, the subproblem would be infeasible
		#@assert L[i] < 0.0 || v[i] > 0.0 # "v" must be a feasible solution to this subproblem
		#@assert U[i] > 0.0 || v[i] < 0.0 # "v" must be a feasible solution to this subproblem
	end
	#@assert 0 <= branching_rule && branching_rule <= 3
	#@assert branching_epsilon > 0.0
	#@assert compute_number_of_labeled_data_points(L, U) < n # at least one data point must be unlabeled

	branching_variable::Int64 = 0
	
	# choose default/backup branching decision (most fractional / branching_rule == 0)
	most_frac::Float64 = +Inf
	for i = 1:n
		if L[i] < 0.0 && U[i] > 0.0 && abs(x[i]) < most_frac
			branching_variable = i
			most_frac = abs(x[i])
		end
	end

	# if possible, choose a better default/backup branching decision where the constraint X_ii >= 1 is active
	if branching_rule > 0
		most_frac = +Inf
		for i = 1:n
			if L[i] < 0.0 && U[i] > 0.0 && abs(x[i]) < most_frac && duals[i] > 1e-5
				branching_variable = i
				most_frac = abs(x[i])
			end
		end
	end

	# largest approximation error
	if branching_rule == 1

		max_error::Float64 = -Inf
		
		for i = 1:n
		
			approx_error = 0.0
			for j = 1:n
				approx_error += abs(C[i,j] * (x[i] * x[j] - X[i,j]))
			end
		
			if approx_error > max_error && L[i] < 0.0 && U[i] > 0.0 && duals[i] > 1e-5 && abs(x[i]) < branching_epsilon
				max_error = approx_error
				branching_variable = i
			end
		
		end
	end

	if branching_rule == 2
		branching_variable = complicated_branching_decision(C, L, U, x, X, duals, v, Z, rlt_sum, l, rhs)
	end

	# box rule
	if branching_rule == 3

		y::Vector{Float64} = sign.(x)
		_, best_sol::Vector{Float64} = evaluate_labeling_qp(C, y, l, rhs)
		#@assert length(best_sol) == n
		#@assert minimum(abs.(best_sol)) >= 1.0

		max_box::Float64 = 0.0

		for i = 1:n
			if L[i] < 0.0 && U[i] > 0.0 && duals[i] > 1e-4 && abs(x[i]) < branching_epsilon && abs(abs(best_sol[i]) - 1.0) < 1e-4
				value::Float64 = min(1.0 - L[i], 1.0 + U[i])
				if value > max_box
					max_box = value
					branching_variable = i
				end
			end
		end

	end

	#@assert 1 <= branching_variable && branching_variable <= n
	#@assert L[branching_variable] < 0.0 && U[branching_variable] > 0.0
	
	return branching_variable
end



function compute_gap(lower_bound, upper_bound)
	#@assert lower_bound >= 0.0
	#@assert 0.0 < upper_bound && upper_bound < +Inf
	return (upper_bound - lower_bound) / upper_bound
end



function can_be_pruned(subproblem::Subproblem, upper_bound::Float64)
	gap = compute_gap(subproblem.lower_bound[], upper_bound)
	return (gap < GAP_PRUNE) ? true : false
end



function check_s3vm_instance(instance::S3vm_instance, before_bb::Bool)

	#@assert instance.n >= 1
	#@assert instance.d >= 1
	#@assert size(instance.X_data,1) == instance.n
	#@assert size(instance.X_data,2) == instance.d
	for i = 1:instance.d
		#@assert abs(mean(instance.X_data[:,i])) < 1e-8
		#@assert abs(std(instance.X_data[:,i]; corrected = false) - 1.0) < 1e-8 || abs(std(instance.X_data[:,i]; corrected = true) - 1.0) < 1e-8
	end
	#@assert length(instance.y) == instance.n
	#@assert length(instance.true_labels) == instance.n
	#@assert length(instance.penalty_parameters) == instance.n
	for i = 1:instance.n
		#@assert instance.y[i] == -1.0 || instance.y[i] == 0.0 || instance.y[i] == +1.0
		#@assert instance.true_labels[i] == -1.0 || instance.true_labels[i] == +1.0
		#@assert instance.penalty_parameters[i] > 0.0
		#@assert instance.y[i] == instance.true_labels[i] || instance.y[i] == 0.0
	end
	#@assert size(instance.kernel_matrix,1) == instance.n
	#@assert size(instance.kernel_matrix,2) == instance.n
	#@assert issymmetric(instance.kernel_matrix)
	#@assert eigmin(instance.kernel_matrix) > -1e-8
	#@assert size(instance.D,1) == instance.n
	#@assert size(instance.D,2) == instance.n
	#@assert issymmetric(instance.D)
	#@assert norm(instance.D - diagm(0.5 ./ instance.penalty_parameters)) < 1e-8
	#@assert size(instance.K,1) == instance.n
	#@assert size(instance.K,2) == instance.n
	#@assert issymmetric(instance.K)
	#@assert eigmin(instance.K) > 0.0
	#@assert norm(instance.K - (instance.kernel_matrix + instance.D)) < 1e-8
	#@assert size(instance.Kinv,1) == instance.n
	#@assert size(instance.Kinv,2) == instance.n
	#@assert issymmetric(instance.Kinv)
	#@assert eigmin(instance.Kinv) > 0.0
	#@assert norm(instance.Kinv - inv(instance.K)) < 1e-6
	#@assert norm(instance.K * instance.Kinv - I(instance.n)) < 1e-6
	#@assert size(instance.C,1) == instance.n
	#@assert size(instance.C,2) == instance.n
	#@assert issymmetric(instance.C)
	#@assert eigmin(instance.C) > 0.0
	#@assert norm(instance.C - 0.5 * instance.Kinv) < 1e-6
	#@assert issymmetric(instance.Cinv)
	#@assert eigmin(instance.Cinv) > 0.0
	#@assert norm(instance.C * instance.Cinv - I(instance.n)) < 1e-6
	#@assert norm(instance.Cinv - inv(instance.C)) < 1e-6
	#@assert norm(instance.Cinv .- 2.0 .* instance.K) < 1e-6
	#@assert size(instance.C_tilde,1) == instance.n + 1
	#@assert size(instance.C_tilde,2) == instance.n + 1
	#@assert issymmetric(instance.C_tilde)
	#@assert eigmin(instance.C_tilde) >= 0.0
	#@assert norm(instance.C_tilde - [0.5 * instance.Kinv zeros(instance.n,1); zeros(1,instance.n) 0]) < 1e-8
	#@assert instance.kernel == "linear" || instance.kernel == "rbf"
	#@assert instance.kernel == "rbf" && instance.gamma > 0.0 || instance.kernel == "linear"
	#@assert norm(instance.kernel_matrix - compute_kernel_matrix(instance.n, instance.X_data, instance.kernel, instance.gamma)) < 1e-8

	# these things will change during B&B but should be initialized with specific values
	if before_bb == true
		#@assert instance.upper_bound[] == +Inf
		#@assert instance.x == zeros(Float64,instance.n)
		#@assert instance.labeling == zeros(Float64,instance.n)
		#@assert instance.predictions == zeros(Float64,instance.n)
	else
		#@assert 0.0 < instance.upper_bound[] && instance.upper_bound[] < +Inf
		#@assert length(instance.x) == instance.n
		#@assert length(instance.labeling) == instance.n
		#@assert length(instance.predictions) == instance.n
		#@assert minimum(abs.(instance.x)) >= 1.0
		#@assert abs(instance.upper_bound[] - instance.x' * instance.C * instance.x) < 1e-8
		#@assert minimum(abs.(instance.labeling)) == 1.0
		#@assert maximum(abs.(instance.labeling)) == 1.0
		#@assert minimum(abs.(instance.predictions)) == 1.0
		#@assert maximum(abs.(instance.predictions)) == 1.0
		#@assert sign.(instance.x) == instance.labeling
		for i = 1:instance.n
			#@assert instance.y[i] * instance.x[i] >= 0.0
			#@assert instance.y[i] * instance.labeling[i] >= 0.0
			#@assert instance.y[i] == 0.0 || instance.y[i] == instance.labeling[i]
		end
	end

end



function solve_s3vm_instance(instance::S3vm_instance, parameters::MyParameters)

	check_s3vm_instance(instance, true)

	@printf "\n\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
	@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"

	n::Int64 = instance.n
	d::Int64 = instance.d

	l::Int64 = count(x -> x != 0.0, instance.y)
	rhs::Float64 = 1.0 * sum(instance.y[1:l]) * (n - l) / l

	for i = 1:l
		#@assert instance.y[i] != 0.0
	end

	@printf "\nWelcome to our S3VM solver!\n"
	@printf "Please note that the data must be provided in standardized format.\n\n"

	@printf "file/instance: %s\n" basename(instance.file)
	@printf "number of data points: %d\n" n
	@printf "number of features: %d\n" d
	@printf "range of penalty parameters: [%.17e, %.17e]\n" minimum(instance.penalty_parameters) maximum(instance.penalty_parameters)
	@printf "kernel: %s" instance.kernel
	if instance.kernel == "rbf"
		@printf " with gamma = %.17e" instance.gamma
	end
	@printf "\n"

	num_unlabeled::Int64 = sum(instance.y .== 0.0)
	num_labeled::Int64 = n - num_unlabeled

	@printf "number of labeled data points: %d (%.2f%%)\n" num_labeled 100.0 * num_labeled / n
	@printf "number of unlabeled data points : %d (%.2f%%)\n" num_unlabeled 100.0 * num_unlabeled / n

	num_label_minus::Int64 = sum(instance.y .== -1.0)
	num_label_plus::Int64 = sum(instance.y .== +1.0)

	@printf "number of given labeled data points with label \"-1\": %d (%.2f%%)\n" num_label_minus 100.0 * num_label_minus / num_labeled
	@printf "number of given labeled data points have label \"+1\": %d (%.2f%%)\n\n" num_label_plus 100.0 * num_label_plus / num_labeled

	@printf "Here are the accuracies on all unlabeled data points using only labeled data points to build an SVM (with the same penalty parameters):\n"
	_, _, labeling_linear::Vector{Float64}, labeling_rbf::Vector{Float64} = compute_and_print_accuracy_using_labeled_points(instance)
	#@assert minimum(abs.(labeling_linear)) == 1.0
	#@assert maximum(abs.(labeling_linear)) == 1.0
	for i = 1:n
		#@assert instance.y[i] * labeling_linear[i] >= 0.0
		#@assert instance.y[i] * labeling_rbf[i] >= 0.0
	end

	if instance.kernel == "linear"
		heuristic_labeling(instance, parameters.use_local_search, labeling_linear)
	end

	if instance.kernel == "rbf"
		heuristic_labeling(instance, parameters.use_local_search, labeling_rbf)
	end

	tstart::Float64 = time()

	x::Vector{Float64}, X::Matrix{Float64} = solve_basic_sdp_relaxation(n, instance.C_tilde, instance.y)
	run_heuristic(instance.C, x, X, instance.y, nothing, instance, 1, parameters.use_local_search)

	L::Vector{Float64}, U::Vector{Float64} = get_initial_L_and_U(n, instance.C, instance.Cinv, instance.C_tilde, instance.y, instance.upper_bound[])
	#@assert length(L) == n
	#@assert length(U) == n
	
	for i = 1:n
		#@assert isfinite(L[i])
		#@assert isfinite(U[i])
		#@assert L[i] < U[i] + TOL_BOUND
	end
	
	volume_before_bb::Float64 = sum(U - L)
    
	# create root subproblem
	root_node::Subproblem = Subproblem(Threads.Atomic{Float64}(0.0), 0, 0, deepcopy(L), deepcopy(U), zeros(Float64, n), +Inf, RLT_Cut[])
	number_of_created_subproblems::Int64 = 1
	number_of_unexplored_nodes::Int64 = 1
	number_of_explored_nodes::Int64 = 0
	
	# initialize priority queue with root subproblem only
	pq = PriorityQueue(Base.Order.Forward, root_node => 0.0)

	print_output::Int64 = 0

	global_lower_bound::Float64 = 0.0

	output_string::String = instance.file
	output_string = @sprintf "%s & %d" output_string num_labeled
	output_string = @sprintf "%s & %d" output_string num_unlabeled
	output_string = @sprintf "%s & %s" output_string instance.kernel
	if instance.kernel == "rbf"
		output_string = @sprintf "%s, gamma = %g" output_string instance.gamma
	end
	output_string = @sprintf "%s & %g" output_string maximum(instance.penalty_parameters)
	output_string = @sprintf "%s & %g" output_string minimum(instance.penalty_parameters)

	max_threads::Int64 = min(Threads.nthreads(), parameters.num_threads)
	if parameters.num_threads == 0
		max_threads = Threads.nthreads()
	end
	@show Threads.nthreads()
	@show parameters.num_threads
	@printf "We use %d threads for parallel branch-and-bound\n\n" max_threads

	if parameters.num_threads == 1
		@printf "Since the value \"1\" was provided for \"num_threads\", Mosek will use its default number of threads for each SDP.\n\n"
	end

	channel::Channel{ReturnValue} = Channel{ReturnValue}(max_threads)

	threads_running::Int64 = 0

	active_subproblems::Vector{Union{Subproblem,Nothing}} = Vector{Union{Subproblem,Nothing}}(undef, max_threads)
	for i = 1:max_threads
		active_subproblems[i] = nothing
	end

	# we are only done if the priority queue is empty AND no other thread is still running (more subproblems could be created)
	while !isempty(pq) && number_of_explored_nodes + threads_running < parameters.max_bb_nodes && (time() - tstart < parameters.time_limit || number_of_explored_nodes == 0) || threads_running > 0

		# there must be at least one idle thread at this point in the program
        # if all threads a are running, then the master thread must be blocked due to the "take!(channel)" call
        #@assert 0 <= threads_running && threads_running < max_threads

		#@assert number_of_created_subproblems == number_of_explored_nodes + number_of_unexplored_nodes

		# start new tasks on idle threads if possible
		while !isempty(pq) && threads_running < max_threads && number_of_explored_nodes + threads_running < parameters.max_bb_nodes && (time() - tstart < parameters.time_limit || number_of_explored_nodes == 0) 
			subproblem::Subproblem = dequeue!(pq)
			threads::Int64 = (parameters.num_threads == 1) ? 0 : Int64(max(1, floor(max_threads / (1.0 + subproblem.depth))))
			Threads.@spawn evaluate_subproblem(subproblem, instance, parameters, threads, channel)
			for i = 1:max_threads
				#@assert i < max_threads || isnothing(active_subproblems[max_threads])
				if isnothing(active_subproblems[i])
					active_subproblems[i] = subproblem
					break
				end
			end
			threads_running += 1
			#@assert 1 <= threads_running && threads_running <= max_threads && threads_running <= Threads.nthreads()
			sleep(0.001)
		end

		# since we cannot start more threads (because there are no subproblems left or all threads are already running), we wait for a thread to put! its result into the channel
		ret::ReturnValue = take!(channel)
		#@assert 1 <= threads_running && threads_running <= max_threads
		threads_running -= 1 # exactly one thread has finished its work
		#@assert !isready(channel) || threads_running > 0

		lower_bound::Float64 = ret.lower_bound
		x = ret.x
		X = ret.X
		iter::Int64 = ret.iter
		cutting_planes::Int64 = ret.cutting_planes
		local_upper_bound::Float64 = ret.local_upper_bound
		volume_end::Float64 = ret.volume_end
		volume_diff::Float64 = ret.volume_diff
		duals::Vector{Float64} = ret.duals
		Z::Matrix{Float64} = ret.Z
		rlt_sum::Vector{Float64} = ret.rlt_sum
		final_cuts::Vector{RLT_Cut} = ret.final_cuts
		time_spent::Float64 = ret.time_spent
		subproblem = ret.subproblem

		#@assert subproblem.lower_bound[] == lower_bound
		#@assert global_lower_bound <= lower_bound
		#@assert length(x) == n
		#@assert (n, n) == size(X)
		#@assert issymmetric(X)
		#@assert eigmin(X) > -1e-8
		#@assert iter >= 0
		#@assert cutting_planes >= 0
		#@assert lower_bound <= local_upper_bound
		#@assert volume_diff >= 0.0
		#@assert length(duals) == n
		#@assert minimum(duals) >= 0.0
		#@assert (n + 1, n + 1) == size(Z)
		#@assert issymmetric(Z)
		#@assert eigmin(Z) > -1e-8
		#@assert length(rlt_sum) == n
		#@assert minimum(rlt_sum) >= 0.0
		#@assert time_spent > 0.0

		# compute new global lower bound
		global_lower_bound = lower_bound
		for i = 1:max_threads
			if !isnothing(active_subproblems[i])
				global_lower_bound = min(global_lower_bound, active_subproblems[i].lower_bound[])
			end
			if subproblem === active_subproblems[i]
				active_subproblems[i] = nothing
			end
		end
		if !isempty(pq)
			(subproblem_peek::Subproblem, _) = peek(pq)
			global_lower_bound = min(global_lower_bound, subproblem_peek.lower_bound[])
		end

		if global_lower_bound > (1.0 - parameters.gap_prune) * instance.upper_bound[]
			global_lower_bound = (1.0 - parameters.gap_prune) * instance.upper_bound[]
		end

		#@assert subproblem.lower_bound[] <= lower_bound
		#@assert global_lower_bound <= subproblem.lower_bound[]
		#@assert lower_bound <= local_upper_bound
		#@assert minimum(duals) >= 0.0
		#@assert minimum(rlt_sum) >= 0.0

		if subproblem.depth == 0
			output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * compute_gap(lower_bound, instance.upper_bound[])
		end

		number_of_unlabeled_points::Int64 = n - compute_number_of_labeled_data_points(subproblem.L, subproblem.U) 

		number_of_explored_nodes += 1
		number_of_unexplored_nodes -= 1
		
		#@assert number_of_unexplored_nodes >= 0
		#@assert number_of_created_subproblems == number_of_explored_nodes + number_of_unexplored_nodes
				
		gap = compute_gap(lower_bound, instance.upper_bound[])
		
		branching_variable::Int64 = 0
		depth::Int64 = subproblem.depth

		if gap >= parameters.gap_prune

			#@assert number_of_unlabeled_points > 0

			# branching

			#@assert compute_number_of_labeled_data_points(subproblem.L, subproblem.U) != n

			branching_variable = get_branching_variable(n, instance.C, subproblem.L, subproblem.U, x, X, duals, subproblem.x, Z, rlt_sum, parameters.branching_rule, parameters.branching_epsilon, l, rhs)

			#@assert 1 <= branching_variable && branching_variable <= n
			#@assert L[branching_variable] < 0.0 && U[branching_variable] > 0.0

			# label "-1"
			child1::Subproblem = Subproblem(Threads.Atomic{Float64}(lower_bound), number_of_created_subproblems, depth + 1, deepcopy(subproblem.L), deepcopy(subproblem.U), zeros(n), +Inf, deepcopy(final_cuts))
			child1.U[branching_variable] = -1.0
			enqueue!(pq, child1, lower_bound)
			
			number_of_unexplored_nodes += 1
			number_of_created_subproblems += 1

			# label "+1"
			child2::Subproblem = Subproblem(Threads.Atomic{Float64}(lower_bound), number_of_created_subproblems, depth + 1, deepcopy(subproblem.L), deepcopy(subproblem.U), zeros(n), +Inf, deepcopy(final_cuts))
			child2.L[branching_variable] = +1.0
			enqueue!(pq, child2, lower_bound)
			
			number_of_unexplored_nodes += 1
			number_of_created_subproblems += 1

		end # branching

		# call garbage collector
		if mod(number_of_explored_nodes, 15) == 0
			GC.gc(true); GC.gc(false)
		end
		
		if mod(print_output, 20) == 0
			@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
			@printf "|%11s|%7s|%7s|%9s|%5s|%4s|%5s|%4s|%5s|%11s|%9s|%7s|%16s|%16s|%10s|%16s|%16s|%10s|\n" "B&B time" "expl." "unexpl." "node time" "depth" "iter" "cuts" "var" "unlab" "volume" "diff" "rel" "local LB" "local UB" "local gap" "global LB" "global UB" "global gap"
			@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
		end
		print_output += 1
		
		@printf "|%11.1f|%7d|%7d|%9.1f|%5d|%4d|%5d|%4d|%5d|%11.2f|%9.2f|%6.2f%%|%16.8f|%16.8f|%9.4f%%|%16.8f|%16.8f|%9.4f%%|\n" time() - tstart number_of_explored_nodes number_of_unexplored_nodes time_spent depth iter cutting_planes branching_variable number_of_unlabeled_points volume_end volume_diff 100.0 * volume_end / volume_before_bb lower_bound local_upper_bound 100.0 * compute_gap(lower_bound, instance.upper_bound[]) global_lower_bound instance.upper_bound[] 100.0 * max(parameters.gap_prune, compute_gap(global_lower_bound, instance.upper_bound[]))

	end # branch-and-bound loop

	close(channel)

	check_s3vm_instance(instance, false)
	
	@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
	
	#@assert number_of_created_subproblems == number_of_explored_nodes || number_of_explored_nodes >= parameters.max_bb_nodes || time() - tstart >= parameters.time_limit
	#@assert number_of_unexplored_nodes == 0 || number_of_explored_nodes >= parameters.max_bb_nodes || time() - tstart >= parameters.time_limit

	gap = parameters.gap_prune
	if !isempty(pq)
		(subproblem, _) = peek(pq)
		global_lower_bound = subproblem.lower_bound[]
		gap = compute_gap(global_lower_bound, instance.upper_bound[])
	end

	@printf "\nB&B time : %.2f seconds\n" time() - tstart
	@printf "explored nodes : %d\n" number_of_explored_nodes
	@printf "best objective value found: %.17e\n" instance.upper_bound[]
	@printf "relative gap : %.4f%%\n" 100.0 * gap
	
	accuracy_labeled_direct, accuracy_labeled_prediction, accuracy_unlabeled_direct, accuracy_unlabeled_prediction = print_accuracy(instance)
	#@assert accuracy_labeled_direct == 1.0


	@printf "\nAgain: here are the accuracies on all unlabeled data points using only labeled data points to build an SVM (with the same penalty parameters):\n"
	accuracy_linear, accuracy_rbf = compute_and_print_accuracy_using_labeled_points(instance)
	
	@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
	@printf "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
	
	output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * gap
	output_string = @sprintf "%s & %d" output_string number_of_explored_nodes
	output_string = @sprintf "%s & %.2fs" output_string time() - tstart
	output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * accuracy_labeled_prediction
	output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * accuracy_unlabeled_direct
	output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * accuracy_unlabeled_prediction
	if instance.kernel == "linear"
		output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * accuracy_linear
	else
		output_string = @sprintf "%s & %.2f\\%%" output_string 100.0 * accuracy_rbf
	end

	output_string = @sprintf "%s %s" output_string "\\\\"
	open("./output/bb_output_file.txt", "a") do file
		println(file, output_string)
		close(file)
	end

end # solve_s3vm_instance()




function evaluate_subproblem(subproblem::Subproblem, instance::S3vm_instance, parameters::MyParameters, max_threads::Int64, channel::Channel{ReturnValue})

	tstart_subproblem::Float64 = time()

	n::Int64 = instance.n

	gap::Float64 = compute_gap(subproblem.lower_bound[], instance.upper_bound[])

	# check whether global_upper_bound was updated and we can actually prune this node now
	if gap < parameters.gap_prune

		@printf "global_upper_bound was updated in the meanwhile and we can prune this node now\n"

		ret::ReturnValue = ReturnValue(
			subproblem.lower_bound[],
			zeros(Float64, n),
			zeros(Float64, n, n),
			0,
			0,
			+Inf,
			sum(subproblem.U - subproblem.L),
			0.0,
			zeros(Float64, n),
			zeros(Float64, n + 1, n + 1),
			zeros(Float64, n),
			Vector{RLT_Cut}(),
			time() - tstart_subproblem,
			subproblem
		)

		#@assert ret.lower_bound == subproblem.lower_bound[]

		put!(channel, ret)

	elseif compute_number_of_labeled_data_points(subproblem.L, subproblem.U) == n

		# construct labeling
		y::Vector{Float64} = zeros(Float64,n)
		for i = 1:n
			if subproblem.L[i] > 0.0
				y[i] = +1.0
			else
				y[i] = -1.0
			end
		end

		l::Int64 = count(x -> x != 0.0, instance.y)
		rhs::Float64 = 1.0 * sum(instance.y[1:l]) * (n - l) / l

		# solve convex QP
		objective_value::Float64, sol::Vector{Float64} = evaluate_labeling_qp(instance.C, y, l, rhs)
		@printf "We have reached a leaf node. We solve the convex QP to get an upper bound.\n"
		@printf "Moreover, we also use this upper bound as a lower bound since a dualbound for the convex QP cannot be obtained at the moment.\n"
		@printf "This means that the accuracy of the QP solver is crucial in order to obtain the desired duality gap.\n"

		try_to_update_global_upper_bound(n, objective_value, sol, y, instance)

		Threads.atomic_xchg!(subproblem.lower_bound, objective_value)

		ret = ReturnValue(
			objective_value,
			zeros(Float64, n),
			zeros(Float64, n, n),
			0,
			0,
			+Inf,
			sum(subproblem.U - subproblem.L),
			0.0,
			zeros(Float64, n),
			zeros(Float64, n + 1, n + 1),
			zeros(Float64, n),
			Vector{RLT_Cut}(),
			time() - tstart_subproblem,
			subproblem
		)

		#@assert ret.lower_bound == subproblem.lower_bound[]

		put!(channel, ret)

	else

		ret = sdp_cut_and_tighten(n, instance.C, instance.Cinv, instance.C_tilde, subproblem.L, subproblem.U, max_threads, subproblem, subproblem.inherited_cuts, instance, parameters)
		
		#@assert ret.lower_bound == subproblem.lower_bound[]

		put!(channel, ret)

	end

end





# returns: lower bound (can be Inf if infeasible), X ((n+1) x (n+1) - matrix)
# L and U can be changed by the function
# the global variable "global_upper_bound" can be changed by the function
# TODO start with intial set of RLT cuts inherited from the parent node
function sdp_cut_and_tighten(
	n::Int64,
	C::Matrix{Float64},
	Cinv::Matrix{Float64},
	C_tilde::Matrix{Float64},
	L::Vector{Float64},
	U::Vector{Float64},
	num_threads_mosek::Int64,
	subproblem::Subproblem,
	start_cuts::Vector{RLT_Cut},
	instance::S3vm_instance,
	parameters::MyParameters)

    #@assert n >= 1
	#@assert size(C,1) == n
	#@assert size(C,2) == n
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	#@assert size(Cinv,1) == n
	#@assert size(Cinv,2) == n
	#@assert issymmetric(Cinv)
	#@assert eigmin(Cinv) > 0.0
	#@assert norm(C * Cinv - I(n)) < 1e-8
	#@assert norm(Cinv * C - I(n)) < 1e-8
	#@assert size(C_tilde,1) == n + 1
	#@assert size(C_tilde,2) == n + 1
	#@assert issymmetric(C_tilde)
	#@assert eigmin(C_tilde) >= 0.0
	#@assert eigmin(C_tilde[1:n,1:n]) > 0.0
	#@assert length(L) == n
	#@assert length(U) == n
	#@assert num_threads_mosek >= 0
	#@assert isnothing(subproblem) || typeof(subproblem) == Subproblem
	#@assert isnothing(start_cuts) || typeof(start_cuts) == Vector{RLT_Cut}
	for i = 1:n
		#@assert isfinite(L[i])
		#@assert isfinite(U[i])
		#@assert L[i] < U[i] + TOL_BOUND
	end
    
	#@assert compute_number_of_labeled_data_points(L, U) < n # this function should not be called if this is the case

	tstart_subproblem::Float64 = time()

	lower_main_diagonal::Vector{Float64} = zeros(Float64, n)
	upper_main_diagonal::Vector{Float64} = zeros(Float64, n)

	for i = 1:n
		lower_main_diagonal[i] = 1.0 # TODO different value?
		upper_main_diagonal[i] = max(L[i]^2, U[i]^2)
	end

	volume_start::Float64 = sum(U - L)

	volume::Float64 = sum(U - L)

	added_rlt_cuts = Dict{RLT_Cut, Union{ConstraintRef, Nothing}}()

	#added_tri_cuts = Dict{Tri_Cut, Union{ConstraintRef, Nothing}}()
	
	if !isnothing(start_cuts)
		for cut::RLT_Cut in collect(start_cuts)
			added_rlt_cuts[cut] = nothing
		end
	end

	iter::Int64 = 0

	lower_bound::Float64 = 0.0
	
	local_upper_bound::Float64 = +Inf

	l::Int64 = count(x -> x != 0.0, instance.y)
	rhs::Float64 = 1.0 * sum(instance.y[1:l]) * (n - l) / l
	for i = 1:l
		#@assert instance.y[i] != 0.0
	end

	while true

		iter += 1

		project_bound_constraints!(L, U)

		# create model and set Mosek parameters
		model = Model(Mosek.Optimizer)
		set_silent(model)
		
		if num_threads_mosek > 0	                      
			set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", num_threads_mosek)
		end
		
		# add psd matrix variable
		@variable(model, X_var[1:n+1, 1:n+1], Symmetric)
		psd_constraint::ConstraintRef = @constraint(model, X_var in PSDCone())

		one_corner::ConstraintRef = @constraint(model, X_var[n+1,n+1] == 1.0)

		balancing_constraint::ConstraintRef = @constraint(model, sum(X_var[n+1,i] for i in l+1:n) == rhs)

		if parameters.add_product_constraints
			@constraint(model, product_constraints[j = 1:n], sum(X_var[i,j] for i = l+1:n) == rhs * X_var[n+1,j])
		end

		lower_bound_constraints::Vector{Union{ConstraintRef, Nothing}} = Vector{Union{ConstraintRef, Nothing}}(undef, n)
		upper_bound_constraints::Vector{Union{ConstraintRef, Nothing}} = Vector{Union{ConstraintRef, Nothing}}(undef, n)
		lower_main_diagonal_constraints::Vector{Union{ConstraintRef, Nothing}} = Vector{Union{ConstraintRef, Nothing}}(undef, n)
		upper_main_diagonal_constraints::Vector{Union{ConstraintRef, Nothing}} = Vector{Union{ConstraintRef, Nothing}}(undef, n)

		for i = 1:n
			lower_bound_constraints[i] = nothing
			upper_bound_constraints[i] = nothing
			lower_main_diagonal_constraints[i] = nothing
			upper_main_diagonal_constraints[i] = nothing
		end

		if parameters.use_small_relaxation
			for i = 1:n
				if L[i] > 0.0
					lower_bound_constraints[i] = @constraint(model, X_var[n+1,i] >= L[i])
				end
				if U[i] < 0.0
					upper_bound_constraints[i] = @constraint(model, - X_var[n+1,i] >= - U[i])
				end
				if L[i] < 0.0 && U[i] > 0.0
					lower_main_diagonal_constraints[i] = @constraint(model, X_var[i,i] >= lower_main_diagonal[i])
				end
				upper_main_diagonal_constraints[i] = @constraint(model, - X_var[i,i] >= - upper_main_diagonal[i])
			end
		else # add everything
			for i = 1:n
				lower_bound_constraints[i] = @constraint(model, X_var[n+1,i] >= L[i])
				upper_bound_constraints[i] = @constraint(model, - X_var[n+1,i] >= - U[i])
				lower_main_diagonal_constraints[i] = @constraint(model, X_var[i,i] >= lower_main_diagonal[i])
				upper_main_diagonal_constraints[i] = @constraint(model, - X_var[i,i] >= - upper_main_diagonal[i])
			end
		end

		# add RLT constraints
		#for cut in collect(keys(added_rlt_cuts))
		for cut::RLT_Cut in keys(added_rlt_cuts)
		
		    i::Int64 = cut.i
		    j::Int64 = cut.j
		    type::Int64 = cut.type

		    #@assert 1 <= i && i < j && j <= n
		    #@assert 1 <= type && type <= 4

		    if type == 1
		        added_rlt_cuts[cut] = @constraint(model, X_var[i,j] - U[i] * X_var[n+1,j] - U[j] * X_var[n+1,i] + U[i] * U[j] >= 0.0)
		    elseif type == 2
		        added_rlt_cuts[cut] = @constraint(model, X_var[i,j] - L[i] * X_var[n+1,j] - L[j] * X_var[n+1,i] + L[i] * L[j] >= 0.0)
		    elseif type == 3
		        added_rlt_cuts[cut] = @constraint(model, L[i] * X_var[n+1,j] + U[j] * X_var[n+1,i] - L[i] * U[j] - X_var[i,j] >= 0.0)
		    elseif type == 4
		        added_rlt_cuts[cut] = @constraint(model, U[i] * X_var[n+1,j] + L[j] * X_var[n+1,i] - U[i] * L[j] - X_var[i,j] >= 0.0)
		    end

		end

		# add triangle inequalities
		# for cut::Tri_Cut in keys(added_tri_cuts)
		
		#     i::Int64 = cut.i
		#     j::Int64 = cut.j
		# 	k::Int64 = cut.k
		#     type::Int64 = cut.type

		#     #@assert 1 <= i && i < j && j < k <= n
		#     #@assert 1 <= type && type <= 4

		#     if type == 1
		#         added_tri_cuts[cut] = @constraint(model, (U[k] - L[k]) * X_var[i,j] + (U[j] - L[j]) * X_var[i,k] + (U[i] - L[i]) * X_var[j,k] + (L[j] * L[k] - U[j] * U[k]) * X_var[n+1,i] + (L[i] * L[k] - U[i] * U[k]) * X_var[n+1,j] + (L[i] * L[j] - U[i] * U[j]) * X_var[n+1,k] + U[i] * U[j] * U[k] - L[i] * L[j] * L[k] >= 0.0)
		#     elseif type == 2
		#         added_tri_cuts[cut] = @constraint(model, (U[k] - L[k]) * X_var[i,j] + (L[j] - U[j]) * X_var[i,k] + (L[i] - U[i]) * X_var[j,k] + (L[k] * U[j] - L[j] * U[k]) * X_var[n+1,i] + (L[k] * U[i] - L[i] * U[k]) * X_var[n+1,j] + (U[i] * U[j] - L[i] * L[j]) * X_var[n+1,k] + L[i] * L[j] * U[k] - L[k] * U[i] * U[j] >= 0.0)
		#     elseif type == 3
		#         added_tri_cuts[cut] = @constraint(model, (L[k] - U[k]) * X_var[i,j] + (U[j] - L[j]) * X_var[i,k] + (L[i] - U[i]) * X_var[j,k] + (L[j] * U[k] - L[k] * U[j]) * X_var[n+1,i] + (U[i] * U[k] - L[i] * L[k]) * X_var[n+1,j] + (L[j] * U[i] - L[i] * U[j]) * X_var[n+1,k] + L[i] * L[k] * U[j] - L[j] * U[i] * U[k] >= 0.0)
		#     elseif type == 4
		#         added_tri_cuts[cut] = @constraint(model, (L[k] - U[k]) * X_var[i,j] + (L[j] - U[j]) * X_var[i,k] + (U[i] - L[i]) * X_var[j,k] + (U[j] * U[k] - L[j] * L[k]) * X_var[n+1,i] + (L[i] * U[k] - L[k] * U[i]) * X_var[n+1,j] + (L[i] * U[j] - L[j] * U[i]) * X_var[n+1,k] + L[j] * L[k] * U[i] - L[i] * U[j] * U[k] >= 0.0)
		#     end

		# end

		# set objective
		@objective(model, Min, LinearAlgebra.dot(C_tilde, X_var))
		
		# solve model
		optimize!(model)
		
		#-----------------------------------------------------------------------
		#-----------------------------------------------------------------------
		#-----------------------------------------------------------------------
		
		# check whether the model was solved with sufficient accuracy
		
		# The objective value of a solved problem can be obtained via objective_value().
		# The best known bound on the optimal objective value can be obtained via objective_bound().
		# If the solver supports it, the value of the dual objective can be obtained via dual_objective_value().
		
		if termination_status(model) != MOI.OPTIMAL
			@printf "%s in in sdp_cut_and_tighten(); id = %d, depth = %d, #unlab = %d\n" termination_status(model) subproblem.id subproblem.depth n - compute_number_of_labeled_data_points(L, U)
		end

		weird_objective_value::Bool = objective_value(model) < 1e-10 ? true : false

		subproblem_infeasible::Bool = false
		if weird_objective_value || abs(objective_value(model) - dual_objective_value(model)) / (1.0 + abs(objective_value(model)) + abs(dual_objective_value(model))) > 1.0
			subproblem_infeasible = true
			@printf "Mosek reported \"%s\". However we got objective_value(model) = %.17e and dual_objective_value(model) = %.17e. We treat this subproblem (id = %d, depth = %d, #unlab = %d) as infeasible.\n" termination_status(model) objective_value(model) dual_objective_value(model) subproblem.id subproblem.depth n - compute_number_of_labeled_data_points(L, U)
		end

		if termination_status(model) == MOI.INFEASIBLE || subproblem_infeasible

			@printf "Subproblem (id = %d, depth = %d, #unlab = %d) is infeasible!\n" subproblem.id subproblem.depth n - compute_number_of_labeled_data_points(L, U)

			Threads.atomic_xchg!(subproblem.lower_bound, +Inf)

			volume_end::Float64 = sum(U - L)
			volume_diff::Float64 = volume_start - volume_end

			ret::ReturnValue = ReturnValue(
				+Inf,
				zeros(Float64, n),
				zeros(Float64, n, n),
				iter,
				length(added_rlt_cuts),
				local_upper_bound,
				volume_end,
				volume_diff,
				zeros(Float64, n),
				zeros(Float64, n + 1, n + 1),
				zeros(Float64, n),
				Vector{RLT_Cut}(),
				time() - tstart_subproblem,
				subproblem
			)

			return ret
		end

		##@assert termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.SLOW_PROGRESS
		
		#@assert abs(objective_value(model) - objective_bound(model)) / (1.0 + abs(objective_value(model)) + abs(objective_bound(model))) < 1e-5
		#@assert abs(objective_value(model) - dual_objective_value(model)) / (1.0 + abs(objective_value(model)) + abs(dual_objective_value(model))) < 1e-5
		#@assert abs(dual_objective_value(model) - objective_bound(model)) / (1.0 + abs(dual_objective_value(model)) + abs(objective_bound(model))) < 1e-5

		# get SDP solution (RLT cuts are ignored at the moment)
		X::Matrix{Float64} = value.(X_var)
		Z::Matrix{Float64} = dual(psd_constraint)

		dual_L::Vector{Float64} = zeros(Float64, n)
		dual_U::Vector{Float64} = zeros(Float64, n)
		dual_LM::Vector{Float64} = zeros(Float64, n)
		dual_UM::Vector{Float64} = zeros(Float64, n)

		for i = 1:n
			if !isnothing(lower_bound_constraints[i])
				dual_L[i] = dual(lower_bound_constraints[i])
			end
			if !isnothing(upper_bound_constraints[i])
				dual_U[i] = dual(upper_bound_constraints[i])
			end
			if !isnothing(lower_main_diagonal_constraints[i])
				dual_LM[i] = dual(lower_main_diagonal_constraints[i])
			end
			if !isnothing(upper_main_diagonal_constraints[i])
				dual_UM[i] = dual(upper_main_diagonal_constraints[i])
			end
		end

		#dual_L::Vector{Float64} = dual.(lower_bound_constraints)
		#dual_U::Vector{Float64} = dual.(upper_bound_constraints)
		#dual_LM::Vector{Float64} = dual.(lower_main_diagonal_constraints)
		#dual_UM::Vector{Float64} = dual.(upper_main_diagonal_constraints)
		
		#@assert size(X,1) == n + 1
		#@assert size(X,2) == n + 1
		#@assert size(Z,1) == n + 1
		#@assert size(Z,2) == n + 1
		#@assert length(dual_L) == n
		#@assert length(dual_U) == n
		#@assert length(dual_LM) == n
		#@assert length(dual_UM) == n
		#@assert issymmetric(X)
		#@assert issymmetric(Z)
		#@assert eigmin(X) > -1e-8 # X must be positive semidefinite
		#@assert eigmin(Z) > -1e-8 # Z must be positive semidefinite
		#@assert minimum(dual_L) > -1e-5
		#@assert minimum(dual_U) > -1e-5
		#@assert minimum(dual_LM) > -1e-5
		#@assert minimum(dual_UM) > -1e-5
		
		# check whether primal solution is roughly feasible (not so important)
		#@assert abs(X[n+1,n+1] - 1.0) < 1e-4
		for i = 1:n
			#@assert X[i,i] > 0.9999
			##@assert X[n+1,i] > L[i] - 1e-4
			##@assert X[n+1,i] < U[i] + 1e-4
			#@assert X[i,i] >= X[n+1,i]^2 - 1e-4
		end
		
		# make dual multipliers feasible
		dual_L .= max.(dual_L, 0.0)
		dual_U .= max.(dual_U, 0.0)
		dual_LM .= max.(dual_LM, 0.0)
		dual_UM .= max.(dual_UM, 0.0)
		
		#@assert minimum(dual_L) >= 0.0
		#@assert minimum(dual_U) >= 0.0
		#@assert minimum(dual_LM) >= 0.0
		#@assert minimum(dual_UM) >= 0.0
		#-----------------------------------------------------------------------
		
		# construct the dual slack matrix Z ("my_Z") and keep track of the dual objective value
		
		my_dual_objective_value::Float64 = 0.0
		
		my_Z::Matrix{Float64} = deepcopy(C_tilde)
		#@assert size(my_Z,1) == n + 1
		#@assert size(my_Z,2) == n + 1
		#@assert issymmetric(my_Z)
		#@assert eigmin(my_Z) >= 0.0
		#@assert eigmin(my_Z[1:n,1:n]) > 0.0
		
		# the "1" in the corner
		my_Z[n+1,n+1] -= dual(one_corner) # is a free variable
		my_dual_objective_value += dual(one_corner)

		my_dual_objective_value += rhs * dual(balancing_constraint)
		for i = l+1:n
			my_Z[n+1,i] -= 0.5 * dual(balancing_constraint)
			my_Z[i,n+1] -= 0.5 * dual(balancing_constraint)
		end

		# product constraints
		if parameters.add_product_constraints
			for j = 1:n
				my_dual_objective_value += 0.0
				for i = l+1:n
					my_Z[i,j] -= 0.5 * dual(product_constraints[j])
					my_Z[j,i] -= 0.5 * dual(product_constraints[j])
				end
				my_Z[n+1,j] += 0.5 * rhs * dual(product_constraints[j])
				my_Z[j,n+1] += 0.5 * rhs * dual(product_constraints[j])
			end
		end
		
		for i = 1:n
		
			my_Z[n+1,i] -= 0.5 * dual_L[i]
			my_Z[i,n+1] -= 0.5 * dual_L[i]
			my_dual_objective_value += dual_L[i] * L[i]
			
			my_Z[n+1,i] += 0.5 * dual_U[i]
			my_Z[i,n+1] += 0.5 * dual_U[i]
			my_dual_objective_value -= dual_U[i] * U[i]
			
			my_Z[i,i] -= dual_LM[i]
			my_dual_objective_value += dual_LM[i] * lower_main_diagonal[i]
			
			my_Z[i,i] += dual_UM[i]
			my_dual_objective_value -= dual_UM[i] * upper_main_diagonal[i]
			
		end
		
		#@assert issymmetric(my_Z)
		
		#-----------------------------------------------------------------------
		
		# RLT cuts
		for cut::RLT_Cut in collect(keys(added_rlt_cuts))
		
			dual_value::Float64 = dual(added_rlt_cuts[cut])
			#@assert dual_value > -1e-7
			
			dual_value = max(dual_value, 0.0) # project dual multiplier
			
			# get RLT cut information
			i::Int64 = cut.i
			j::Int64 = cut.j
			type::Int64 = cut.type
			
			if type == 1
				my_Z[i,j] -= 0.5 * dual_value
				my_Z[j,i] -= 0.5 * dual_value
				my_Z[n+1,j] += 0.5 * dual_value * U[i]
				my_Z[j,n+1] += 0.5 * dual_value * U[i]
				my_Z[n+1,i] += 0.5 * dual_value * U[j]
				my_Z[i,n+1] += 0.5 * dual_value * U[j]
				my_dual_objective_value -= dual_value * U[i] * U[j]
		    elseif type == 2
				my_Z[i,j] -= 0.5 * dual_value
				my_Z[j,i] -= 0.5 * dual_value
				my_Z[n+1,j] += 0.5 * dual_value * L[i]
				my_Z[j,n+1] += 0.5 * dual_value * L[i]
				my_Z[n+1,i] += 0.5 * dual_value * L[j]
				my_Z[i,n+1] += 0.5 * dual_value * L[j]
				my_dual_objective_value -= dual_value * L[i] * L[j]
		    elseif type == 3
				my_Z[i,j] += 0.5 * dual_value
				my_Z[j,i] += 0.5 * dual_value
				my_Z[n+1,j] -= 0.5 * dual_value * L[i]
				my_Z[j,n+1] -= 0.5 * dual_value * L[i]
				my_Z[n+1,i] -= 0.5 * dual_value * U[j]
				my_Z[i,n+1] -= 0.5 * dual_value * U[j]
				my_dual_objective_value += dual_value * L[i] * U[j]
		    elseif type == 4
				my_Z[i,j] += 0.5 * dual_value
				my_Z[j,i] += 0.5 * dual_value
				my_Z[n+1,j] -= 0.5 * dual_value * U[i]
				my_Z[j,n+1] -= 0.5 * dual_value * U[i]
				my_Z[n+1,i] -= 0.5 * dual_value * L[j]
				my_Z[i,n+1] -= 0.5 * dual_value * L[j]
				my_dual_objective_value += dual_value * U[i] * L[j]
			end
			
		end # cut loop

		# triangle inequalities
		# for cut::Tri_Cut in collect(keys(added_tri_cuts))

		# 	dual_value::Float64 = dual(added_tri_cuts[cut])
		# 	#@assert dual_value > -1e-7
			
		# 	dual_value = max(dual_value, 0.0) # project dual multiplier
			
		# 	# get RLT cut information
		# 	i::Int64 = cut.i
		# 	j::Int64 = cut.j
		# 	k::Int64 = cut.k
		# 	type::Int64 = cut.type
		# 	#@assert 1 <= i && i < j && j < k && k <= n
		# 	#@assert 1 <= type && type <= 4
			
		# 	if type == 1
		# 		my_Z[i,j] -= 0.5 * dual_value * (U[k] - L[k])
		# 		my_Z[j,i] -= 0.5 * dual_value * (U[k] - L[k])
		# 		my_Z[i,k] -= 0.5 * dual_value * (U[j] - L[j])
		# 		my_Z[k,i] -= 0.5 * dual_value * (U[j] - L[j])
		# 		my_Z[j,k] -= 0.5 * dual_value * (U[i] - L[i])
		# 		my_Z[k,j] -= 0.5 * dual_value * (U[i] - L[i])
		# 		my_Z[n+1,i] -= 0.5 * dual_value * (L[j] * L[k] - U[j] * U[k])
		# 		my_Z[i,n+1] -= 0.5 * dual_value * (L[j] * L[k] - U[j] * U[k])
		# 		my_Z[n+1,j] -= 0.5 * dual_value * (L[i] * L[k] - U[i] * U[k])
		# 		my_Z[j,n+1] -= 0.5 * dual_value * (L[i] * L[k] - U[i] * U[k])
		# 		my_Z[n+1,k] -= 0.5 * dual_value * (L[i] * L[j] - U[i] * U[j])
		# 		my_Z[k,n+1] -= 0.5 * dual_value * (L[i] * L[j] - U[i] * U[j])
		# 		my_dual_objective_value -= dual_value * (U[i] * U[j] * U[k] - L[i] * L[j] * L[k])
		# 	elseif type == 2
		# 		my_Z[i,j] -= 0.5 * dual_value * (U[k] - L[k])
		# 		my_Z[j,i] -= 0.5 * dual_value * (U[k] - L[k])
		# 		my_Z[i,k] -= 0.5 * dual_value * (L[j] - U[j])
		# 		my_Z[k,i] -= 0.5 * dual_value * (L[j] - U[j])
		# 		my_Z[j,k] -= 0.5 * dual_value * (L[i] - U[i])
		# 		my_Z[k,j] -= 0.5 * dual_value * (L[i] - U[i])
		# 		my_Z[n+1,i] -= 0.5 * dual_value * (L[k] * U[j] - L[j] * U[k])
		# 		my_Z[i,n+1] -= 0.5 * dual_value * (L[k] * U[j] - L[j] * U[k])
		# 		my_Z[n+1,j] -= 0.5 * dual_value * (L[k] * U[i] - L[i] * U[k])
		# 		my_Z[j,n+1] -= 0.5 * dual_value * (L[k] * U[i] - L[i] * U[k])
		# 		my_Z[n+1,k] -= 0.5 * dual_value * (U[i] * U[j] - L[i] * L[j])
		# 		my_Z[k,n+1] -= 0.5 * dual_value * (U[i] * U[j] - L[i] * L[j])
		# 		my_dual_objective_value -= dual_value * (L[i] * L[j] * U[k] - L[k] * U[i] * U[j])
		# 	elseif type == 3
		# 		my_Z[i,j] -= 0.5 * dual_value * (L[k] - U[k])
		# 		my_Z[j,i] -= 0.5 * dual_value * (L[k] - U[k])
		# 		my_Z[i,k] -= 0.5 * dual_value * (U[j] - L[j])
		# 		my_Z[k,i] -= 0.5 * dual_value * (U[j] - L[j])
		# 		my_Z[j,k] -= 0.5 * dual_value * (L[i] - U[i])
		# 		my_Z[k,j] -= 0.5 * dual_value * (L[i] - U[i])
		# 		my_Z[n+1,i] -= 0.5 * dual_value * (L[j] * U[k] - L[k] * U[j])
		# 		my_Z[i,n+1] -= 0.5 * dual_value * (L[j] * U[k] - L[k] * U[j])
		# 		my_Z[n+1,j] -= 0.5 * dual_value * (U[i] * U[k] - L[i] * L[k])
		# 		my_Z[j,n+1] -= 0.5 * dual_value * (U[i] * U[k] - L[i] * L[k])
		# 		my_Z[n+1,k] -= 0.5 * dual_value * (L[j] * U[i] - L[i] * U[j])
		# 		my_Z[k,n+1] -= 0.5 * dual_value * (L[j] * U[i] - L[i] * U[j])
		# 		my_dual_objective_value -= dual_value * (L[i] * L[k] * U[j] - L[j] * U[i] * U[k])
		# 	elseif type == 4
		# 		my_Z[i,j] -= 0.5 * dual_value * (L[k] - U[k])
		# 		my_Z[j,i] -= 0.5 * dual_value * (L[k] - U[k])
		# 		my_Z[i,k] -= 0.5 * dual_value * (L[j] - U[j])
		# 		my_Z[k,i] -= 0.5 * dual_value * (L[j] - U[j])
		# 		my_Z[j,k] -= 0.5 * dual_value * (U[i] - L[i])
		# 		my_Z[k,j] -= 0.5 * dual_value * (U[i] - L[i])
		# 		my_Z[n+1,i] -= 0.5 * dual_value * (U[j] * U[k] - L[j] * L[k])
		# 		my_Z[i,n+1] -= 0.5 * dual_value * (U[j] * U[k] - L[j] * L[k])
		# 		my_Z[n+1,j] -= 0.5 * dual_value * (L[i] * U[k] - L[k] * U[i])
		# 		my_Z[j,n+1] -= 0.5 * dual_value * (L[i] * U[k] - L[k] * U[i])
		# 		my_Z[n+1,k] -= 0.5 * dual_value * (L[i] * U[j] - L[j] * U[i])
		# 		my_Z[k,n+1] -= 0.5 * dual_value * (L[i] * U[j] - L[j] * U[i])
		# 		my_dual_objective_value -= dual_value * (L[j] * L[k] * U[i] - L[i] * U[j] * U[k])
		# 	end
			
		# end # cut loop
		
		#-----------------------------------------------------------------------

		#@assert issymmetric(my_Z)
		#@assert eigmin(my_Z) > -1e-6
		#@assert norm(Z - my_Z) / abs(1.0 + norm(Z) + norm(my_Z)) < 1e-5
		#@assert abs(dual_objective_value(model) - my_dual_objective_value) / max(1.0 + abs(dual_objective_value(model)) + abs(my_dual_objective_value)) < 1e-3
				
		#-----------------------------------------------------------------------
		
		# compute dual bound
		
		max_sum_trace::Float64 = 1.0
		for i = 1:n
			max_sum_trace += max(L[i]^2, U[i]^2)
		end
		# "max_sum_trace" is now an upper bound on the maximum eigenvalue of the optimal SDP solution X
		
		# compute the sum of all negative eigenvalues of my_Z
		eigvals_my_Z::Vector{Float64} = eigvals(my_Z)
		sum_neg::Float64 = 0.0
		for i = 1:n+1
			if eigvals_my_Z[i] < 0.0
				sum_neg += eigvals_my_Z[i]
			end
		end
		
		# correct "my_dual_objective_value"
		my_dual_objective_value += max_sum_trace * sum_neg
		#@assert abs(dual_objective_value(model) - my_dual_objective_value) / max(1.0 + abs(dual_objective_value(model)) + abs(my_dual_objective_value)) < 1e-2

		if my_dual_objective_value > subproblem.lower_bound[]
			Threads.atomic_xchg!(subproblem.lower_bound, my_dual_objective_value)
		end
		
		#@printf "\neigmax(X) = %g\n" eigmax(X)
		#@printf "bound on eigmax = %g\n" max_sum_trace
		#@printf "sum_neg = %g\n" sum_neg
		#@printf "penalty using eigmax = %g\n" eigmax(X) * sum_neg
		#@printf "penalty using max_sum_trace = %g\n" max_sum_trace * sum_neg
		#@printf "|| Z _mosek - my_Z || = %g\n" norm(Z - my_Z)
		#@printf "mosek dual = %.8f\n" dual_objective_value(model)
		#@printf "my_dual_objective_value = %.8f\n" my_dual_objective_value
		#@printf "| mosek_dual - my_dual_objective_value | = %g\n" abs(dual_objective_value(model) - my_dual_objective_value)
		
		#-----------------------------------------------------------------------
		#-----------------------------------------------------------------------
		#-----------------------------------------------------------------------
		
		# it is important to compute "lower_bound_improvement" first; otherwise, we would overwrite the last value of "lower_bound"
		#lower_bound_improvement = objective_value(model) - lower_bound
		#lower_bound = objective_value(model)
		
		lower_bound_improvement::Float64 = my_dual_objective_value - lower_bound
		old_lower_bound::Float64 = lower_bound
		lower_bound = max(lower_bound, my_dual_objective_value)

		#@assert lower_bound_improvement / (1.0 + abs(old_lower_bound) + abs(lower_bound)) > -1e-2
		
		gap::Float64 = compute_gap(lower_bound, instance.upper_bound[])
		
		success::Bool = false
		
		if gap >= parameters.gap_prune

			better_upper_bound_found::Bool = false
		
			if parameters.locally_infeasible_solutions == 0 || parameters.locally_infeasible_solutions == 1

				yl = zeros(n)
				for i = 1:n
					if L[i] > 0.0
						yl[i] = +1.0
					end
					if U[i] < 0.0
						yl[i] = -1.0
					end
				end

				success, upper_bound::Float64 = run_heuristic(C, value.(X_var[n+1, 1:n]), value.(X_var[1:n, 1:n]), yl, subproblem, instance, parameters.num_runs_heuristics, parameters.use_local_search)

				# all solutions computed are feasible for this subproblem
				local_upper_bound = min(local_upper_bound, upper_bound)
				#@assert instance.upper_bound[] <= upper_bound

				better_upper_bound_found = success

				# check that the solution stored in subproblem.x is feasible
				if !isnothing(subproblem)
					for i = 1:n
						#@assert yl[i] * subproblem.x[i] >= 0.0
					end
					#@assert minimum(abs.(subproblem.x)) >= 1.0
				end

			end

			if parameters.locally_infeasible_solutions == 1 || parameters.locally_infeasible_solutions == 2

				yl::Vector{Float64} = deepcopy(instance.y)

				success, upper_bound = run_heuristic(C, value.(X_var[n+1, 1:n]), value.(X_var[1:n, 1:n]), yl, subproblem, instance, parameters.num_runs_heuristics, parameters.use_local_search)
				
				if success
					better_upper_bound_found = true
				end

			end
		
			# try to compute tighter bound constraints
			if better_upper_bound_found && compute_gap(lower_bound, instance.upper_bound[]) >= parameters.gap_prune && parameters.recompute_bound_constraints && (subproblem.depth == 0 || parameters.num_threads == 1)
				labeled_before::Int64 = compute_number_of_labeled_data_points(L, U)
				compute_L_and_U!(n, C, Cinv, C_tilde, L, U, "qp", instance.upper_bound[], l, rhs)
				project_bound_constraints!(L, U)
				labeled_after::Int64 = compute_number_of_labeled_data_points(L, U)
				@printf "new volume: %.4f\n" sum(U - L)
				@printf "%d data points were labeled by recomputing the bound constraints\n" labeled_after - labeled_before
			end

		end
		
		#@assert instance.upper_bound[] <= local_upper_bound

		diff::Float64 = instance.upper_bound[] - lower_bound
		
		if diff > 0.0
			
			#@assert TOL_P > 0.0
			#@assert TOL_D > 0.0
			
			# bound tightening
			if parameters.use_bound_tightening
			
				for i = 1:n

					# apply to lower bound constraints
					if X[n+1,i] <= L[i] + TOL_P && dual_L[i] > TOL_D
						U[i] = min(U[i], L[i] + diff / dual_L[i] + TOL_BOUND)
					end

					# apply to upper bound constraints
					if X[n+1,i] >= U[i] - TOL_P && dual_U[i] > TOL_D
						L[i] = max(L[i], U[i] - diff / dual_U[i] - TOL_BOUND)
					end

					# apply to main diagonal (lower bound)
					if X[i,i] <= lower_main_diagonal[i] + TOL_P && dual_LM[i] > TOL_D
						upper_main_diagonal[i] = min(upper_main_diagonal[i], lower_main_diagonal[i] + diff / dual_LM[i] + TOL_BOUND)
						L[i] = max(L[i], - sqrt(lower_main_diagonal[i] + diff / dual_LM[i]) - TOL_BOUND)
						U[i] = min(U[i], sqrt(lower_main_diagonal[i] + diff / dual_LM[i]) + TOL_BOUND)
					end

					# apply to main diagonal (upper bound)
					if X[i,i] >= upper_main_diagonal[i] - TOL_P && dual_UM[i] > TOL_D
						lower_main_diagonal[i] = max(lower_main_diagonal[i], upper_main_diagonal[i] - diff / dual_UM[i] - TOL_BOUND)
						if upper_main_diagonal[i] - diff / dual_UM[i] >= 1
						    if L[i] > - sqrt(upper_main_diagonal[i] - diff / dual_UM[i] + TOL_BOUND)
						        L[i] = max(L[i], +1.0)
						    end
						    if U[i] < sqrt(upper_main_diagonal[i] - diff / dual_UM[i] - TOL_BOUND)
						        U[i] = min(U[i], -1.0)
						    end
						end
					end

				end
				
			end # bound tightening
			
		end # diff
		
		# check whether problem is infeasible now
		for i = 1:n
			if L[i] > -1.0 + TOL_BOUND && U[i] < +1.0 - TOL_BOUND || L[i] > U[i] + TOL_BOUND
				# problem is infeasible
				#@assert false
				error("This subproblem is infeasible with respect to the box constraints!\n")
			end
		end
		
		project_bound_constraints!(L, U)

################################################################################

		rlt_sum::Vector{Float64} = zeros(n)

		removed::Int64 = 0
		# remove (almost) inactive RLT cuts
		for cut::RLT_Cut in collect(keys(added_rlt_cuts))
		
			rlt_sum[cut.i] += abs(dual(added_rlt_cuts[cut]))
			rlt_sum[cut.j] += abs(dual(added_rlt_cuts[cut]))
		
		    if abs(dual(added_rlt_cuts[cut])) < TOL_D # TODO also look at primal violation
		        delete!(added_rlt_cuts, cut)
		        removed += 1
		    end
		end

		# for cut::Tri_Cut in collect(keys(added_tri_cuts))
		#     if abs(dual(added_tri_cuts[cut])) < TOL_D # TODO also look at primal violation
		#         delete!(added_tri_cuts, cut)
		#         removed += 1
		#     end
		# end

		gap = compute_gap(lower_bound, instance.upper_bound[])

		if gap >= parameters.gap_prune
		
			# add violated RLT cuts
			new_rlt_cuts::Vector{Tuple{RLT_Cut, Float64}} = Vector{Tuple{RLT_Cut, Float64}}() # TODO we could also add the violation to the struct!
			for i = 1:n
				for j = i + 1:n

				    val::Float64 = X[i,j] - U[i] * X[n+1,j] - U[j] * X[n+1,i] + U[i] * U[j]
				    if val < parameters.minimum_violation
				        push!(new_rlt_cuts, (RLT_Cut(i, j, 1), val))
				    end

				    val = X[i,j] - L[i] * X[n+1,j] - L[j] * X[n+1,i] + L[i] * L[j]
				    if val < parameters.minimum_violation
				        push!(new_rlt_cuts, (RLT_Cut(i, j, 2), val))
				    end

				    val = L[i] * X[n+1,j] + U[j] * X[n+1,i] - L[i] * U[j] - X[i,j]
				    if val < parameters.minimum_violation
				        push!(new_rlt_cuts, (RLT_Cut(i, j, 3), val))
				    end

				    val = U[i] * X[n+1,j] + L[j] * X[n+1,i] - U[i] * L[j] - X[i,j]
				    if val < parameters.minimum_violation
				        push!(new_rlt_cuts, (RLT_Cut(i, j, 4), val))
				    end

				end
			end

			# find and add the most violated cuts that were not added before
			# NOTE: since we have strengthened the RLT cuts, we might add an already included cut again
			sort!(new_rlt_cuts, by = x -> x[2])
			max_new_rlt_cuts::Int64 = Int64(floor(min(parameters.factor_maximum_cuts_dualbound * n, length(new_rlt_cuts))))
			added::Int64 = 0
			for (cut::RLT_Cut, _) in new_rlt_cuts
				# only add cut if it is not stored yet
				if !haskey(added_rlt_cuts, cut)
				    added_rlt_cuts[cut] = nothing
				    added += 1
				    if added >= max_new_rlt_cuts
				        break # maximum number of added cuts reached
				    end
				end
			end

			# add violated triangle inequalities
			# new_tri_cuts::Vector{Tuple{Tri_Cut, Float64}} = Vector{Tuple{Tri_Cut, Float64}}() # TODO we could also add the violation to the struct!
			# for i = 1:n
			# 	for j = i + 1:n
			# 		for k = j + 1:n

			# 			val::Float64 = (U[k] - L[k]) * X[i,j] + (U[j] - L[j]) * X[i,k] + (U[i] - L[i]) * X[j,k] + (L[j] * L[k] - U[j] * U[k]) * X[n+1,i] + (L[i] * L[k] - U[i] * U[k]) * X[n+1,j] + (L[i] * L[j] - U[i] * U[j]) * X[n+1,k] + U[i] * U[j] * U[k] - L[i] * L[j] * L[k]
			# 			if val < parameters.minimum_violation
			# 				push!(new_tri_cuts, (Tri_Cut(i, j, k, 1), val))
			# 			end

			# 			val = (U[k] - L[k]) * X[i,j] + (L[j] - U[j]) * X[i,k] + (L[i] - U[i]) * X[j,k] + (L[k] * U[j] - L[j] * U[k]) * X[n+1,i] + (L[k] * U[i] - L[i] * U[k]) * X[n+1,j] + (U[i] * U[j] - L[i] * L[j]) * X[n+1,k] + L[i] * L[j] * U[k] - L[k] * U[i] * U[j]
			# 			if val < parameters.minimum_violation
			# 				push!(new_tri_cuts, (Tri_Cut(i, j, k, 2), val))
			# 			end

			# 			val = (L[k] - U[k]) * X[i,j] + (U[j] - L[j]) * X[i,k] + (L[i] - U[i]) * X[j,k] + (L[j] * U[k] - L[k] * U[j]) * X[n+1,i] + (U[i] * U[k] - L[i] * L[k]) * X[n+1,j] + (L[j] * U[i] - L[i] * U[j]) * X[n+1,k] + L[i] * L[k] * U[j] - L[j] * U[i] * U[k]
			# 			if val < parameters.minimum_violation
			# 				push!(new_tri_cuts, (Tri_Cut(i, j, k, 3), val))
			# 			end

			# 			val = (L[k] - U[k]) * X[i,j] + (L[j] - U[j]) * X[i,k] + (U[i] - L[i]) * X[j,k] + (U[j] * U[k] - L[j] * L[k]) * X[n+1,i] + (L[i] * U[k] - L[k] * U[i]) * X[n+1,j] + (L[i] * U[j] - L[j] * U[i]) * X[n+1,k] + L[j] * L[k] * U[i] - L[i] * U[j] * U[k]
			# 			if val < parameters.minimum_violation
			# 				push!(new_tri_cuts, (Tri_Cut(i, j, k, 4), val))
			# 			end

			# 		end # k
			# 	end # j
			# end # I

			# sort!(new_tri_cuts, by = x -> x[2])
			# max_new_tri_cuts::Int64 = min(parameters.factor_maximum_cuts_dualbound * n, length(new_tri_cuts))
			# added = 0
			# for (cut::Tri_Cut, _) in new_tri_cuts
			# 	# only add cut if it is not stored yet
			# 	if !haskey(added_tri_cuts, cut)
			# 	    added_tri_cuts[cut] = nothing
			# 	    added += 1
			# 	    if added >= max_new_tri_cuts
			# 	        break # maximum number of added cuts reached
			# 	    end
			# 	end
			# end
			
		end
		
################################################################################
		
		new_volume::Float64 = sum(U - L)
		gap = compute_gap(lower_bound, instance.upper_bound[])

		# decide whether we should stop
		if lower_bound_improvement / old_lower_bound < parameters.rel_tol_cutting_plane_approach || gap < parameters.gap_prune || iter >= parameters.max_cutting_plane_iter
		
			volume_end = sum(U - L)
			volume_diff = volume_start - volume_end
			#@assert volume_diff >= 0.0
			
			final_cuts::Vector{RLT_Cut} = RLT_Cut[]
			for cut::RLT_Cut in collect(keys(added_rlt_cuts))
				push!(final_cuts, cut)
			end

			ret = ReturnValue(
				max(subproblem.lower_bound[], lower_bound),
				value.(X_var[n+1, 1:n]),
				value.(X_var[1:n, 1:n]),
				iter,
				length(added_rlt_cuts),
				local_upper_bound,
				volume_end,
				volume_diff,
				dual_LM,
				Z,
				rlt_sum,
				final_cuts,
				time() - tstart_subproblem,
				subproblem
			)

			return ret
		    #return max(subproblem.lower_bound, lower_bound), value.(X_var[n+1, 1:n]), value.(X_var[1:n, 1:n]), iter, length(added_rlt_cuts), local_upper_bound, volume_end, volume_diff, dual_LM, Z, rlt_sum, final_cuts
		end

		volume = new_volume

	end # iter loop

	
end # sdp_cut_and_tighten()



function find_best_kernel_alignment(n::Int64, X_data::Matrix{Float64}, y::Vector{Float64})

	#@assert n >= 1
	#@assert size(X_data,1) == n
	#@assert length(y) == n
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0
	
	best_gamma::Float64 = 0.0
	best_kernel::String = "nothing" # linear or rbf
	best_alignment::Float64 = -1.0
	best_kernel_matrix::Matrix{Float64} = zeros(n,n)
	
	ideal_kernel::Matrix{Float64} = y * y'
	
	# linear kernel
	kernel_matrix::Matrix{Float64} = compute_kernel_matrix(n, X_data, "linear", nothing)
	centered_kernel_matrix::Matrix{Float64} = (I(n) - 1.0 / n * ones(n,n)) * kernel_matrix * (I(n) - 1.0 / n * ones(n,n))
	kernel_alignment::Float64 = dot(vec(ideal_kernel)', vec(centered_kernel_matrix)) / (n * norm(centered_kernel_matrix))
	@printf "alignment of linear kernel: %.17e\n" kernel_alignment
	
	best_kernel = "linear"
	best_alignment = kernel_alignment
	best_gamma = 0.0
	best_kernel_matrix = deepcopy(kernel_matrix)
	
	gamma_min::Float64 = 0.001
	gamma_max::Float64 = 10.0
	num_steps::Int64 = 15
	
	factor_gamma::Float64 = (gamma_max / gamma_min)^(1.0 / num_steps)
	
	gamma::Float64 = gamma_min
	
	while gamma <= gamma_max + 1e-8
	
		kernel_matrix = compute_kernel_matrix(n, X_data, "rbf", gamma)
		centered_kernel_matrix = (I(n) - 1.0 / n * ones(n,n)) * kernel_matrix * (I(n) - 1.0 / n * ones(n,n))
		
		kernel_alignment = dot(vec(ideal_kernel)', vec(centered_kernel_matrix)) / (n * norm(centered_kernel_matrix))
		@printf "alignment of rbf kernel with gamma = %.17e: %.17e\n" gamma kernel_alignment
		
		if kernel_alignment > best_alignment
			best_gamma = gamma
			best_alignment = kernel_alignment
			best_kernel = "rbf"
			best_kernel_matrix = deepcopy(kernel_matrix)
		end
		
		gamma *= factor_gamma
		
	end

	#@assert best_kernel == "linear" || best_kernel == "rbf"
	#@assert best_kernel == "linear" || best_gamma > 0.0
	
	return best_kernel, best_kernel_matrix, best_gamma, best_alignment

end



function train_svm(n::Int64, C_penalty::Float64, kernel_matrix::Matrix{Float64}, y::Vector{Float64})

	#@assert n >= 1
	#@assert C_penalty > 0.0
	#@assert length(y) == n
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0
	#@assert size(kernel_matrix,1) == n
	#@assert size(kernel_matrix,2) == n
	#@assert issymmetric(kernel_matrix)
	#@assert eigmin(kernel_matrix) > -1e-8
	
	K::Matrix{Float64} = kernel_matrix + 0.5 / C_penalty * I(n)
	#@assert issymmetric(K)
	#@assert eigmin(K) > 0.0
    
    cost_matrix::Matrix{Float64} = 0.5 * inv(K)
    cost_matrix = 0.5 * (cost_matrix + cost_matrix')
	#@assert issymmetric(cost_matrix)
	#@assert eigmin(cost_matrix) > 0.0
    
    model = Model(() -> Gurobi.Optimizer(gurobi_env); add_bridges = false)
    set_silent(model)
    
    #set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "BarQCPConvTol", 5e-9)
    
	@variable(model, x_var[i = 1:n])
    
	@constraint(model, [i = 1:n], y[i] * x_var[i] >= 1.0)
    
    @objective(model, Min, x_var' * cost_matrix * x_var)
    
	optimize!(model)
	
	if termination_status(model) != MOI.OPTIMAL
		@printf "in train_svm():\n"
		@show termination_status(model)
	end

	##@assert termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.SLOW_PROGRESS

	x::Vector{Float64} = value.(x_var)
	#@assert length(x) == n

	for i = 1:n
		#@assert y[i] * x[i] > 1.0 - 1e-10
		if x[i] < 0.0
			x[i] = min(x[i], -1.0)
		end
		if x[i] > 0.0
			x[i] = max(x[i], +1.0)
		end
	end
	
	alpha::Vector{Float64} = inv(K .* (y * y')) * (x .* y)
	#@assert length(alpha) == n

	for i = 1:n
		#@assert alpha[i] > -1e-6
		alpha[i] = max(0.0, alpha[i])
	end
	
	return alpha
	
end



function get_accuracy_cross_validation(
	n::Int64,
	y::Vector{Float64},
	folds::Vector{Tuple{Vector{Int64}, Vector{Int64}}},
	C_penalty::Float64,
	kernel_matrix::Matrix{Float64})

	#@assert n >= 1
	#@assert length(y) == n
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0
	#@assert C_penalty > 0.0
	#@assert size(kernel_matrix,1) == n
	#@assert size(kernel_matrix,2) == n
	#@assert issymmetric(kernel_matrix)
	#@assert eigmin(kernel_matrix) > -1e-8
	#@assert length(folds) >= 2
	
	accuracy::Float64 = 0.0
	
	for fold in folds
	
		train::Vector{Int64} = fold[1]
		test::Vector{Int64} = fold[2]

		length(train) >= 1
		length(test) >= 1
		
		y_train::Vector{Float64} = y[train]
		y_test::Vector{Float64} = y[test]
		
		n_train::Int64 = length(y_train)
		
		kernel_matrix_train::Matrix{Float64} = kernel_matrix[train,train]
	
		alpha::Vector{Float64} = train_svm(n_train, C_penalty, kernel_matrix_train, y_train)
		#@assert length(alpha) == n_train
		#@assert minimum(alpha) >= 0.0
		
		correct::Int64 = 0
		
		for i::Int64 in test
		
			value::Float64 = 0.0
			for j = 1:n_train
				value += y_train[j] * alpha[j] * kernel_matrix[i,train[j]]
			end
			
			prediction::Float64 = sign(value)
			#@assert abs(prediction) == 1.0
			
			if prediction == y[i]
				correct += 1
			end
			
		end # prediction loop
		
		accuracy += 1.0 * correct / length(test)
		
	end # folds
	
	return accuracy / length(folds)

end # get_accuracy_cross_validation()



function find_best_penalty_parameter(
	n::Int64,
	y::Vector{Float64},
	folds::Vector{Tuple{Vector{Int64}, Vector{Int64}}},
	kernel_matrix::Matrix{Float64},
	penalty_parameter_min::Float64,
	penalty_parameter_max::Float64,
	penalty_parameter_num_steps::Int64)

	#@assert n >= 1
	#@assert length(y) == n
	#@assert minimum(abs.(y)) == 1.0
	#@assert maximum(abs.(y)) == 1.0
	#@assert size(kernel_matrix,1) == n
	#@assert size(kernel_matrix,2) == n
	#@assert issymmetric(kernel_matrix)
	#@assert eigmin(kernel_matrix) > -1e-8
	#@assert length(folds) >= 2
	#@assert 0.0 < penalty_parameter_min && penalty_parameter_min < penalty_parameter_max
	#@assert penalty_parameter_num_steps >= 2
	
	best_accuracy::Float64 = -1.0
	best_penalty_parameter::Float64 = 0.0
	
	factor_penalty_parameter::Float64 = (penalty_parameter_max / penalty_parameter_min)^(1.0 / penalty_parameter_num_steps)
	
	penalty_parameter::Float64 = penalty_parameter_min
	
	while penalty_parameter <= penalty_parameter_max + 1e-8
			
		accuracy::Float64 = get_accuracy_cross_validation(n, y, folds, penalty_parameter, kernel_matrix)
		#@assert 0.0 <= accuracy && accuracy <= 1.0
		
		@printf "accuracy with penalty parameter C = %.17e: %.2f%%\n" penalty_parameter 100.0 * accuracy
		
		if accuracy > best_accuracy
			best_accuracy = accuracy
			best_penalty_parameter = penalty_parameter
		end
	
		penalty_parameter *= factor_penalty_parameter
	
	end
	
	#@assert best_penalty_parameter > 0.0
	
	return best_accuracy, best_penalty_parameter

end



function create_instance(
	file::String,
	n::Int64,
	l::Int64,
	X_data::Matrix{Float64},
	true_labels::Vector{Float64},
	penalty_parameters::Vector{Float64},
	kernel_type::String,
	gamma::Float64)

	#@assert n >= 1
	#@assert 1 <= l && l <= n
	#@assert size(X_data,1) == n
	#@assert length(true_labels) == n
	#@assert minimum(abs.(true_labels)) == 1.0
	#@assert maximum(abs.(true_labels)) == 1.0
	#@assert length(penalty_parameters) == n
	#@assert minimum(penalty_parameters) > 0.0
	#@assert kernel_type == "linear" || kernel_type == "rbf" && gamma > 0.0

	kernel_matrix::Matrix{Float64} = compute_kernel_matrix(n, X_data, kernel_type, gamma)
	#@assert issymmetric(kernel_matrix)
	#@assert eigmin(kernel_matrix) > -1e-8
	D::Matrix{Float64} = diagm(0.5 ./ penalty_parameters)
	K::Matrix{Float64} = kernel_matrix + D
	#@assert issymmetric(K)
	#@assert eigmin(K) > 0.0
	Kinv::Matrix{Float64} = inv(K)
	Kinv = 0.5 .* (Kinv .+ Kinv')
	C::Matrix{Float64} = 0.5 .* Kinv
	#@assert issymmetric(C)
	#@assert eigmin(C) > 0.0
	C_tilde::Matrix{Float64} = [C zeros(n,1); zeros(1,n) 0.0]
	Cinv::Matrix{Float64} = 2.0 .* K
	#@assert issymmetric(Cinv)
	#@assert eigmin(Cinv) > 0.0

	instance::S3vm_instance = S3vm_instance(
		basename(file),
		n,
		size(X_data,2),
		X_data,
		[true_labels[1:l]; zeros(Float64,n - l)],
		true_labels,
		penalty_parameters,
		kernel_matrix,
		D,
		K,
		Kinv,
		C,
		Cinv,
		C_tilde,
		kernel_type,
		gamma,
		Threads.Atomic{Float64}(+Inf),
		zeros(Float64,n),
		zeros(Float64,n),
		zeros(Float64,n),
		ReentrantLock()
	)

	return instance

end # create_instance



function run_tests(instances_file::String, parameters_file::String)

	@printf "--------------------------------------------------------------------------------\n"

	files::Vector{String} = []

	content = read(instances_file, String)

	@printf "Instances:\n\n"
	for line in split(content, '\n')
		line_modified = strip(split(line, '#', limit=2)[1]) # remove comments
        if length(line_modified) > 1
			push!(files, line_modified)
			@printf "%s\n" line_modified
		end
	end

	parameters::MyParameters = get_parameters(parameters_file)
	@printf "\n--------------------------------------------------------------------------------\n"
	@printf "Parameters:\n\n"
	for field_name in fieldnames(MyParameters)
		println("$field_name: $(getfield(parameters, field_name))")
	end

	for file::String in files

		@printf "\n--------------------------------------------------------------------------------\n"
		@printf "--------------------------------------------------------------------------------\n"
		@printf "--------------------------------------------------------------------------------\n"
		@printf "--------------------------------------------------------------------------------\n\n"

		# READ FILE/INSTANCE
		n::Int64, X_data::Matrix{Float64}, y::Vector{Float64}, l::Int64 = read_data(file, parameters.seed, parameters.perc)
		#@assert minimum(abs.(y)) == 1.0
		#@assert maximum(abs.(y)) == 1.0
		
		if parameters.use_everything_for_cross_validation
			# use all data and labels for cross-validation
			l_backup::Int64 = l
			l = n
		end

		# PRINT KERNEL ALIGNMENTS
		@printf "\n--------------------------------------------------------------------------------\n\n"
		@printf "Here are the kernel alignments on all labeled data points:\n"
		kernel_type_best_alignment::String, _, best_gamma::Float64, best_alignment::Float64 = find_best_kernel_alignment(l, X_data[1:l, :], y[1:l])
		#@assert kernel_type_best_alignment == "linear" || kernel_type_best_alignment == "rbf" && best_gamma > 0.0
		#@assert 0.0 <= best_alignment && best_alignment <= 1.0
		@printf "=> Best kernel according to alignment: %s" kernel_type_best_alignment
		if kernel_type_best_alignment == "rbf"
			@printf " (gamma = %.17e)" best_gamma
		end
		@printf " with alignment %.17e\n\n" best_alignment
		@printf "--------------------------------------------------------------------------------\n"
		
		# DO CROSS-VALIDATION
		folds::Vector{Tuple{Vector{Int64}, Vector{Int64}}} = collect(ScikitLearn.Skcore.StratifiedKFold(y[1:l], n_folds=parameters.kfold))
		if length(folds) != parameters.kfold
			error("Error in k-fold cross-validation. Maybe \"kfold\" is too large?\n")
		end
		for fold in folds
			#@assert length(fold[1]) + length(fold[2]) == l
			#@assert sort(union(fold[1], fold[2])) == 1:l
			#@assert length(fold[1]) >= 1
			#@assert length(fold[2]) >= 1
		end

		@printf "\nWe test the linear kernel:\n"
		kernel_matrix_linear::Matrix{Float64} = compute_kernel_matrix(n, X_data, "linear", nothing)
		best_accuracy_linear::Float64, best_penalty_parameter_linear::Float64 = find_best_penalty_parameter(l, y[1:l], folds, kernel_matrix_linear[1:l,1:l], parameters.penalty_parameter_min, parameters.penalty_parameter_max, parameters.penalty_parameter_num_steps)
		
		gamma::Float64 = 1.0 / size(X_data,2)
		@printf "\nWe test the rbf kernel with gamma = %.17e:\n" gamma
		kernel_matrix_rbf::Matrix{Float64} = compute_kernel_matrix(n, X_data, "rbf", gamma)
		best_accuracy_rbf::Float64, best_penalty_parameter_rbf::Float64 = find_best_penalty_parameter(l, y[1:l], folds, kernel_matrix_rbf[1:l,1:l], parameters.penalty_parameter_min, parameters.penalty_parameter_max, parameters.penalty_parameter_num_steps)
		
		best_kernel_type::String = (best_accuracy_linear > best_accuracy_rbf) ? "linear" : "rbf"
		best_penalty_parameter::Float64 = (best_accuracy_linear > best_accuracy_rbf) ? best_penalty_parameter_linear : best_penalty_parameter_rbf
		@printf "\nAccording to our 10-fold cross-validation, the best choice for the kernel is: %s with C = %.17e" best_kernel_type best_penalty_parameter
		
		if parameters.kernel_selection == "user"
			best_kernel_type = parameters.kernel_type
			gamma = parameters.gamma
			if gamma == 0.0
				gamma = 1.0 / size(X_data,2)
			end
			best_penalty_parameter = parameters.penalty_parameter
		end

		if parameters.use_everything_for_cross_validation
			l = l_backup
		end

		#@assert l < n
		penalty_parameters::Vector{Float64} = [best_penalty_parameter * ones(l); (1.0 * l / (n - l)) * parameters.penalty_factor_unlabeled * best_penalty_parameter * ones(n - l)]
		instance::S3vm_instance = create_instance(file, n, l, X_data, y, penalty_parameters, best_kernel_type, gamma)

		solve_s3vm_instance(instance, parameters)

		output_file = @sprintf "./output/%s_%d_%g" basename(file) parameters.seed parameters.perc
		output_data = [instance.X_data instance.y instance.true_labels instance.labeling instance.predictions]

		open(output_file, "w") do io
			writedlm(io, output_data, ' ')
			println(io, instance.kernel * " kernel")
			if instance.kernel == "rbf"
				println(io, @sprintf "gamma = %.16f" instance.gamma)
			end
			println(io, "penalty parameters:")
			writedlm(io, instance.penalty_parameters', ' ')
			if instance.kernel == "linear"
				println(io, "w:")
				w::Vector{Float64} = compute_w(instance)
				writedlm(io, w', ' ')
				for i = 1:n
					#@assert sign(w' * instance.X_data[i,:]) == instance.predictions[i]
				end
			end
			close(io)
		end
		
	end

end # run_tests()


