using PCPOP,Mosek,MosekTools,JuMP
include("./three_meas_corr.jl")

# objective generator if key is extracted from Alice's side
function obj_i_term_A(ρ,Pi,αi,βi)
    res_αi=Pi[1]*ρ[1] + Pi[2]*ρ[2]
    res_βi=(Pi[1]+Pi[2])*(ρ[1]+ρ[2])
    res=-0.5*(αi*res_αi + βi*res_βi)
    return res
end

# objective generator if key is extracted from Bob's side
function obj_i_term_B(ρ,Pi,αi,βi,M)
    res_αi=Pi[1]*(ρ[1]+ρ[2])*M[1] + Pi[2]*(ρ[1]+ρ[2])*M[2]
    res_βi=(Pi[1]+Pi[2])*(ρ[1]+ρ[2])
    res=-0.5*(αi*res_αi + βi*res_βi)
    return res
end


"""
    Implements the (4,3) protocol.

    The below function computes the conditional entropy H(A|E) for a given guessing probability G and visibility v, while sweeping over the parameter eta.
    The results are saved to a specified file path :fname.
    Key is always extracted from Alice's side.
    It localizes the objective function to save resources.

    Arguments:
    - G: Guessing probability.
    - v: Visibility parameter.
    - fname: File path to save the results.
    - level: Level of the NPA hierarchy (default: "1+B*E").
    - step_size: Step size for the grid points (default: 0.1).
    - eta_start: Starting value for eta (default: 1).
    - Alice: Whether to extract key from Alice's inputs (default: true).
    - start_grid: Starting value for the grid (default: 0.0).
    - stop_grid: Stopping value for the grid (default: 1.0).
    - uniform: Whether to use uniform grid points (default: true).
    - optimizer: Optimizer to use (default: Mosek.Optimizer).
"""
function cond_entropy_four_local_eta(G,v,fname;level="1+B*E",step_size=0.1,eta_start=1,Alice=true,pure=false,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[3,0] E[2,0] BE[5,0] 
    # Bob and Eve's measurements commute
    @comms B E
    # Projector constraints for Bob and Eve's measurements
    Projector.([B;E])
    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]
    σ = BE[5]

    # Information constraints
    op_ge = [σ-0.25*ρ[x] for x in 1:4]

    # Additional constraints for mixed states
    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])

    # Sweep over eta values and compute the conditional entropy H(A|E) for each eta, saving the results to the specified file path.
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the current eta value
        η=plx[i]

        #  Probability constraints for the protocol
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:4 for y in 1:3]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:4])
        # Information constraints: Eve's guessing probability P_g^A ≤ G
        tr_ge = [ [-σ, -G]]

        obj_val=0
        if Alice
            # Compute the conditional entropy H(A|B) for the current eta and visibility where Bob doesn't bin. Used to determine when to stop the sweep (keyrate non-positive).
            cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)
            objs=[obj_i_term_A(ρ[1:2],E,α[i],β[i]) for i in 1:tot_points]
        else
            # Compute the conditional entropy H(B|A) for the current eta and visibility where Bob bins. Used to determine when to stop the sweep (keyrate non-positive).
            cond_entropy = conditional_entropy_four_B(η;v=v,bin=true)
            objs=[obj_i_term_B(ρ[1:2],E,α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points]
        end

        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
        end
        # Compute H(A|E) from the optimal value
        ply[i]=(1+obj_val)/log(2)
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.00001
            break
        end
    end
    return plx,ply
end


function cond_entropy_four_local_visibility(G,η,fname;level = "1+B*E",step_size=0.1,v_start=1,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    @pcmonoid M B[3,0] E[2,0] BE[5,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:4])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]
    σ = BE[5]

    op_ge = [σ-0.25*ρ[x] for x in 1:4]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end

    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)

        v=plx[i]

        tr_eq=[[1*ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:4 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ 1*ρ[x], 1] for x in 1:4])

        tr_ge = [ [-1*σ, -G]]

        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)
            objs=[obj_i_term_A(ρ[1:2],E,α[i],β[i]) for i in 1:tot_points]
        else
            cond_entropy = conditional_entropy_four_B(η;v=v,bin=true)
            objs=[obj_i_term_B(ρ[1:2],E,α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points]
        end

        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true)

        old_obj=0
        for obj in objs
            S = S+old_obj-obj
            model,V = model_new_obj(model,S,V,mons,LMI,-1)
            set_optimizer(model, optimizer)

            optimize!(model)
            ovi = objective_value(model)
            old_obj = obj
            obj_val += ovi
        end

        ply[i]=(1+obj_val)/log(2)

        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.00001
            break
        end
    end
    return plx,ply
end



"""
Implements the (6,3) protocol.
Key is always extracted from Alice's side and the states are assumed to be mixed.
"""
function cond_entropy_six_local_eta(G,v;level="1+B*E",fname = "./plots/Guess_prob/von-neumann/eta/adapt/mixed/6Alice_",step_size=0.1,eta_start=1.0,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[3,0] E[6,0] BE[7,0] 
    # Bob and Eve's measurements commute
    @comms B E
    # Projector constraints for Bob and Eve's measurements
    Projector.([B;E])
    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]

    # Information constraints
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    # Additional constraints for mixed states
    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])

    # Sweep over eta values and compute the conditional entropy H(A|E) for each eta, saving the results to the specified file path.
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    fname = "$(fname)$(level)_G=$(G).txt"

    for i in 1:length(plx)
        # grab the current eta value
        η=plx[i]

        # Probability constraints for the protocol
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:6 for y in 1:3]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
        # Information constraints: Eve's guessing probability P_g^A ≤ G
        tr_ge = [ [-σ, -G]]

        obj_val=0
        # Compute the conditional entropy H(A|B) for the current eta and visibility where Bob doesn't bin. Used to determine when to stop the sweep (keyrate non-positive).
        cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)

        # For 6-state protocol, we use three compatible state pairs for key generation:
        # Pair 1: states 1,4 (Z-basis) with measurement 1
        # Pair 2: states 2,5 (X-basis) with measurement 2  
        # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
        objs1=[obj_i_term_A([ρ[1],ρ[4]],E[1:2],α[j],β[j]) for j in 1:tot_points]  # First pair (1,4)
        objs2=[obj_i_term_A([ρ[2],ρ[5]],E[3:4],α[j],β[j]) for j in 1:tot_points]  # Second pair (2,5)
        objs3=[obj_i_term_A([ρ[3],ρ[6]],E[5:6],α[j],β[j]) for j in 1:tot_points]  # Third pair (3,6)
        objs = +(objs1, objs2, objs3)

        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = PCPOP.npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        old_obj=0
        obj_val=0
        
        for obj in objs
            S = S+old_obj-obj
            model,V = model_new_obj(model,S,V,mons,LMI,-1)
            set_optimizer(model, optimizer)

            optimize!(model)
            ovi = objective_value(model)
            old_obj = obj
            obj_val += ovi
        end

        ply[i]=(3+obj_val)/(3*log(2))
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.00001
            break
        end
    end
    return plx,ply
end


function cond_entropy_six_local_visibility(G,η;fname="./plots/Guess_prob/von-neumann/visibility/adapt/mixed/6Alice_",level="1+B*E",step_size=0.1,v_start=1.0,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    @pcmonoid M B[3,0] E[6,0] BE[7,0] 
    @comms B E
    Projector.([B;E])
    build(M)



    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]

    # Information constraints
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    # Additional constraints for mixed states
    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])

    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    fname = "$(fname)$(level)_G=$(G).txt"
    for i in 1:length(plx)

        v=plx[i]

        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:6 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
        tr_ge = [ [-σ, -G]]

        obj_val=0
        cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)
        # For 6-state protocol, we use three compatible state pairs for key generation:
        # Pair 1: states 1,4 (Z-basis) with measurement 1
        # Pair 2: states 2,5 (X-basis) with measurement 2  
        # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
        objs1=[obj_i_term_A([ρ[1],ρ[4]],E[1:2],α[j],β[j]) for j in 1:tot_points]  # First pair (1,4)
        objs2=[obj_i_term_A([ρ[2],ρ[5]],E[3:4],α[j],β[j]) for j in 1:tot_points]  # Second pair (2,5)
        objs3=[obj_i_term_A([ρ[3],ρ[6]],E[5:6],α[j],β[j]) for j in 1:tot_points]  # Third pair (3,6)
        objs = +(objs1, objs2, objs3)

        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = PCPOP.npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        old_obj=0
        obj_val=0
        
        for obj in objs
            S = S+old_obj-obj
            model,V = model_new_obj(model,S,V,mons,LMI,-1)
            set_optimizer(model, optimizer)

            optimize!(model)
            ovi = objective_value(model)
            old_obj = obj
            obj_val += ovi
        end
        
        ply[i]=(3+obj_val)/(3*log(2))

        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end

        if ply[i]-cond_entropy < 0.00001
            break
        end
    end
    return plx,ply
end