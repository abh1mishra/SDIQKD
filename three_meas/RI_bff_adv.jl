using PCPOP,Mosek,MosekTools,JuMP
include("./three_meas_corr.jl")

function obj_i_term_A(ρ,Pi,αi,βi)
    res_αi=Pi[1]*ρ[1] + Pi[2]*ρ[2]
    res_βi=(Pi[1]+Pi[2])*(ρ[1]+ρ[2])
    res=-0.5*(αi*res_αi + βi*res_βi)
    return res
end

function obj_i_term_B(ρ,Pi,αi,βi,M)
    res_αi=Pi[1]*(ρ[1]+ρ[2])*M[1] + Pi[2]*(ρ[1]+ρ[2])*M[2]
    res_βi=(Pi[1]+Pi[2])*(ρ[1]+ρ[2])
    res=-0.5*(αi*res_αi + βi*res_βi)
    return res
end

function obj_i_term_ua_A(ρ,Pi,αi,βi)
    den=length(ρ)
    res_αi=Pi[1]*sum(ρ[i] for i in 1:floor(Int,den/2))+Pi[2]*sum(ρ[i] for i in floor(Int,den/2)+1:den)
    res_βi=(Pi[1] + Pi[2])*sum(ρ)
    res=-(1/den)*(αi*res_αi + βi*res_βi)
    return res
end

function obj_i_term_ua_B(ρ,Pi,αi,βi,M)
    den=length(ρ)
    den_h=floor(Int,den/2)
    res_αi=Pi[1]*sum((ρ[i]+ρ[i+den_h])*M[1,i] for i in 1:den_h) + Pi[2]*sum((ρ[i]+ρ[i+den_h])*M[2,i] for i in 1:den_h )
    res_βi=sum(ρ[i] for i in 1:den) * sum(Pi[i] for i in 1:length(Pi))
    res=-(1/den)*(αi*res_αi + βi*res_βi)
    return res
end

function cond_entropy_four_eta(G,level,fname;step_size=0.1,Alice=true,pure=false,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    tot=2*tot_points
    @pcmonoid M B[3,0] E[tot,0] BE[5,0] 
    # @pcmonoid M B[3,0] E[1,0] BE[5,0] 
    @comms B E
    # @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:4])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]
    σ = BE[5]
    PE=reshape(E,tot_points,2)
    # PE=[E[1];1-E[1]]
    op_ge = [σ-0.25*ρ[x] for x in 1:4]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end
    tr_ge = [ [-σ, -G]]

    if Alice
        obj=sum(obj_i_term_A(ρ[1:2],PE[i,:],α[i],β[i]) for i in 1:tot_points)
    else
        obj=sum(obj_i_term_B(ρ[1:2],PE[i,:],α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points)
    end

    plx = [1-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        η=plx[i]
        tr_eq = [ [ ρ[x], 1] for x in 1:4]
        tr_eq=vcat(tr_eq,[[ρ[x]*PB[1,y],prob_four(1, x, y,plx[η] ;bin=true)] for x in 1:4 for y in 1:3])


        # obj = (ρ[1]*PB[1,1]*PE[1] + ρ[1]*PB[2,1]*PE[1] + ρ[2]*PB[1,1]*PE[2] + ρ[2]*PB[2,1]*PE[2])/2
        ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,cyclic=true,normalize=false,list_vars=M.vertices,optimizer=optimizer)
        ply[i]=(1+ov)/log(2)
        return model
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
    end
    return plx,ply
end

function cond_entropy_prob_four_G(η,level,step_size,fname;Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    tot=2*tot_points
    @pcmonoid M B[3,0] E[tot,0] BE[5,0] 
    # @pcmonoid M B[3,0] E[1,0] BE[5,0] 
    @comms B E
    # @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:4])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]
    σ = BE[5]
    PE=reshape(E,tot_points,2)
    # PE=[E[1];1-E[1]]
    op_ge = [σ-0.25*ρ[x] for x in 1:4]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end
    tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η ;bin=true)] for x in 1:4 for y in 1:3]
    tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:4])

    if Alice
        obj=sum(obj_i_term_A(ρ[1:2],PE[i,:],α[i],β[i]) for i in 1:tot_points)
    else
        obj=sum(obj_i_term_B(ρ[1:2],PE[i,:],α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points)
    end

    plx=[0.5+i/100 for i in 0:50]
    ply=ones(length(plx))
    for i in 1:length(plx)
        G=plx[i]
        tr_ge = [ [-σ, -G]]
        # obj = (ρ[1]*PB[1,1]*PE[1] + ρ[1]*PB[2,1]*PE[1] + ρ[2]*PB[1,1]*PE[2] + ρ[2]*PB[2,1]*PE[2])/2
        ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,cyclic=true,normalize=false,list_vars=M.vertices,optimizer=optimizer)
        ply[i]=(1+ov)/log(2)
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
    end
    return plx,ply
end


function cond_entropy_four_local_eta(G,level,v,fname;step_size=0.1,eta_start=1,Alice=true,pure=false,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    
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
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        η=plx[i]
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:4 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:4])
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)
            objs=[obj_i_term_A(ρ[1:2],E,α[i],β[i]) for i in 1:tot_points]
        else
            cond_entropy = conditional_entropy_four_B(η;v=v,bin=true)
            objs=[obj_i_term_B(ρ[1:2],E,α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points]
        end
        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
        end
        ply[i]=(1+obj_val)/log(2)
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


function cond_entropy_four_local_visibility(G,level,fname;step_size=0.1,v_start=1,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    
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

    η=1.0

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
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


function cond_entropy_four_local_G(η,level,step_size,fname;G_start=0.5,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
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
    plx=[G_start+i/100 for i in 0:50]
    ply=ones(length(plx))
    for i in 1:length(plx)
        G=plx[i]
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η ;bin=true)] for x in 1:4 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:4])
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_four_A(η;bin=false)
            objs=[obj_i_term_A(ρ[1:2],E,α[i],β[i]) for i in 1:tot_points]
        else
            cond_entropy = conditional_entropy_four_B(η;bin=true)
            objs=[obj_i_term_B(ρ[1:2],E,α[i],β[i],[PB[1,1],PB[2,1]]) for i in 1:tot_points]
        end
        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,cyclic=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
            if primal_status(model) != MOI.FEASIBLE_POINT || dual_status(model) != MOI.FEASIBLE_POINT
                println("Warning: Optimization did not converge to optimal, status being primal=$(primal_status(model)), dual=$(dual_status(model))")
            end
        end
        ply[i]=(1+obj_val)/log(2)
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end

"""
    pl_guess_prob_six_local_eta(G, level, step_size, fname; Alice=true, pure=true, start_grid=0.0, stop_grid=1.0, bounded=false, uniform=true, optimizer=Mosek.Optimizer)

Computes Eve's guessing probability P_g^K for the 6-state QKD protocol as a function of detector efficiency η using localized objective functions.

# Arguments
- `G`: Upper bound on Eve's state guessing probability P_g^A ≤ G
- `level`: NPA hierarchy level
- `step_size`: Grid resolution for localization
- `fname`: Output filename to save results
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)
- `pure`: Whether Alice's states are pure (default: true)
- `start_grid`: Starting point for grid generation (default: 0.0)
- `stop_grid`: Ending point for grid generation (default: 1.0)
- `bounded`: Whether to use bounded optimization (default: false)
- `uniform`: Whether to use uniform grid spacing (default: true)
- `optimizer`: Optimization solver (default: Mosek.Optimizer)

# Returns
- `(plx, ply)`: Tuple of (η values, guessing probabilities P_g^K)

# Protocol Context
Implements the enhanced 6-state protocol with localized objective functions using binary functional form (BFF) approach.
The protocol uses 6 states at three different angles with ± variants:
- States 1,4: Z-basis (0, π)
- States 2,5: X-basis (π/2, 3π/2) 
- States 3,6: (Z+X)/√2-basis (π/4, 5π/4)

The localized approach decomposes the objective function into a sum of local terms that are optimized individually.
"""
function cond_entropy_six_local_eta(G,v;level="1+B*E",fname = "./plots/Guess_prob/von-neumann/eta/adapt/mixed/6Alice_",step_size=0.1,eta_start=1.0,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
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
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])

    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    fname = "$(fname)$(level)_G=$(G).txt"

    for i in 1:length(plx)
        η=plx[i]
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
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


function cond_entropy_six_local_visibility(G,η;fname="./plots/Guess_prob/von-neumann/visibility/adapt/mixed/6Alice_",level="1+B*E",step_size=0.1,v_start=1.0,pure=false,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    @pcmonoid M B[3,0] E[6,0] BE[7,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)



    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
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
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end



function cond_entropy_six_local_G(η,level,step_size,fname;start_G=1/3,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    @pcmonoid M B[3,0] E[2,0] BE[7,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;bin=true )] for x in 1:6 for y in 1:3]
    tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
    plx=[start_G+i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        G=plx[i]
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_six_A(η;bin=false)
            # For 6-state protocol, we use three compatible state pairs for key generation:
            # Pair 1: states 1,4 (Z-basis) with measurement 1
            # Pair 2: states 2,5 (X-basis) with measurement 2  
            # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
            objs1=[obj_i_term_A([ρ[1],ρ[4]],E,α[j],β[j]) for j in 1:tot_points]  # First pair (1,4)
            objs2=[obj_i_term_A([ρ[2],ρ[5]],E,α[j],β[j]) for j in 1:tot_points]  # Second pair (2,5)
            objs3=[obj_i_term_A([ρ[3],ρ[6]],E,α[j],β[j]) for j in 1:tot_points]  # Third pair (3,6)
            objs = vcat(objs1, objs2, objs3)
        else
            cond_entropy = conditional_entropy_six_B(η;bin=true)
            # Bob's key extraction using all three measurement bases
            objs1=[obj_i_term_B([ρ[1],ρ[4]],E,α[j],β[j],[PB[1,1],PB[2,1]]) for j in 1:tot_points]  # Z measurement
            objs2=[obj_i_term_B([ρ[2],ρ[5]],E,α[j],β[j],[PB[1,2],PB[2,2]]) for j in 1:tot_points]  # X measurement
            objs3=[obj_i_term_B([ρ[3],ρ[6]],E,α[j],β[j],[PB[1,3],PB[2,3]]) for j in 1:tot_points]  # (Z+X)/√2 measurement
            objs = vcat(objs1, objs2, objs3)
        end
        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,cyclic=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
        end
        ply[i]=(3+obj_val)/(3*log(2))
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


function cond_entropy_six_ua_local_eta(G,level,v,fname;step_size=0.1,eta_start=1.0,Alice=true,pure=false,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    @pcmonoid M B[3,0] E[2,0] BE[7,0]
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        η=plx[i]
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η ;v=v,bin=true)] for x in 1:6 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)
            # For 6-state protocol, we use three compatible state pairs for key generation:
            # Pair 1: states 1,4 (Z-basis) with measurement 1
            # Pair 2: states 2,5 (X-basis) with measurement 2  
            # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
            objs=[obj_i_term_ua_A(ρ,E,α[j],β[j]) for j in 1:tot_points] 
        else
            cond_entropy = conditional_entropy_six_B(η;v=v,bin=true)
            # Bob's key extraction using all three measurement bases
            objs=[obj_i_term_ua_B(ρ,E,α[j],β[j],PB) for j in 1:tot_points]  # Z measurement
        end
        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,cyclic=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
        end
        ply[i]=(1+obj_val)/(log(2))
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end

function cond_entropy_six_ua_local_visibility(G,level,fname;step_size=0.1,v_start=1.0,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    @pcmonoid M B[3,0] E[2,0] BE[7,0]
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)

    η=1.0

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        v=plx[i]
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η ;v=v,bin=true)] for x in 1:6 for y in 1:3]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)
            # For 6-state protocol, we use three compatible state pairs for key generation:
            # Pair 1: states 1,4 (Z-basis) with measurement 1
            # Pair 2: states 2,5 (X-basis) with measurement 2  
            # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
            objs=[obj_i_term_ua_A(ρ,E,α[j],β[j]) for j in 1:tot_points] 
        else
            cond_entropy = conditional_entropy_six_B(η;v=v,bin=true)
            # Bob's key extraction using all three measurement bases
            objs=[obj_i_term_ua_B(ρ,E,α[j],β[j],PB) for j in 1:tot_points]  # Z measurement
        end
        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = PCPOP.npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

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
        ply[i]=(1+obj_val)/(log(2))
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


function cond_entropy_six_ua_local_G(η,level,step_size,fname;start_G=1/3,Alice=true,pure=true,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)
    @pcmonoid M B[3,0] E[2,0] BE[7,0]
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:6]
    σ = BE[7]
    op_ge = [σ-1/6*ρ[x] for x in 1:6]

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η ;bin=true)] for x in 1:6 for y in 1:3]
    tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:6])
    plx=[start_G+i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        G=plx[i]
        tr_ge = [ [-σ, -G]]
        obj_val=0
        if Alice
            cond_entropy = conditional_entropy_six_A(η;bin=false)
            # For 6-state protocol, we use three compatible state pairs for key generation:
            # Pair 1: states 1,4 (Z-basis) with measurement 1
            # Pair 2: states 2,5 (X-basis) with measurement 2  
            # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
            objs=[obj_i_term_ua_A(ρ,E,α[j],β[j]) for j in 1:tot_points] 
        else
            cond_entropy = conditional_entropy_six_B(η;bin=true)
            # Bob's key extraction using all three measurement bases
            objs=[obj_i_term_ua_B(ρ,E,α[j],β[j],PB) for j in 1:tot_points]  # Z measurement
        end
        for obj in objs
            ov,model,dict,pri_mat=npa(obj,level;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,cyclic=true,normalize=false,optimizer= optimizer,list_vars=M.vertices)
            obj_val += ov
        end
        ply[i]=(1+obj_val)/(log(2))
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end