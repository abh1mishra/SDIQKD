using PCPOP,Mosek,MosekTools,JuMP
include("./four_meas_corr.jl")

function obj_i_term_A(ρ,Pi,αi,βi)
    res_αi=Pi[1]*ρ[1] + Pi[2]*ρ[2]
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

# Key is always extracted from Alice's side
function cond_entropy_four_local_eta(G,v;fname="./plots/Guess_prob/von_neumann/eta/adapt/",level="1+B*E",step_size=0.1,eta_start=1,pure=false,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 4
    nmeas=4
    nbasis=2

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2*nbasis,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:nstates])  # Alice's 4 pure states
    end
    build(M)

    # extract the operators for the prtocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]  # Alice's 4 quantum states
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]
    
    # Additional constraints for non-pure states
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])
    end

    # parameter sweep over eta
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        η=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y) 
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)

        objs1=[obj_i_term_A(ρ[1:2],E[1:2],α[i],β[i]) for i in 1:tot_points]
        objs2=[obj_i_term_A(ρ[3:4],E[3:4],α[i],β[i]) for i in 1:tot_points]
        objs = objs1+objs2

        # grab the monomial basis and the model for the NPA hierarchy
        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        # iterate thr objectives and update the model for each one and optimize
        obj_val=0
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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(2+obj_val)/(2*log(2))

        fpath = fname*"v=$(round(v;digits=2))/4Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_four_local_visibility(G,η;fname="./plots/Guess_prob/von_neumann/visibility/adapt/",level="1+B*E",step_size=0.1,v_start=1,pure=false,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 4
    nmeas=4
    nbasis=2

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2*nbasis,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:nstates])  # Alice's 4 pure states
    end
    build(M)

    # extract the operators for the prtocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]  # Alice's 4 quantum states
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]
    
    # Additional constraints for non-pure states
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])
    end

    # parameter sweep over eta
    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        v=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y) 
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)

        objs1=[obj_i_term_A(ρ[1:2],E[1:2],α[i],β[i]) for i in 1:tot_points]
        objs2=[obj_i_term_A(ρ[3:4],E[3:4],α[i],β[i]) for i in 1:tot_points]
        objs = objs1+objs2

        # grab the monomial basis and the model for the NPA hierarchy
        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        # iterate thr objectives and update the model for each one and optimize
        obj_val=0
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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(2+obj_val)/(2*log(2))

        fpath = fname*"eta=$(round(η;digits=2))/4Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.000001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_four_ua_local_eta(G,v;fname="./plots/Guess_prob/von_neumann/eta/non-adapt/",level="1+B*E",step_size=0.1,eta_start=1,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 4
    nmeas=4

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    build(M)

    # extract the operators for the prtocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]  # Alice's 4 quantum states
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]
    
    # Additional constraints for non-pure states
    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])

    # parameter sweep over eta
    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        η=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y) 
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)
        ρ_ua = [ρ[1], ρ[3], ρ[2], ρ[4]]
        objs=[obj_i_term_ua_A(ρ_ua,E,α[i],β[i]) for i in 1:tot_points]

        # grab the monomial basis and the model for the NPA hierarchy
        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        # iterate thr objectives and update the model for each one and optimize
        obj_val=0
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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(1+obj_val)/(log(2))

        fpath = fname*"v=$(round(v;digits=2))/4Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_four_ua_local_visibility(G,η;fname="./plots/Guess_prob/von_neumann/visibility/non-adapt/",level="1+B*E",step_size=0.1,v_start=1,start_grid=0.0,stop_grid=1.0,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 4
    nmeas=4

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    build(M)

    # extract the operators for the prtocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]  # Alice's 4 quantum states
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]
    
    # Additional constraints for non-pure states
     op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])


    # parameter sweep over eta
    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        v=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y) 
        tr_eq=[[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        # Normalization constraints: Tr(ρₓ)=1
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_four_A(η;v=v,bin=false)

        ρ_ua = [ρ[1], ρ[3], ρ[2], ρ[4]]
        objs=[obj_i_term_ua_A(ρ_ua,E,α[i],β[i]) for i in 1:tot_points]

        # grab the monomial basis and the model for the NPA hierarchy
        ops,ops_principal=basis_gen(objs[1],level,[],op_ge,tr_eq,tr_ge,M.vertices,-1)
        model,S,V,mons,LMI = npa_dual(0,ops,ops_principal;op_ge=op_ge,tr_eq=tr_eq,tr_ge=tr_ge,min=true,tracial=true,normalize=false,change_objective=true,progress=true)

        # iterate thr objectives and update the model for each one and optimize
        obj_val=0
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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(1+obj_val)/(log(2))

        fpath = fname*"eta=$(round(η;digits=2))/4Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end

# Key is always extracted from Alice's side
function cond_entropy_six_local_eta(G,v;fname="./plots/Guess_prob/von_neumann/eta/adapt/",level="1+B*E",step_size=0.1,eta_start=1.0,pure=false,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 6
    nmeas=4
    nbasis=3

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2*nbasis,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:nstates])  # Alice's 6 pure states
    end
    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]

    # Additional constraints for non-pure states
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])
    end

    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        η=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y)
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(3+obj_val)/(3*log(2))

        fpath = fname*"v=$(round(v;digits=2))/6Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_six_local_visibility(G,η;fname="./plots/Guess_prob/von_neumann/visibility/adapt/",level="1+B*E",step_size=0.1,v_start=1.0,pure=false,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 6
    nmeas=4
    nbasis=3

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2*nbasis,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:nstates])  # Alice's 6 pure states
    end
    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]

    # Additional constraints for non-pure states
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])
    end

    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        v=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y)
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(3+obj_val)/(3*log(2))

        fpath = fname*"eta=$(round(η;digits=2))/6Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_six_ua_local_eta(G,v;fname="./plots/Guess_prob/von_neumann/eta/non-adapt/",level="1+B*E",step_size=0.1,eta_start=1.0,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 6
    nmeas=4

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])

    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]

    # Additional constraints for non-pure states

    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])

    plx=[eta_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        η=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y)
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)
        # For 6-state protocol, we use three compatible state pairs for key generation:
        # Pair 1: states 1,4 (Z-basis) with measurement 1
        # Pair 2: states 2,5 (X-basis) with measurement 2  
        # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
        objs=[obj_i_term_ua_A(ρ,E,α[j],β[j]) for j in 1:tot_points] 

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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(1+obj_val)/(log(2))

        fpath = fname*"v=$(round(v;digits=2))/6Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end


# Key is always extracted from Alice's side
function cond_entropy_six_ua_local_visibility(G,η;fname="./plots/Guess_prob/von_neumann/visibility/non-adapt/",level="1+B*E",step_size=0.1,v_start=1.0,start_grid=0.0,stop_grid=1.0,bounded=false,uniform=true,optimizer=Mosek.Optimizer)
    # protoocol parameters
    nstates = 6
    nmeas=4

    # Adv BFF grid points
    points = grid_points(step_size;start=start_grid,stop=stop_grid,uniform=uniform)
    α,β = grid_to_coeffs(points)
    tot_points=length(points)

    # Setup the monoid
    @pcmonoid M B[nmeas,0] E[2,0] BE[nstates+1,0] 
    @comms B E
    Projector.([B;E])
    build(M)

    # extract the operators for the protocol
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    ρ = BE[1:nstates]
    σ = BE[end]    # Auxiliary operator for guessing constraint

    # Information constraints
    op_ge = [σ-(1/nstates)*ρ[x] for x in 1:nstates]
    tr_ge = [ [-σ, -G]]

    # Additional constraints for non-pure states
    op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:nstates])


    plx=[v_start-i/100 for i in 0:100]
    ply=ones(length(plx))

    for i in 1:length(plx)
        # grab the eta value for this iteration
        v=plx[i]

        # Constraint on full prob. distribution: p(b=1|x,y)
        tr_eq=[[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:nstates for y in 1:nmeas]
        tr_eq = vcat(tr_eq,[ [ ρ[x], 1] for x in 1:nstates])

        cond_entropy = conditional_entropy_six_A(η;v=v,bin=false)
        # For 6-state protocol, we use three compatible state pairs for key generation:
        # Pair 1: states 1,4 (Z-basis) with measurement 1
        # Pair 2: states 2,5 (X-basis) with measurement 2  
        # Pair 3: states 3,6 ((Z+X)/√2-basis) with measurement 3
        objs=[obj_i_term_ua_A(ρ,E,α[j],β[j]) for j in 1:tot_points] 

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

        # Compute the H(A|E) from the objective value and store it in ply
        ply[i]=(1+obj_val)/(log(2))

        fpath = fname*"eta=$(round(η;digits=2))/6Alice_$(level)_G=$(round(G;digits=2)).txt"
        mkpath(dirname(fpath))
        open(fpath,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]-cond_entropy < 0.001
            break
        end
    end
    return plx,ply
end