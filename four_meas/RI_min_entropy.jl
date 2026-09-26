using PCPOP,Mosek,MosekTools,JuMP
include("./four_meas_corr.jl")

function pl_guess_prob_four_vis(G,level,fname;pure=false,adapt=true,start_v=1)
    if !adapt
        # Non-adaptive Eve: single measurement strategy
        @pcmonoid M B[4,0] E[1,0] BE[5,0]
    else
        # Adaptive Eve: can choose measurement based on protocol round
        @pcmonoid M B[4,0] E[2,0] BE[5,0]
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:4])  # Alice's 4 pure states
    end
    build(M)

    ρ = BE[1:4]  # Alice's 6 quantum states
    σ = BE[5]    # Auxiliary operator for guessing constraint
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    if !adapt
        PE=[E[1];1-E[1]]  # Eve's single measurement
    else
        PE=[E[1] E[2]; 1-E[1] 1-E[2]]  # Eve's 2 adaptive measurements
    end
    η=1.0
    # Guessing probability constraint: σ ≥ (1/4)ρₓ ∀x
    op_ge = [σ-1/4*ρ[x] for x in 1:4]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end
    

    plx = [start_v-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        v=plx[i]
        tr_eq = [ [ ρ[x], 1] for x in 1:4]
        tr_ge = [ [-σ, -G]]
        tr_eq = vcat(tr_eq, [[ρ[x]*PB[1,y],prob_four(1, x, y,η;v=v,bin=true )] for x in 1:4 for y in 1:4])
        if !adapt
            # Non-adaptive Eve case
            cond_entropy = conditional_entropy_four_A(η;v=v,bin=true)

            # Alice's key extraction
            obj2A = (ρ[1]*PE[1] + ρ[2]*PE[2] + ρ[3]*PE[1] + ρ[4]*PE[2])/4
        else
            cond_entropy = conditional_entropy_four_A(η;v=v,bin=true)
            # key from 2 observables from Alice's side
            obj2A = (ρ[1]*PE[1,1] + ρ[2]*PE[2,1] + ρ[3]*PE[1,2] + ρ[4]*PE[2,2])/4
        end


        ov,model,_ = pcpop(obj2A,level;tr_eq=tr_eq,op_ge=op_ge,tr_ge=tr_ge,min=false,normalize=false,tracial=true,progress=true)
        ply[i] = min_entropy(ov)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]-cond_entropy<1e-6
            break
        end
    end
    return plx,ply
end


function pl_guess_prob_six_vis(G,level,fname;pure=false,adapt=true,start_v=1)
    if !adapt
        # Non-adaptive Eve: single measurement strategy
        @pcmonoid M B[4,0] E[1,0] BE[7,0]
    else
        # Adaptive Eve: can choose measurement based on protocol round
        @pcmonoid M B[4,0] E[3,0] BE[7,0]
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])  # Alice's 6 pure states
    end
    build(M)

    ρ = BE[1:6]  # Alice's 6 quantum states
    σ = BE[7]    # Auxiliary operator for guessing constraint
    PB=[B[1] B[2] B[3] B[4];1-B[1] 1-B[2] 1-B[3] 1-B[4]]  # Bob's 4 measurements
    if !adapt
        PE=[E[1];1-E[1]]  # Eve's single measurement
    else
        PE=[E[1] E[2] E[3]; 1-E[1] 1-E[2] 1-E[3]]  # Eve's 2 adaptive measurements
    end
    η=1.0
    # Guessing probability constraint: σ ≥ (1/4)ρₓ ∀x
    op_ge = [σ-1/6*ρ[x] for x in 1:6]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    

    plx = [start_v-i/100 for i in 0:100]
    ply=ones(length(plx))
    for i in 1:length(plx)
        v=plx[i]
        tr_eq = [ [ ρ[x], 1] for x in 1:6]
        tr_ge = [ [-σ, -G]]
        tr_eq = vcat(tr_eq, [[ρ[x]*PB[1,y],prob_six(1, x, y,η;v=v,bin=true )] for x in 1:6 for y in 1:4])
        if !adapt
            # Non-adaptive Eve case
            cond_entropy = conditional_entropy_six_A(η;v=v,bin=true)

            # Alice's key extraction
            obj3A = (ρ[1]*PE[1] + ρ[4]*PE[2] + ρ[2]*PE[1] + ρ[5]*PE[2] + ρ[3]*PE[1] + ρ[6]*PE[2])/6
        else
            cond_entropy = conditional_entropy_six_A(η;v=v,bin=true)
            # key from 2 observables from Alice's side
            obj3A = (ρ[1]*PE[1,1] + ρ[4]*PE[2,1] + ρ[2]*PE[1,2] + ρ[5]*PE[2,2] + ρ[3]*PE[1,3] + ρ[6]*PE[2,3])/6
        end


        ov,model,_ = pcpop(obj3A,level;tr_eq=tr_eq,op_ge=op_ge,tr_ge=tr_ge,min=false,normalize=false,tracial=true,progress=true)
        ply[i] = min_entropy(ov)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]-cond_entropy<1e-6
            break
        end
    end
    return plx,ply
end