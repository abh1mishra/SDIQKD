function npa_model(list_vars,level;
    obj=0,
    min=true,
    op_eq = 0, 
    op_ge = 0,
    tr_eq = 0,
    tr_ge = 0,
    lvl_principal=0,
    cyclic=false,
    normalize=true,
    optimizer=Mosek.Optimizer)
    if lvl_principal==0
        ops, ops_principal = get_monomials(obj,level; op_eq = op_eq, op_ge = op_ge, tr_eq = tr_eq, tr_ge = tr_ge,list_vars=list_vars)
    else
        if !isempty(list_vars)
            ops_principal= mons_at_level(list_vars, level)
            ops= mons_at_level(list_vars, lvl_principal)
        else
            ops_principal = mons_at_level(sum(vcat([obj, op_ge..., op_eq...],[tr_ge[i][1] for i in 1:length(tr_ge)], [tr_eq[i][1] for i in 1:length(tr_eq)])), level)
            ops = ops_at_level(sum(vcat([obj, op_ge..., op_eq...],[tr_ge[i][1] for i in 1:length(tr_ge)], [tr_eq[i][1] for i in 1:length(tr_eq)])), level_principal)
        end
    end
    G=[]
    if op_eq!=0
        max_degree = maximum([degree(g) for g in op_eq])
        G=macaulay_grobner(Polynomial.(op_eq),max_degree)
    end
    println("ops_p_len",length(ops_principal))
    model=Model(optimizer)
    principal_moments_matrix, unique_mons, unique_vars = cyclic ? cyclic_npa_moments_block(ops_principal,model) : npa_moments_block(ops_principal,model;G=G)
    # Add the constraints for the principal moment matrix

    if tr_eq!=0
        for i in 1:length(tr_eq)
            tr_eq_p=0
            if cyclic
                for (m,c) in 1*tr_eq[i][1]
                    m1,m2= (cyclic_reduce(m),cyclic_reduce(m'))
                    m_i=findfirst(x->(x==m1 || x==m2),unique_mons)
                    tr_eq_p+=c*unique_vars[m_i]
                end
            else
                tr_eq_poly=real_rep(1*reduce_grobner(tr_eq[i][1],G))
                for (m,c) in tr_eq_poly
                    m_i=findfirst(x->x==m,unique_mons)
                    tr_eq_p+=c*unique_vars[m_i]
                end
            end
            @constraint(model, tr_eq_p == tr_eq[i][2])
        end
    end
    if tr_ge!=0
        for i in 1:length(tr_ge)
            tr_ge_p=0
            if cyclic
                for (m,c) in 1*tr_ge[i][1]
                    m1,m2= (cyclic_reduce(m),cyclic_reduce(m'))
                    m_i=findfirst(x->(x==m1 || x==m2),unique_mons)
                    tr_ge_p+=c*unique_vars[m_i]
                end
            else
                tr_ge_poly=real_rep(1*reduce_grobner(tr_ge[i][1],G))
                for (m,c) in tr_ge_poly
                    m_i=findfirst(x->x==m,unique_mons)
                    tr_ge_p+=c*unique_vars[m_i]
                end
            end
            @constraint(model, tr_ge_p >= tr_ge[i][2])
        end
    end
    if op_ge!=0
        for i in 1:length(op_ge)
            if cyclic
                cyclic_npa_moments_block(ops,model; cPoly=op_ge[i],unique_mons=unique_mons, unique_vars=unique_vars) 
            else
                npa_moments_block(ops,model; cPoly=op_ge[i], unique_mons=unique_mons, unique_vars=unique_vars,G=G) 
            end
        end
    end

    if normalize
        id_elem=one(first(ops_principal))
        if cyclic
            id_elem=cyclic_reduce(id_elem)
            @constraint(model,unique_vars[findfirst(check->check==id_elem,unique_mons)]==1.0)
        else
            @constraint(model,unique_vars[findfirst(check->check==id_elem,unique_mons)]==1.0)
        end
    end
    return model,unique_vars,unique_mons
end
