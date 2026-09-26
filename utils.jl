function conditional_entropy(joint_prob::Array{Float64,2})
    nA, nB = size(joint_prob)
    H = 0.0

    # Compute P(B=b)
    P_B = sum(joint_prob, dims=1)

    for b in 1:nB
        if P_B[b] == 0.0
            continue  # skip B=b if its probability is 0
        end
        for a in 1:nA
            if joint_prob[a,b] == 0.0
                continue  # skip terms with P(a,b)=0
            end
            P_a_given_b = joint_prob[a,b] / P_B[b]
            H -= joint_prob[a,b] * log2(P_a_given_b)
        end
    end
    return H
end



function min_entropy(p)
    return -log2(max(p,1-p))
end

function xlog2x(x)
    if x<0 || x>1
        0
    elseif x==0 || x==1
        0
    else
        -x*log2(x)
    end
end

function hshannon(prob)
    sum(xlog2x.(prob))
end

function hbin(x)
    hshannon((x,1-x))
end



#################

# Grid points for BFF_adv


function grid_points(epsilon;start=0,stop=1,uniform=true)
    points=0
    if uniform
        points = collect(range(start, stop, step=epsilon))
    else
        t = start
        if start == 0
            t+= 0.0001 # to avoid division by zero in the next step
        end
        points = [t]
        while true
            next_t = t + sqrt(epsilon * t)
            if next_t >= stop
                break
            end
            push!(points, next_t)
            t = next_t
        end
    end
    if stop-points[end] > 0.001
        push!(points, stop) # ensure the last point is exactly 'stop'
    end
    return points
end

function grid_to_coeffs(points)
    if iszero(points[1])
        t= points[2:end]
    else
        t= points
    end
    alpha= zeros(length(t))
    beta= zeros(length(t))
    alpha[1]=-((1+t[1]/(t[2]-t[1]))*log(t[2]/t[1])-1 )
    beta[1]=t[1]*((1+t[1]/(t[2]-t[1]))*log(t[2]/t[1])-1 )
    alpha[end]=-(1-(t[end-1]/(t[end]-t[end-1]))*log(t[end]/t[end-1]))
    beta[end]=t[end]*(1-(t[end-1]/(t[end]-t[end-1]))*log(t[end]/t[end-1]))
    for i in 2:length(t)-1
        alpha[i]=-( (1+t[i]/(t[i+1]-t[i]))*log(t[i+1]/t[i]) - log(t[i]/t[i-1])*t[i-1]/(t[i]-t[i-1]) )
        beta[i]=t[i]*( (1+t[i]/(t[i+1]-t[i]))*log(t[i+1]/t[i]) - log(t[i]/t[i-1])*t[i-1]/(t[i]-t[i-1]) )
    end
    if iszero(points[1])
        return [-1;alpha], [t[1];beta]
    end
    return alpha, beta

end