def AttentionOut(Q, K, V, N, d):
    X = gemm(Q, transpose(K)) / sqrt(d)
    P = zero(N, N)
    # can use online softmax to speed up
    for i in range(N):
        mx = -INF
        for j in range(N):
            mx = max(mx, X[i, j])
        
        sum_exp = 0
        for j in range(N):
            sum_exp += exp(X[i ,j] - mx)
        
        for j in range(N):
            P[i, j] = exp(X[i, j] - mx) / sum_exp
    O = gemm(P, V)
    return O


SRAMSIZE = 256 * 1024 
P = 106 # SM number
# Standard FlashAttention-style forward pseudocode:
# keep running row-wise max m, normalizer l, and unnormalized accumulator acc.
def flashattention2(Q, K, V, O, N, d):
    BN = calc_block_size(SRAMSIZE, d)
    rows_per_unit = ceil_div(N, P)
    scale = 1 / sqrt(d)

    parallel_for p in range(P):
        start_row = p * rows_per_unit
        end_row = min(start_row + rows_per_unit, N)

        for i in range(start_row, end_row, BN):
            current_BN = min(BN, end_row - i)
            Q_tile = load_to_sram(Q[i:i + current_BN, :])

            m_tile = fill(current_BN, -INF)
            l_tile = zero(current_BN)
            acc_tile = zero(current_BN, d)

            for j in range(0, N, BN):
                current_KN = min(BN, N - j)
                K_tile = load_to_sram(K[j:j + current_KN, :])
                V_tile = load_to_sram(V[j:j + current_KN, :])

                S_block = gemm(Q_tile, transpose(K_tile)) * scale

                for r in range(current_BN):
                    block_max = max(S_block[r, :current_KN])
                    new_max = max(m_tile[r], block_max)

                    exp_scale = exp(m_tile[r] - new_max)
                    probs_block = exp(S_block[r, :current_KN] - new_max)

                    l_tile[r] = l_tile[r] * exp_scale + sum(probs_block)
                    acc_tile[r, :] = acc_tile[r, :] * exp_scale + gemv(probs_block, V_tile)
                    m_tile[r] = new_max

            O_tile = zero(current_BN, d)
            for r in range(current_BN):
                O_tile[r, :] = acc_tile[r, :] / l_tile[r]

            store_to_gdram(O[i:i + current_BN, :], O_tile)





        
