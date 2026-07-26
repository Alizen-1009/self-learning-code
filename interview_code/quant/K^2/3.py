P = 106 # SM num

def matmul(Q, K, O, M_dim, N_dim, K_dim, m, n, k):
    BM = math.ceil(M_dim, m)
    BN = math.ceil(N_dim, n)
    BK = math.ceil(K_dim, k)
    num_out_blocks = BM * BN

    parallel_for p in range(P):
        for task_id in range(p, num_out_blocks, P)
            block_i = task_id // BN
            block_j = task_id % BN

            row_start = block_i * m
            col_start = block_j * n

            C_sram = init_sram_buffer(shape(m, n), dtype=float16)
            for block_k in range(BK):
                k_start = block_k * k
                A_sram = load_2_sram(Q[row_start: row_start + m, k_start: k_start + k])
                KT_tmp = load_2_sram(K[col_start:col_start + n, k_start: k_start + k])
                B_sram = transpose(KT_tmp)

                # C += A * B
                C_sram = gemm(A_sram, B_sram, C_sram)
            store_2_gdram(O[row_start: row_start + m, col_start: col_start + n], C_sram)
