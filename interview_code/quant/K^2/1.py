P  = 106 # SM_SIZE
M = 256 * 1024 # 256KB

def calc_best_block_size(M, n, d, k, N, dtype = "float16"):
    """
    consider Qtile, K_tile, S_tile, TopK_val, TopK_idx
    this is a dummy func
    BN, Bn affect effency extremely, need to tune
    """
    #QByte = BN * 2 * d
    #KByte = Bn * 2 * d
    #TopkByte = BN * k * (4 + 2)
    #SByte = Bn * BN * 2
    scale = N // n   
    return BN, Bn

def topk_stable(topk_val_buffer, topk_idx_buffer, new_val, new_idx k):
    for m in range(k):
        if new_val > topk_val_buffer[m] or (new_val == topk_val_buffer[m] and new_idx < topk_idx_buffer[m]):
            shift_right(topk_idx_buffer, start = m, end = k - 1)
            shift_right(topk_val_buffer, start = m, end = k - 1)
            topk_idx_buffer[m] = new_idx
            topk_val_buffer[m] = new_val
            break;
    

def compute_tile_topk(Q, K, O, N, n, d, k, M):
    BN, Bn = calc_best_block_size(M, n, d, k, N)
    rows_per_unit = N // P
    parallel_for p in range(P):
        start_row = p * rows_per_unit
        end_row = min(start_row + rows_per_unit, N)
        for i in range(start_row, end_row, BN):
            Q_tile = load_2_sram(Q[i: i +BN, :])
            local_topk_val_buffer = init_sram_buffer(shape[current_BN, k], val = -INF, dtype =float16)
            local_topk_idx_buffer = init_sram_buffer(shape[current_BN, k], val = 0, dtype =int32)
            current_BN = min(BN, end_row - i)
            for j in range(0, n, Bn):
                current_Bn = min(Bn, n - j)
                K_tile = load_2_sram(K[j: j + current_Bn, :])
                S_tile = gemm(Q_tile, transpose(K_tile))
                for r in range(current_BN):
                    for c in range(current_Bn):
                        val = S_tile[r, c]
                        idx = j + c
                        topk_stable(local_topk_val_buffer, local_topk_idx_buffer, val, idx, k)
            store_2_gdram(O[i:i +current_BN, :], local_topk_buffer)

                    

