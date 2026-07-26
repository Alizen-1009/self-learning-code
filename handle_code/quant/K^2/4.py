def distrubute_data(P, L, drop_last=False, shuffle=False):
    idx = list(range(L))

    if shuffle:
        random.shuffle(idx)

    if drop_last:
        total_size = (L // P) * P
        idx = idx[:total_size]
    else:
        num_pre_process = math.ceil(L / P)
        total_size = num_pre_process * P
        pad_size = total_size - L
        idx = idx + [-1] * pad_size
    
    results = {}
    for rank in range(P):
        results[f"P[{rank}]"] = idx[rank::P]

    return results