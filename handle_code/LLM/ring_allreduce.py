from mpi4py import MPI


def ring_allreduce(data, comm=MPI.COMM_WORLD):
    """逐元素求和，返回所有 rank 的总和列表，不修改输入。

    所有 rank 必须传入相同长度的数值列表，并一起调用此函数。
    同一通信域中不要并发使用消息 tag 0、1。
    """
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    left = (rank - 1) % ranks
    right = (rank + 1) % ranks

    # 切成 ranks 块；不能整除时，前面的块各多一个元素。
    chunks = []
    start = 0
    for i in range(ranks):
        size = len(data) // ranks + (i < len(data) % ranks)
        chunks.append(list(data[start:start + size]))
        start += size

    # 1. Reduce-scatter：向右发送，从左接收并累加。
    for step in range(ranks - 1):
        send_id = (rank - step) % ranks
        recv_id = (rank - step - 1) % ranks
        received = comm.sendrecv(
            chunks[send_id], dest=right, source=left,
            sendtag=0, recvtag=0,
        )
        chunks[recv_id] = [
            local + remote for local, remote in zip(chunks[recv_id], received)
        ]

    # 此时 rank r 持有块 (r + 1) % ranks 的完整求和结果。
    # 2. All-gather：向右传递完整结果，只复制，不再累加。
    for step in range(ranks - 1):
        send_id = (rank + 1 - step) % ranks
        recv_id = (rank - step) % ranks
        chunks[recv_id] = comm.sendrecv(
            chunks[send_id], dest=right, source=left,
            sendtag=1, recvtag=1,
        )

    return [value for chunk in chunks for value in chunk]
