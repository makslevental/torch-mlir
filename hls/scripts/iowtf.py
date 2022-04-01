import os
import numpy as np
import asyncio

MATRIX_DIM = 11
NUM_EL_MATRIX = MATRIX_DIM * MATRIX_DIM
DATA_TYPE = np.float32
DATA_BYTES = np.dtype(DATA_TYPE).itemsize

MAT_A = np.arange(0, NUM_EL_MATRIX, dtype=DATA_TYPE).reshape(MATRIX_DIM, MATRIX_DIM)
MAT_B = np.arange(0, NUM_EL_MATRIX, dtype=DATA_TYPE).reshape(MATRIX_DIM, MATRIX_DIM)
MAT_C = MAT_A @ MAT_B


async def to_device():
    xdma_axis_wr_data = os.open("/dev/xdma0_h2c_0", os.O_WRONLY)

    print(f"{MAT_A=}")

    buffer = MAT_A
    os.write(xdma_axis_wr_data, buffer.tobytes())


async def from_device():
    xdma_axis_rd_data = os.open("/dev/xdma0_c2h_0", os.O_RDONLY)

    buffer_size = 2 * DATA_BYTES
    data = os.read(xdma_axis_rd_data, buffer_size)
    output = np.frombuffer(data, dtype=DATA_TYPE)
    print(f"{output=}")
    print(np.frombuffer(os.read(xdma_axis_rd_data, 14 * 4), dtype=DATA_TYPE))


async def matmul():
    # don't flip the order!
    await to_device()
    await from_device()


asyncio.run(matmul())

# alternative method using struct.pack
# mat_a = [i for i in range(NUM_EL_MATRIX)]
# mat_b = [i for i in range(NUM_EL_MATRIX)]
# input_data = struct.pack(f"<{2*NUM_EL_MATRIX}{DTYPE}", *(mat_a + mat_b))
# output = np.array(struct.unpack(f"<{NUM_EL_MATRIX // DATA_BYTES}f", data))
# os.pwrite(xdma_axis_wr_data, buffer.tobytes(), 0)
# data = os.pread(xdma_axis_rd_data, buffer_size, 0)
