import math
import struct
import numpy as np

magic_constant = 0x7F000000
binary_magic_constants = 0b1111111000000000000000000000000
int_magic_constant = 127 << 24
magic_constant_str = '0b1111111000000000000000000000000'


# for i in range (1, 255):
#     print(i << 24)


def float_from_integer(integer):
    return struct.unpack('!f', struct.pack('!I', integer))[0]


def int_from_float(float):
    return struct.unpack('!I', struct.pack('!f', float))[0]


# f = 1.100000023841858
# i = 1066192077
#
# assert float_from_integer(i) == f
# assert int_from_float(f) == i

# f = 1 * 2 ** 50
# i = int_from_float(f)
# print(i)
# print(f"{i:032b}", f"{50:07b}")

# inv_i = magic_constant - i
# inv_f = float_from_integer(inv_i)
#
# print(inv_i)
# print(f"{inv_i:032b}", f"{127-50:07b}")
# print(inv_f, 1/f)


# f = 1.00
# int32bits = np.asarray(f, dtype=np.float32).view(np.int32).item()  # item() optional

# f = 1 * 2 ** (-50)
# print(np.float32(f))
# print(1/np.float32(f))

# for i in range(1, 100):
#     f = np.float32(float_from_integer(i))
#
#     inv_i = int_magic_constant - i
#     print(inv_i)
#     print(f"{inv_i:032b}")
#
#     inv_f = float_from_integer(inv_i)
#     print(inv_f, 1 / f)
#     print(math.log2(inv_f), math.log2(1 / f))

abs_error = 0
rel_error = 0
N = 100000
# for _ in range(1, N):
#     exp = np.random.randint(0, 100)
#     factor = 2 ** exp
#     u = np.float32(100 * np.random.random())
#     f = np.float32(u * factor)
#     # print(f, math.log2(f))
#     i = int_from_float(f)
#     # print(i, f"{i:032b}"[1:8])
#     assert float_from_integer(i) == f
#     # f = np.float32(float_from_integer(i))
#     #
#     inv_i = int_magic_constant - i
#     # print(inv_i)
#     # print(f"{inv_i:032b}")
#
#     inv_f = float_from_integer(inv_i)
#     abs_error += abs(inv_f - (1 / f))
#     rel_error += abs(inv_f - (1 / f))/f
#     # print(math.log2(inv_f), math.log2(1 / f))
#
# print("avg abs_error", abs_error/N)
# print("avg rel_error", rel_error/N)

for i in range(1, 255):
    shifted = i << 23
    # print(f"0b0'{i:08b}'"+"0"*23, end=", ")
    print(hex(shifted), end=", ")
    if not i % 10:
        print()
    # print(f"{shifted:b}")

# print(int_from_float(100))