# Python program to convert a real value
# to IEEE 754 Floating Point Representation.

# Function to convert a
# fraction to binary form.
def binaryOfFraction(fraction):

    # Declaring an empty string
    # to store binary bits.
    binary = str()

    # Iterating through
    # fraction until it
    # becomes Zero.
    while fraction:

        # Multiplying fraction by 2.
        fraction *= 2

        # Storing Integer Part of
        # Fraction in int_part.
        if fraction >= 1:
            int_part = 1
            fraction -= 1
        else:
            int_part = 0

        # Adding int_part to binary
        # after every iteration.
        binary += str(int_part)

    # Returning the binary string.
    return binary


# Function to get sign bit,
# exp bits and mantissa bits,
# from given real no.
def floatingPoint(real_no):

    # Setting Sign bit
    # default to zero.
    sign_bit = 0

    # Sign bit will set to
    # 1 for negative no.
    if real_no < 0:
        sign_bit = 1

    # converting given no. to
    # absolute value as we have
    # already set the sign bit.
    real_no = abs(real_no)

    # Converting Integer Part
    # of Real no to Binary
    int_str = bin(int(real_no))[2:]

    # Function call to convert
    # Fraction part of real no
    # to Binary.
    fraction_str = binaryOfFraction(real_no - int(real_no))

    # Getting the index where
    # Bit was high for the first
    # Time in binary repres
    # of Integer part of real no.
    ind = int_str.index("1")

    # The Exponent is the no.
    # By which we have right
    # Shifted the decimal and
    # it is given below.
    # Also converting it to bias
    # exp by adding 127.
    exp_str = bin((len(int_str) - ind - 1) + 127)[2:]

    # getting mantissa string
    # By adding int_str and fraction_str.
    # the zeroes in MSB of int_str
    # have no significance so they
    # are ignored by slicing.
    mant_str = int_str[ind + 1 :] + fraction_str

    # Adding Zeroes in LSB of
    # mantissa string so as to make
    # it's length of 23 bits.
    mant_str = mant_str + ("0" * (23 - len(mant_str)))

    # Returning the sign, Exp
    # and Mantissa Bit strings.
    return sign_bit, exp_str, mant_str


# Function for converting decimal to binary
def float_bin(my_number, places=3):
    my_whole, my_dec = str(my_number).split(".")
    my_whole = int(my_whole)
    res = (str(bin(my_whole)) + ".").replace("0b", "")

    for x in range(places):
        my_dec = str("0.") + str(my_dec)
        temp = "%1.20f" % (float(my_dec) * 2)
        my_whole, my_dec = temp.split(".")
        res += my_whole
    return res


def IEEE754(n):
    # identifying whether the number
    # is positive or negative
    sign = 0
    if n < 0:
        sign = 1
        n = n * (-1)
    p = 30
    # convert float to binary
    dec = float_bin(n, places=p)

    dotPlace = dec.find(".")
    onePlace = dec.find("1")
    # finding the mantissa
    if onePlace > dotPlace:
        dec = dec.replace(".", "")
        onePlace -= 1
        dotPlace -= 1
    elif onePlace < dotPlace:
        dec = dec.replace(".", "")
        dotPlace -= 1
    mantissa = dec[onePlace + 1 :]

    # calculating the exponent(E)
    exponent = dotPlace - onePlace
    exponent_bits = exponent + 127

    # converting the exponent from
    # decimal to binary
    exponent_bits = bin(exponent_bits).replace("0b", "")

    mantissa = mantissa[0:23]

    # the IEEE754 notation in binary
    final = str(sign) + exponent_bits.zfill(8) + mantissa

    # convert the binary to hexadecimal
    hstr = "0x%0*X" % ((len(final) + 3) // 4, int(final, 2))
    return (hstr, final)


# Driver Code
if __name__ == "__main__":
    #
    # # Function call to get
    # # Sign, Exponent and
    # # Mantissa Bit Strings.
    # sign_bit, exp_str, mant_str = floatingPoint(6.25)
    #
    # # Final Floating point Representation.
    # ieee_32 = str(sign_bit) + "|" + exp_str + "|" + mant_str
    #
    # # Printing the ieee 32 representation.
    # print("IEEE 754 representation of 6.25 is :")
    # print(ieee_32)
    #
    # # Final Floating point Representation.
    # ieee_32 = str(sign_bit) + exp_str + mant_str
    #
    # # Printing the ieee 32 representation.
    # print("IEEE 754 representation of -2.250000 is :")
    # print(ieee_32)

    d = 28229.0
    hex, bin = IEEE754(d)
    print(d)
    print(hex)
    print(bin)
    print(int(bin, 2))
