from BitVector import BitVector


def int_to_bit_vector(number: int, number_of_bits: int) -> BitVector:
    number_as_str_binary = bin(number)[2:]

    number_as_str_binary = '0'*(number_of_bits - len(number_as_str_binary)) + number_as_str_binary
    number_as_str_binary = number_as_str_binary[-number_of_bits:]

    return BitVector(bitlist=[1 if digit == '1' else 0 for digit in number_as_str_binary])
