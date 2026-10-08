#!/usr/bin/env python3

import argparse
import sys
def rol32(x,k):
    """Auxiliary function (left rotation for 32-bit words)"""
    return ((x << k) | (x >> (32-k))) & 0xffffffff

def murmur3_32(key, seed):
    """Computes the 32-bit murmur3 hash"""
    key = key.encode('utf-8')
    length_key = len(key)
    

    # derived from murmur3 original implementation
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xe6546b64
    
    #initialization of the hash value
    hash1 = seed & 0xffffffff
    

    for index in range(0, length_key - (length_key % 4), 4):


        # idea: tkae 4 separate bytes and combine them into a 32-bit word 
        k1 = key[index] | (key[index+1] << 8) | (key[index+2] << 16) | (key[index+3] << 24)

        # take number and mix it up with some operations (multiplication, rotation, xor)
        # the & 0xffffffff is to ensure we are working with 32-bit words and avoid overflow issues in Python
        k1 = (k1 * c1) & 0xffffffff
        k1 = rol32(k1,r1)
        k1 = (k1 * c2) & 0xffffffff

        hash1 = hash1 ^ k1
        hash1 = rol32(hash1,r2)
        hash1 = ((hash1*m) + n) & 0xffffffff    


    # handle the remaining bytes (if the length of the key is not a multiple of 4)
    remaining_bytes = length_key % 4
    if remaining_bytes > 0:
        index = (length_key//4)*4
        k1 = 0
        if remaining_bytes >= 3:
            k1 |= key[index+2] << 16
        if remaining_bytes >= 2:
            k1 |= key[index+1] << 8
        if remaining_bytes >= 1:
            k1 |= key[index]

        k1 = (k1 * c1) & 0xffffffff
        k1 = rol32(k1,r1)
        k1 = (k1 * c2) & 0xffffffff
        hash1 ^= k1
    # finalization step to mix the bits of the hash value
    hash1 ^= length_key
    hash1 ^= (hash1 >> 16)
    hash1 = (hash1 * 0x85ebca6b) & 0xffffffff
    hash1 ^= (hash1 >> 13)
    hash1 = (hash1 * 0xc2b2ae35) & 0xffffffff
    hash1 ^= (hash1 >> 16)      

    return hash1 & 0xffffffff
def auto_int(x):
    """Auxiliary function to help convert e.g. hex integers"""
    return int(x,0)

def dlog2(n):
    """Auxiliary function to compute discrete base2 logarithm"""
    return n.bit_length() - 1

def rho(n):
    """Given a 32-bit number n, return the 1-based position of the first
    1-bit"""
    hashed_value = n & 0xffffffff
    # we check the bits from the left to the right 
    for i in range(31, -1, -1):
        if (hashed_value & (1 << i)) != 0:
            return 32 - i
    return 0


def compute_jr(key,seed,log2m):
    """hash the string key with murmur3_32, using the given seed
    then take the **least significant** log2(m) bits as j
    then compute the rho value **from the left**

    E.g., if m = 1024 and we compute hash value 0x70ffec73
    or 0b01110000111111111110110001110011
    then j = 0b0001110011 = 115
         r = 2
         since the 2nd digit of 0111000011111111111011 is the first 1

    Return a tuple (j,r) of integers
    """
    h = murmur3_32(key,seed)

    j = ~(0xffffffff << log2m) & h
    r = rho(h)
    return j, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes (j,r) pairs for input integers.'
    )
    parser.add_argument('key',nargs='*',help='key(s) to be hashed',type=str)
    parser.add_argument('-s','--seed',type=auto_int,default=0,help='seed value')
    parser.add_argument('-m','--num-registers',type=int,required=True,
                            help=('Number of registers (must be a power of two)'))
    args = parser.parse_args()

    seed = args.seed
    m = args.num_registers
    if m <= 0 or (m&(m-1)) != 0:
        sys.stderr.write(f'{sys.argv[0]}: m must be a positive power of 2\n')
        quit(1)

    log2m = dlog2(m)

    for key in args.key:
        h = murmur3_32(key,seed)

        j, r = compute_jr(key,seed,log2m)

        print(f'{key}\t{j}\t{r}')
        