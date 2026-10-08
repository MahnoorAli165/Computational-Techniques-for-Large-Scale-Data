#!/usr/bin/env python3

import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes MurMurHash3 for the keys.'
    )
    parser.add_argument('key',nargs='*',help='key(s) to be hashed',type=str)
    parser.add_argument('-s','--seed',type=auto_int,default=0,help='seed value')
    args = parser.parse_args()

    seed = args.seed
    for key in args.key:
        h = murmur3_32(key,seed)
        print(f'{h:#010x}\t{key}')
        