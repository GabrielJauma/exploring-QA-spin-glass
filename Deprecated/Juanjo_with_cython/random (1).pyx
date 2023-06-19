import numpy as np
cimport numpy as cnp
cimport cython

cdef class RandomInt32Pool:

    def __init__(self, rng: np.random.Generator, copies, size, high, low: int = 0, memlimit=8000000):
        limit = memlimit // 4 # How many integers can we store max.
        copies = max(1, min(copies, limit // size))
        self.high = high
        self.low = low
        self.size = size
        self.copies = copies
        self.generator = rng
        self.refill()

    cdef void refill(RandomInt32Pool self) nogil:
        with gil:
            self.cache = self.generator.integers(self.low, self.high, (self.copies, self.size), dtype=np.int32)
        self.start = 0

    cdef inline cnp.int32_t[::1] pull(RandomInt32Pool self) nogil:
        cdef Py_ssize_t start = 0
        start = self.start
        if self.start >= self.copies:
            self.refill()
            start = self.start
        self.start += 1
        self.data = self.cache[start, :]
        return self.data

cdef class RandomDoublePool:

    def __init__(self, rng: np.random.Generator, size=None, memlimit=8000000):
        limit = memlimit // 8 # How many doubles can we store max.
        if size is None:
            size = memlimit
        self.size = min(size, limit)
        if rng is None:
            rng = np.random.default_rng()
        self.generator = rng
        self.refill()

    def fill(RandomDoublePool self, double value):
        self.data[:] = value
        self.ndx = 0

    cdef void refill(RandomDoublePool self) nogil:
        with gil:
            self.data = self.generator.random(self.size, dtype=np.double)
        self.ndx = 0

    cdef inline double pull(RandomDoublePool self) nogil:
        cdef:
            Py_ssize_t ndx = self.ndx
            double output = self.data[ndx]
        ndx += 1
        self.ndx = ndx
        if ndx >= self.size:
            self.refill()
        return output

    cdef inline double random(RandomDoublePool self) nogil:
        cdef:
            Py_ssize_t ndx = self.ndx
            double output = self.data[ndx]
        ndx += 1
        self.ndx = ndx
        if ndx >= self.size:
            self.refill()
        return output

import time

cdef unsigned NN = 312
cdef unsigned MM = 156
cdef cnp.uint64_t MATRIX_A = 0xB5026F5AA96619E9ULL
cdef cnp.uint64_t UM = 0xFFFFFFFF80000000ULL
cdef cnp.uint64_t LM = 0x7FFFFFFFULL

# Class version of https://raw.githubusercontent.com/ananswam/cython_random/master/cython_random.pyx

cdef class MersenneTwister:

    def __init__(MersenneTwister self, seed: None):
        if seed is None:
            seed = time.time()
        elif isinstance(seed, np.random.Generator):
            seed = seed.integers(0, 0xffffffffULL, 1)[0]
        else:
            seed = int(seed)
        self.seed(seed)

    cdef cnp.uint64_t seed(MersenneTwister self, cnp.uint64_t seed) nogil:
        cdef unsigned mti
        self.mt[0] = seed
        for mti in range(1,NN):
            self.mt[mti] = (6364136223846793005ULL * (self.mt[mti-1] ^ (self.mt[mti-1] >> 62)) + mti)
        self.mag01[0] = 0ULL
        self.mag01[1] = MATRIX_A
        self.mti = NN

    cdef cnp.uint64_t integer64(MersenneTwister self) nogil:
        cdef int i
        cdef cnp.uint64_t x

        if self.mti >= NN:
            for i in range(NN-MM):
                x = (self.mt[i]&UM) | (self.mt[i+1]&LM)
                self.mt[i] = self.mt[i+MM] ^ (x>>1) ^ self.mag01[int(x&1ULL)]

            for i in range(NN-MM, NN-1):
                x = (self.mt[i]&UM)|(self.mt[i+1]&LM)
                self.mt[i] = self.mt[i+(MM-NN)] ^ (x>>1) ^ self.mag01[int(x&1ULL)]

            x = (self.mt[NN-1]&UM)|(self.mt[0]&LM)
            self.mt[NN-1] = self.mt[MM-1] ^ (x>>1) ^ self.mag01[int(x&1ULL)]
            self.mti = 0

        x = self.mt[self.mti]
        self.mti += 1
        x ^= (x >> 29) & 0x5555555555555555ULL
        x ^= (x << 17) & 0x71D67FFFEDA60000ULL
        x ^= (x << 37) & 0xFFF7EEE000000000ULL
        x ^= (x >> 43);

        return x

    cdef cnp.uint32_t integer32(MersenneTwister self) nogil:
        return self.integer64() >> 10

    cdef void integers32(MersenneTwister self, cnp.int32_t min, cnp.int32_t max, cnp.int32_t[::1] output, Py_ssize_t n) nogil:
        cdef:
            Py_ssize_t i
            cnp.int64_t delta = <cnp.int64_t>(max - min)
        for i in range(n):
            output[i] = (self.integer64() % delta) + min 

    cdef double random(MersenneTwister self) nogil:
        """
        Generate a uniform random variable in [0,1]
        :return: (double) a random uniform number in [0,1]
        """
        return (self.integer64() >> 11) * (1.0/9007199254740992.0)

# Congruential random number generator

cdef class CRNG:

    def __init__(CRNG self, seed: None):
        if seed is None:
            seed = time.time()
        elif isinstance(seed, np.random.Generator):
            seed = seed.integers(0, 0xffffffffULL, 1)[0]
        else:
            seed = int(seed)
        self.seed(seed)

    cdef cnp.uint64_t seed(CRNG self, cnp.uint64_t seed) nogil:
        self.seed_ = seed
        self.mult_ = 6364136223846793005ULL
        self.add_  = 1442695040888963407ULL

    cdef cnp.uint64_t integer64(CRNG self) nogil:
        cdef cnp.uint64_t x = self.mult_ * self.seed_ + self.add_
        self.seed_ = x
        return x

    cdef cnp.uint32_t integer32(CRNG self) nogil:
        return self.integer64() >> 20

    cdef void integers32(CRNG self, cnp.int32_t min, cnp.int32_t max, cnp.int32_t[::1] output, Py_ssize_t n) nogil:
        cdef:
            Py_ssize_t i
            cnp.int64_t delta = <cnp.int64_t>(max - min)
        for i in range(n):
            output[i] = (self.integer32() % delta) + min 

    cdef double random(CRNG self) nogil:
        """
        Generate a uniform random variable in [0,1)
        :return: (double) a random uniform number in [0,1)
        """
        return (self.integer64() >> 11) * (1.0/9007199254740992.0)