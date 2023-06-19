cimport numpy as cnp

cdef class RandomInt32Pool:

    cdef Py_ssize_t size, copies, start
    cdef cnp.int32_t low, high
    cdef cnp.int32_t[:,::1] cache
    cdef cnp.int32_t[::1] data
    cdef object generator

    cdef void refill(RandomInt32Pool rng) nogil
    cdef cnp.int32_t[::1] pull(RandomInt32Pool rng) nogil

cdef class RandomDoublePool:

    cdef Py_ssize_t size, ndx
    cdef double[::1] data
    cdef object generator

    cdef void refill(RandomDoublePool rng) nogil
    cdef inline double pull(RandomDoublePool rng) nogil
    cdef inline double random(RandomDoublePool rng) nogil

cdef class MersenneTwister:

    cdef cnp.uint64_t mt[312]
    cdef cnp.uint64_t mag01[2]
    cdef unsigned mti

    cdef cnp.uint64_t seed(MersenneTwister self, cnp.uint64_t seed) nogil
    cdef cnp.uint64_t integer64(MersenneTwister self) nogil
    cdef cnp.uint32_t integer32(MersenneTwister self) nogil
    cdef void integers32(MersenneTwister self, cnp.int32_t min, cnp.int32_t max, cnp.int32_t[::1] output, Py_ssize_t n) nogil
    cdef double random(MersenneTwister self) nogil


cdef class CRNG:

    cdef cnp.uint64_t seed_, mult_, add_

    cdef cnp.uint64_t seed(CRNG self, cnp.uint64_t seed) nogil
    cdef cnp.uint64_t integer64(CRNG self) nogil
    cdef cnp.uint32_t integer32(CRNG self) nogil
    cdef void integers32(CRNG self, cnp.int32_t min, cnp.int32_t max, cnp.int32_t[::1] output, Py_ssize_t n) nogil
    cdef double random(CRNG self) nogil
