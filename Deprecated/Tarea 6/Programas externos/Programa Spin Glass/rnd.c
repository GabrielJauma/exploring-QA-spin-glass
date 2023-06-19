/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <stdio.h>
#include "rnd.h"

/* initializes mt[N] with a seed */
void init_genrand(struct RndGen * rndGen,unsigned long s)
{
    rndGen->mt[0]= s & 0xffffffffUL;
    for (rndGen->mti=1; rndGen->mti<N; rndGen->mti++) {
        rndGen->mt[rndGen->mti] = 
	    (1812433253UL * (rndGen->mt[rndGen->mti-1] ^ (rndGen->mt[rndGen->mti-1] >> 30)) + rndGen->mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        rndGen->mt[rndGen->mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(struct RndGen * rndGen,unsigned long init_key[], int key_length)
{
    int i, j, k;
    init_genrand(rndGen,19650218UL);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        rndGen->mt[i] = (rndGen->mt[i] ^ ((rndGen->mt[i-1] ^ (rndGen->mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        rndGen->mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { rndGen->mt[0] = rndGen->mt[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        rndGen->mt[i] = (rndGen->mt[i] ^ ((rndGen->mt[i-1] ^ (rndGen->mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        rndGen->mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { rndGen->mt[0] = rndGen->mt[N-1]; i=1; }
    }

    rndGen->mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

/* Initialize */ 
void init_rng (struct RndGen * rndGen) {
    rndGen->mti=N+1;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456};
    int length=4;
    init_by_array(rndGen,init, length);
}
/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(struct RndGen * rndGen)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (rndGen->mti >= N) { /* generate N words at one time */
        int kk;
        for (kk=0;kk<N-M;kk++) {
            y = (rndGen->mt[kk]&UPPER_MASK)|(rndGen->mt[kk+1]&LOWER_MASK);
            rndGen->mt[kk] = rndGen->mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (rndGen->mt[kk]&UPPER_MASK)|(rndGen->mt[kk+1]&LOWER_MASK);
            rndGen->mt[kk] = rndGen->mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (rndGen->mt[N-1]&UPPER_MASK)|(rndGen->mt[0]&LOWER_MASK);
        rndGen->mt[N-1] = rndGen->mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        rndGen->mti = 0;
    }
  
    y = rndGen->mt[rndGen->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(struct RndGen * rndGen)
{
    return genrand_int32(rndGen)*(1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* These real versions are due to Isaku Wada, 2002/01/09 added */


