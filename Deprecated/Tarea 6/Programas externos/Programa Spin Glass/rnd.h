/* Period parameters */  

#ifndef ADD_H_RND
#define ADD_H_RND

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

struct RndGen {
	unsigned long mt[N]; /* the array for the state vector  */
	int mti;
};

typedef struct RndGen RndGen;

void init_genrand(RndGen * rndGen,unsigned long s);
void init_by_array(RndGen * rndGen,unsigned long init_key[], int key_length);
/* Initialize Random Number Generator */
void init_rng ( RndGen * rndGen);
/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(RndGen * rndGen);
/* Generate Uniform [0.1] */
double genrand_real1( RndGen * rndGen);

#endif
