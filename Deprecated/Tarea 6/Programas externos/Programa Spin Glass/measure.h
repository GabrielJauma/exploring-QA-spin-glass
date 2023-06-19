#ifndef MEASURE_H
#define MEASURE_H

#include "ising_data.h"

struct MeasureM {
	double m;
	double m2;
	double m4;
};
typedef struct MeasureM MeasureM;

struct MeasureE {
	double E;
	double E2;
};
typedef struct MeasureE MeasureE;


/* Measure M Funcs*/
void init_measure_m(MeasureM *measure_m);
void make_measure_m(IsingData *ising, MeasureM *measure_m);


/* Measure E Funcs*/
void init_measure_E(MeasureE *measure_E);
void make_measure_E(IsingData *ising, MeasureE *measure_E);


#endif
