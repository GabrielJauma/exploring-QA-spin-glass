#ifndef ISING_H
#define ISING_H


#include "rnd.h"
#include "stdio.h"
#include "math.h"
#include "stdlib.h"

struct IsingData 
{
	int ***lattice_site;
	int ***x_interaction;
	int ***y_interaction;
	int ***z_interaction;
	int L;
	double K,h;
};

typedef struct IsingData IsingData;

void init_ising(IsingData *ising,struct RndGen * rnd_gen);
void free_ising(IsingData *ising);
double calc_ratio(IsingData *ising, int *pos);
double measure_m(IsingData *ising); 
void dump_to_file_m(char * filename,double *m,double *h,int num_of_h );
void spin_configuration_dump_to_file(IsingData *ising, double t, int L);
void list_file_maker(double t_init,double t_final,double t_step); 
void set_h(IsingData *ising,double h);;


#endif
