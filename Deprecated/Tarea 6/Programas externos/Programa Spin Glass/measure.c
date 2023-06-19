#include "measure.h"

/* Measure M Funcs*/
void init_measure_m(MeasureM *measure_m) {
	measure_m->m = 0;
	measure_m->m2 = 0;
	measure_m->m4 = 0;
}
void make_measure_m(IsingData *ising, MeasureM *measure_m) {
	int i, j, k, mtemp = 0, L=ising->L;
	for (i = 0; i < L; i++)									
		for (j = 0; j < L; j++)	
			for (k = 0; k < L; k++)	
			/* Measure Magnetization of lattice*/
				mtemp +=ising->lattice_site[i][j][k];
	measure_m->m = measure_m->m + abs((double) mtemp) / ((double) L*L*L);
	measure_m->m2 =measure_m->m2 + pow(((double) mtemp) / ((double) L*L*L),2 );
	measure_m->m4 =measure_m->m4 + pow(((double) mtemp) / ((double) L*L*L),4 );
}

/* Measure E Funcs*/
void init_measure_E(MeasureE *measure_E) {
	measure_E->E = 0;
	measure_E->E2 = 0;
}
void make_measure_E(IsingData *ising, MeasureE *measure_E) {
	int left, right, up, in, out,  down, tot_spin, current_spin, i, j, k, L=ising->L;
	double tempE=0;
	for (i = 0; i < L; i++)									
		for (j = 0; j < L; j++)	
			for (k = 0; k < L; k++)	
			{
				out = (i == 0  ) ? L - 1 : i - 1;
				in  = (i==(L-1)) ? 0      : i+1;
				down = (j == 0  ) ? L - 1 : j - 1;
				up = (j==(L-1)) ? 0      : j+1;
				left = (k == 0  ) ? L - 1 : k - 1;
				right = (k==(L-1)) ? 0      : k+1;
				tot_spin = ising->x_interaction[out][j][k]*ising->lattice_site[out][j][k]
							+ising->x_interaction[i][j][k]*ising->lattice_site[in][j][k]
							+ising->y_interaction[i][down][k]*ising->lattice_site[i][down][k]
							+ising->y_interaction[i][j][k]*ising->lattice_site[i][up][k]
							+ising->z_interaction[i][j][left]*ising->lattice_site[i][j][left]
							+ising->z_interaction[i][j][k]*ising->lattice_site[i][j][right];
				current_spin = ising->lattice_site[i][j][k];
				tempE+= -0.5 * current_spin * tot_spin  -  ising->h*current_spin;
			}
	measure_E->E+= tempE / ((double) L*L*L);
	measure_E->E2+= pow(tempE / ((double) (L*L*L)),2) ;
}
