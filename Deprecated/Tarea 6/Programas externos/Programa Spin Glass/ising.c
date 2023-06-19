#include "stdio.h"
#include "stdlib.h"
#include "rnd.h"
#include "math.h"
#include "ising_data.h"
#include "measure.h"

/* Performs a single mc move */
void mc_move(IsingData *ising, RndGen *rnd_gen) {
	int pos_arr[3];
	int *pos;
	int i;
	double ratio;
	/* Suggest new position */
	for (i = 0; i < 3; i++)
	{
		pos_arr[i] = floor(ising->L * genrand_real1(rnd_gen));
	}
	pos=&pos_arr[0];
	/* Calc Metropolis Ratio */
	ratio = calc_ratio(ising, pos);
	/* Metropolis reject accept */
	if (ratio > genrand_real1(rnd_gen))
		ising->lattice_site[pos[0]][pos[1]][pos[2]] = -ising->lattice_site[pos[0]][pos[1]][pos[2]];
}

void run_ising(IsingData *ising, int num_burn_runs, int num_runs,
		RndGen * rnd_gen, MeasureM * measure_m,  MeasureE * measure_E, int type_of_measurement)  {
	if (type_of_measurement==1)
		init_measure_m(measure_m);
	if (type_of_measurement==2)
		init_measure_E(measure_E);
	int i, j;
	/* Burn Runs */
	for (i = 0; i < num_burn_runs; i++)
		mc_move(ising, rnd_gen);
	/* Do num_runs mc steps */
	for (i = 0; i < num_runs; i++) 
		{
		for (j = 0; j < ising->L; j++)
			/* Monte carlo Move */
			mc_move(ising, rnd_gen);
		/* Measure Observables */
		if (type_of_measurement==1)
			make_measure_m(ising, measure_m);
		if (type_of_measurement==2)
			make_measure_E(ising, measure_E);
		}
}

int main(void) {
	/* Initialize random number generator */
	struct RndGen rnd_gen;
	init_rng(&rnd_gen);
	/* Get beta, h, Number of runs  */
	int L, num_runs, num_burn_runs, type_of_measurement, aviz_output;
	double t, t_final, t_step, h;
	printf("Enter L for LxLxL lattice. Suggested range [10 to 30] \n");
	scanf("%d", &(L));
	printf("Enter 1  for magnetization measurement \nEnter 2 for energy and heat capacity measurement\n");
	scanf("%d", &(type_of_measurement));
	printf("Generate file with spin positions for Aviz? Yes = 1 No = 0 \n");
	scanf("%d", &(aviz_output));	
	printf("Enter temperature range in units of J . Suggested range [3 to 5]\nTemperature from ");
	scanf("%lf", &(t));
	printf(" to ");
	scanf("%lf", &(t_final));
	printf(" in steps of  ");
	scanf("%lf", &(t_step));
	printf("\nEnter value for magnetic field h. Suggested h=0 \n");
	scanf("%lf", &(h));
	printf("Enter num of  burn runs. Suggested 10000\n");
	scanf("%d", &(num_burn_runs));
	printf("Enter num of runs. Suggested 100000 \n");
	scanf("%d", &(num_runs));
	
	/* Init Ising  */
	IsingData ising;
	ising.L=L;
	init_ising(&ising, &rnd_gen);
	/* Prepeare file dump */
	FILE *fpB;
	if (type_of_measurement==1)
			fpB = fopen("magnetization.txt",  "w");
	if (type_of_measurement==2)
			fpB = fopen("energy.txt",  "w");
	/* LIst file maker for Aviz*/
	if 	(aviz_output==1)	
	list_file_maker(t,t_final,t_step);		
	for (t ; t<t_final ; t+=t_step)
		{
		set_ising_params(&ising, t, h);
		/* Init Measure  */
		MeasureM measure_m;
		MeasureE measure_E;
		/* run ising*/
		run_ising(&ising, num_burn_runs, num_runs, &rnd_gen, &measure_m, &measure_E, type_of_measurement);
		/* Dump calculated data to file*/
		if (type_of_measurement==1)
			{
			fprintf(fpB, "temperature = %f [J], |m| = %f , Binder cumulant = %f \n", t, abs(measure_m.m) / ((double) num_runs), (1-(measure_m.m4 / ((double) num_runs))/(3 * pow(measure_m.m2 / ((double) num_runs),2))) );
			printf("temperature = %f [J], |m| = %f , Binder cumulant = %f \n", t, abs(measure_m.m) / ((double) num_runs) ,(1-(measure_m.m4 / ((double) num_runs))/(3 * pow(measure_m.m2 / ((double) num_runs),2))));
			}
		if (type_of_measurement==2)
			{
			double fm = measure_E.E / ((double) num_runs);
			double sm = (measure_E.E2 / ((double) num_runs));
			fprintf(fpB, "temperature = %f [J], Energy = %f [J], Heat capacity = %f \n", t, measure_E.E / ((double) num_runs), L*L*L*(sm-pow(fm,2)) / ((double) pow(t,2)));
			printf("temperature = %f [J], Energy = %f [J], Heat capacity = %f \n", t, measure_E.E / ((double) num_runs), L*L*L*(sm-pow(fm,2)) / ((double) pow(t,2)));
			}
			/* .xyz maker for Aviz*/
			if 	(aviz_output==1)
			spin_configuration_dump_to_file(&ising,t,L);
		}	
	fclose(fpB);
	/* Memory Free */
	free_ising(&ising);
	return 0;
}

