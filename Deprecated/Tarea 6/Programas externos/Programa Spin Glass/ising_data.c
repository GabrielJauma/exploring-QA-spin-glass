#include "ising_data.h"

void init_ising(IsingData *ising, struct RndGen * rnd_gen){
	int *space;
    int ***Arr3D;
    int i, j, k, y, z, L=ising->L;
    /*Initialize spin lattice*/
    {
    space = malloc(L * L * L * sizeof(int));  /* first we set aside space for the array itself */
    
    Arr3D = malloc(L * sizeof(int **));       		 /* next we allocate space of an array of pointers, each
															to eventually point to the first element of a
															2 dimensional array of pointers to pointers */

		for (z = 0; z < L; z++)								/* and for each of these we assign a pointer to a newly
																allocated array of pointers to a row */
			{
			Arr3D[z] = malloc(L * sizeof(int *));

															/* and for each space in this array we put a pointer to
															the first element of each row in the array space
															originally allocated */

			for (y = 0; y < L; y++)
				{
				Arr3D[z][y] = space + (z*(L * L) + y*L);
				}
			}
	ising->lattice_site=Arr3D;    

	for (i = 0; i < L; i++)									/*Fill with random spins*/
		for (j = 0; j < L; j++)	
			for (k = 0; k < L; k++)	
			{
				ising->lattice_site[i][j][k]=2 * round(genrand_real1(rnd_gen)) - 1;
			}
		}
		    	/*Initialize x interaction lattice*/
			{
			space = malloc(L * L * L * sizeof(int));  /* first we set aside space for the array itself */
    
    Arr3D = malloc(L * sizeof(int **));       		 /* next we allocate space of an array of pointers, each
															to eventually point to the first element of a
															2 dimensional array of pointers to pointers */

		for (z = 0; z < L; z++)								/* and for each of these we assign a pointer to a newly
																allocated array of pointers to a row */
			{
			Arr3D[z] = malloc(L * sizeof(int *));

															/* and for each space in this array we put a pointer to
															the first element of each row in the array space
															originally allocated */

			for (y = 0; y < L; y++)
				{
				Arr3D[z][y] = space + (z*(L * L) + y*L);
				}
			}
	ising->x_interaction=Arr3D;    

	for (i = 0; i < L; i++)									/*Fill with random x interaction*/
		for (j = 0; j < L; j++)	
			for (k = 0; k < L; k++)	
			{
				ising->x_interaction[i][j][k]=2 * round(genrand_real1(rnd_gen)) - 1;
			}
		}
				/*Initialize y interaction lattice*/
			{
			space = malloc(L * L * L * sizeof(int));  /* first we set aside space for the array itself */
    
    Arr3D = malloc(L * sizeof(int **));       		 /* next we allocate space of an array of pointers, each
															to eventually point to the first element of a
															2 dimensional array of pointers to pointers */

		for (z = 0; z < L; z++)								/* and for each of these we assign a pointer to a newly
																allocated array of pointers to a row */
			{
			Arr3D[z] = malloc(L * sizeof(int *));

															/* and for each space in this array we put a pointer to
															the first element of each row in the array space
															originally allocated */

			for (y = 0; y < L; y++)
				{
				Arr3D[z][y] = space + (z*(L * L) + y*L);
				}
			}
	ising->y_interaction=Arr3D;    

	for (i = 0; i < L; i++)									/*Fill with random x interaction*/
		for (j = 0; j < L; j++)	
			for (k = 0; k < L; k++)	
			{
				ising->y_interaction[i][j][k]=2 * round(genrand_real1(rnd_gen)) - 1;
			}
		}
				/*Initialize z interaction lattice*/
			{
			space = malloc(L * L * L * sizeof(int));  /* first we set aside space for the array itself */
    
			Arr3D = malloc(L * sizeof(int **));       		 /* next we allocate space of an array of pointers, each
															to eventually point to the first element of a
															2 dimensional array of pointers to pointers */

			for (z = 0; z < L; z++)								/* and for each of these we assign a pointer to a newly
																allocated array of pointers to a row */
			{
			Arr3D[z] = malloc(L * sizeof(int *));

															/* and for each space in this array we put a pointer to
															the first element of each row in the array space
															originally allocated */

			for (y = 0; y < L; y++)
				{
				Arr3D[z][y] = space + (z*(L * L) + y*L);
				}
			}
			ising->z_interaction=Arr3D;    

			for (i = 0; i < L; i++)									/*Fill with random x interaction*/
				for (j = 0; j < L; j++)	
					for (k = 0; k < L; k++)	
					{
						ising->z_interaction[i][j][k]=2 * round(genrand_real1(rnd_gen)) - 1;
					}
		}
			
}

void free_ising(IsingData *ising) {
	free(ising->lattice_site);
	free(ising->x_interaction);
	free(ising->y_interaction);
	free(ising->z_interaction);
}

double calc_ratio(IsingData *ising, int *pos) {
	int left, right, up, in, out,  down, tot_spin, current_spin, L=ising->L;
	double exparg, ratio; 
	out = (pos[0] == 0  ) ? L - 1 : pos[0] - 1;
	in  = (pos[0]==(L-1)) ? 0      : pos[0]+1;
	down = (pos[1] == 0  ) ? L - 1 : pos[1] - 1;
	up = (pos[1]==(L-1)) ? 0      : pos[1]+1;
	left = (pos[2] == 0  ) ? L - 1 : pos[2] - 1;
	right = (pos[2]==(L-1)) ? 0      : pos[2]+1;
	tot_spin = ising->x_interaction[out][pos[1]][pos[2]]*ising->lattice_site[out][pos[1]][pos[2]]
			  +ising->x_interaction[pos[0]][pos[1]][pos[2]]*ising->lattice_site[in][pos[1]][pos[2]]
	       	  +ising->y_interaction[pos[0]][down][pos[2]]*ising->lattice_site[pos[0]][down][pos[2]]
			  +ising->y_interaction[pos[0]][pos[1]][pos[2]]*ising->lattice_site[pos[0]][up][pos[2]]
			  +ising->z_interaction[pos[0]][pos[1]][left]*ising->lattice_site[pos[0]][pos[1]][left]
			  +ising->z_interaction[pos[0]][pos[1]][pos[2]]*ising->lattice_site[pos[0]][pos[1]][right];
	
	current_spin = ising->lattice_site[pos[0]][pos[1]][pos[2]];
	exparg = -ising->K * 2. * current_spin * tot_spin
			- ising->h * 2. * current_spin;
	ratio = exp(exparg);
	return ratio;
}

void dump_to_file_m(char * filename, double *m, double *h, int num_of_h) {
	FILE *fp;
	fp = fopen(filename, "w");
	int i;
	for (i = 0; i < num_of_h; i++) {
		fprintf(fp, "%f,%f \n", h[i], m[i]);
	}
	fclose(fp);
}
void spin_configuration_dump_to_file(IsingData *ising, double t, int L){
	FILE *spin_conf;
	int i,j,k;
	char filename[sizeof(double)+20];
	sprintf(filename, "spin_conf%f.xyz", t);
	spin_conf=fopen(filename, "w");
	fprintf(spin_conf, "%d\n#comment\n",L*L*L );
	for ( i = 0; i < L; i++)						
				for (j = 0; j < L; j++)	
					for (k = 0; k < L; k++)	
						fprintf(spin_conf, "Sp %d %d %d %d 0 0\n", i, j, k, ising->lattice_site[i][j][k]);
	fclose(spin_conf);
}
void list_file_maker(double t_init,double t_final,double t_step){
	FILE *list_file;
	list_file=fopen("list_file.txt", "w");
	for (t_init ; t_init<t_final ; t_init+=t_step)						
						fprintf(list_file, "spin_conf%f.xyz\n", t_init);
	fclose(list_file);
}


void set_ising_params(IsingData *ising, double t, double h) {
	ising->K = 1. / t;
	ising->h = h / t;
}

