#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include "MLP_Update/slim_net_wrap_rnd.cc"
#include "ga_multi_wrap_pthread.cc"
#include "xor_slim.cc"


void practise_net(double *weights,double * error,double **input);




int main(){
int i,j,k;
int wgtsize;
double **input;
int *config;
double error=6.0;
clock_t startTime, endTime;
config = new int[5];
input = new double*[5];

for(j=0;j<5;j++){
input[j] = new double[5];
}

for(i=0;i<5;i++){
for(j=0;j<5;j++){
input[i][j]=0.88475;
}}

config[0]=xor_net_init();
ga gaga(input,config,practise_net);

//gaga.getwgtsize(wgtsize);
while(1){
//run timer here

startTime = clock();
gaga.run();
endTime = clock();

cout << "It took " << (endTime -startTime) / (CLOCKS_PER_SEC /1000) << " ms. " << endl;

break;
//exit(1);
}

//create_energy_map(2,1,2);    //using last known config / topology


}

void practise_net(double *weights,double * error, double **input){

xor_net(weights,error);


//printf("hi");

}
