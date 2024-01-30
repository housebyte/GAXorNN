#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;

void prepdata(double **data,int max_it){

int select;
double xord[4][3];
xord[0][0]=0; xord[0][1]=0; xord[0][2]=-0.5;
xord[1][0]=0; xord[1][1]=1; xord[1][2]=0.5;
xord[2][0]=1; xord[2][1]=0; xord[2][2]=0.5;
xord[3][0]=1; xord[3][1]=1; xord[3][2]=-0.5;

for(int i=0;i<max_it;i++){

select = rand()%4;

data[i][0] = xord[select][0];
data[i][1] = xord[select][1];
data[i][2] = xord[select][2];

		}


}
int init,endtop,emap=0,wgt1,wgt2;
double *weights_save;
int xor_net(double *weights,double * error);

int xor_net_init(){
double *weights,error;
int wgtsize;
init=1;
wgtsize=xor_net(weights,&error);
init=0;
return wgtsize;
}



int xor_net(double *weights,double * error){
/*init net*/
int in=2,out=1,hid;
int layers=3;
int context=0;
int nodes[layers];
double *input,*output,sumerror;
//ofstream outfile("outfile.dat");


if(init==1){
hid=8;
}else{
endtop=hid=weights[0];
}

output = input = new double;


int max_it=10;
double **data;

data = new double*[max_it];

input = new double[in];
output = new double[out];

for(int i=0;i<max_it;i++){
data[i] = new double[4];
}


for(int n=1;n<=layers-1;n++){
nodes[0]=in;
nodes[n]=hid;
if(context!=0){
nodes[context]=2;
		}
}
nodes[layers]=out;

net jack;
jack.init(layers,nodes,context);  // new net created
//jack.print_nodes();
if(init==1){return jack.wgtsize();}else{
init=0;
/*prep data*/
prepdata(data,max_it);

/*train*/
int j=0,iter=0;

			//jack.wgtupdt(weights);

for(int k=0;k<100;k++){

			//jack.wgtupdt(weights);

for(int j=0;j<max_it/2;j++){

input[0] = data[j][0];
input[1] = data[j][1];
output[0]= data[j][2];

iter=k*(max_it/2)+j;

			//jack.wgtupdt(weights);


jack.run_net(input,1,output,iter,max_it/2); //backprop
		
			
			//jack.emap_wgtupdt(weights,wgt1,wgt2);
			//jack.wgtupdt(weights);
			
			} 
			}
		

jack.wgtupdt(weights);


if(emap==0){
for(int j=0;j<max_it/2;j++){

input[0] = data[j][0];
input[1] = data[j][1];
output[0] =data[j][2];

jack.run_net(input,2,output,j,max_it/2); //feedforward only

sumerror+=sqrt(pow(jack.output[0]-output[0],2));

//if(j>(max_it/2)-2){
//cout<<"inputs:"<<input[0]<<","<<input[1]<<" output:"<<jack.output[0]<<" actual:"<<output[0]<<"\n";
//			}

			}
sumerror=sumerror/(max_it/2);

}else{
for(int i=0;i<100;i++){
j =rand()%max_it/2;
input[0] = data[j][0];
input[1] = data[j][1];
output[0] =data[j][2];

jack.run_net(input,2,output,j,max_it/2); //feedforward only
sumerror+=sqrt(pow(jack.output[0]-output[0],2));
			//sumerror=jack.output[0]-output[0];
			}
}

//cout<<"Err="<<sumerror<<"\n";

*error=sumerror/100;//
//*error=jack.output[0]-output[0];
										//outfile.close();	   

 }
	}
