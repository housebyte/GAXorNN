#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include "../Resource/BPN.C"
#include "slim_net_wrap_rnd.cc"

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

void prepsundata(double **data,int max_it,int numinputs){

NormalizeSunspots();
int t=-1;

for(int numdata=TRAIN_LWB;numdata<=TRAIN_LWB+max_it-1;numdata++){
t++;
for(int i=0;i<numinputs;i++){
data[t][i]=(double) Sunspots [numdata+i];
}
							}

}

int init,endtop,emap=0,wgt1,wgt2;
double *weights_save;
int xor_net(double *weights,double * error,int lastnet);

int xor_net_init(){
double *weights,error;
int wgtsize;
init=1;
wgtsize=xor_net(weights,&error,0);
init=0;
return wgtsize;
}

void save_weights(double *weights,const char *fname,int wgtsize){
FILE *fout;
fout=fopen(fname, "w");
if(fout == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
for(int i=0;i<wgtsize;i++){
fprintf(fout,"%f\n",weights[i]);
}

//fclose(fout);
}

void load_weights(double ** weights,const char *fname,int wgtsize){
char c,*l;
int lines=0;
int count=0,i;
double x;
double *weights_;
weights_ = new double[wgtsize];
FILE *fh;
fh=fopen(fname, "r");
if(fh == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}

char line[100];
while( fgets( line,100,fh ) ){
if( 1==sscanf(line,"%lf",&x) ){
weights_[count]=x;
count++;
}

}
				

*weights = weights_;
fclose(fh);
}

//double **data,*input,*output;

int xor_net(double *weights,double * error,int lastnet){
/*init net*/
int in=30,out=1,hid=10;
int layers=3;
int context=0;
int nodes[layers];
double sumerror;
ofstream outfile("Data/outfile.dat");
ofstream errfile("Data/errfile.dat");
ofstream datfile("Data/datfile.dat");
ofstream nmsefile("Data/nmse.dat");
		
int max_it=200;
//lastnet=1;
if(init==1){
hid=10;
}else{
hid=120;//endtop=hid=weights[0];
}



for(int n=1;n<=layers-1;n++){
nodes[0]=in;
nodes[n]=hid;
//nodes[2]=10;
if(context!=0){
nodes[context]=2;
		}
}
nodes[layers-1]=out;

net jack(layers,nodes,context);  // new net created
jack.print_nodes();


jack.output_ = jack.input_ = new double;
jack.data = new double*[max_it];
jack.input_ = new double[in];
jack.output = new double[out];
for(int i=0;i<max_it;i++){
jack.data[i] = new double[in+out];
}
jack.data_ = new double[in+out];

if(init==1){
//prepsundata(jack.data,max_it,in+out);
return jack.wgtsize();
}else{


init=0;
/*prep data*/
prepsundata(jack.data,max_it,in+out);				//May have processing issues
jack.data__=jack.data;


/*train*/
int j=0,iter=0;
int iterate_=200;
int preload_=0;
int it_lim;
int zero_data=0;
int spars_=25;						//Number of inputs to be subst with outputs
							//jack.wgtupdt(weights);


if(preload_==1){					//If not using Backprop
double **weights__;
 int wgtsize__=jack.wgtsize();
 weights__= new double*[wgtsize__];
load_weights(weights__,"Data/WeightArray_tanh.dat",wgtsize__);   //Include wgtsize check
jack.setwgts(*weights__);
		}


for(int k=0;k<iterate_;k++){
sumerror=0;
			//jack.wgtupdt(weights);


jack.data_=jack.data[0];

int m=-1;
for(int j=0;j<max_it/2;j++){
m++;
if(m==30){m=0;}
if(j>0&&zero_data==1){
					//Start Learning predictions based on previous outputs
						//After this point we collect 30 inputs from outputs						//So max_it/4+30 equals no data
for(int d=0;d<in-1;d++){
jack.data_[d]=jack.data_[d+1];				//Push Values back
}

if(m>spars_){
jack.data_[in-1] = jack.output[0];			//Add last output
	      }else{
jack.data_[in-1] = jack.data[j][30];	      
	      }
	      
	}else{
jack.data_=jack.data[j];	
	}      
		



for(int t=0;t<in;t++){

jack.data__[j][t] = jack.input_[t] = jack.data_[t];
//jack.input_[t] = jack.data[j][t];

		     }

jack.output_[0] = jack.data[j][30];

iter=k*(max_it/2)+j;

			//jack.wgtupdt(weights);
if(preload_==1&&iter==0){iter=1;}			//Stop init if preloaded

jack.run_net(jack.input_,1,jack.output_,iter,max_it/2); //backprop
sumerror+=sqrt(pow(jack.output[0]-jack.output_[0],2));
		
if(lastnet==1&&k==iterate_-1&&iterate_!=0){
//errfile<<sqrt(pow(jack.output[0]-jack.output_[0],2))<<"\n";
}			
			//jack.emap_wgtupdt(weights,wgt1,wgt2);
			//jack.wgtupdt(weights);
			
			} 
			
sumerror=sumerror/(max_it/2);
nmsefile<<sumerror<<"\n";			
			
			}
		

//jack.wgtupdt(weights);

//Load Weights for Non Learning iterate_ = 0
if(iterate_==0){					//If not using Backprop
double **weights__;
 int wgtsize__=jack.wgtsize();
 weights__= new double*[wgtsize__];
load_weights(weights__,"Data/WeightArray_tanh.dat",wgtsize__);   //Include wgtsize check
jack.setwgts(*weights__);
		}
		
sumerror=0;
if(emap==0){
for(int j=max_it/2;j<max_it;j++){

for(int t=0;t<in;t++){
jack.input_[t] = jack.data[j][t];
		     }

jack.output_[0] = jack.data[j][30];

jack.run_net(jack.input_,2,jack.output_,j,max_it/2); //feedforward only
sumerror+=sqrt(pow(jack.output[0]-jack.output_[0],2));

if(j>(max_it/2)-2){
//cout<<"inputs:"<<jack.input_[0]<<","<<jack.input_[1]<<" output:"<<jack.output[0]<<" actual:"<<jack.output_[0]<<"\n";
			}
if(lastnet==1){
errfile<<sqrt(pow(jack.output[0]-jack.output_[0],2))<<"\n";
outfile<<jack.output[0]<<"\n";
datfile<<jack.output_[0]<<"\n";
	     }

			}
sumerror=sumerror/(max_it/2);

}else{
for(int i=0;i<100;i++){
j =rand()%max_it/2;
for(int t=0;t<30;t++){
jack.input_[t] = jack.data[j][t];
		     }

jack.output_[0] = jack.data[j][30];

jack.run_net(jack.input_,2,jack.output_,j,max_it/2); //feedforward only
sumerror+=sqrt(pow(jack.output[0]-jack.output_[0],2));
			//sumerror=jack.output[0]-output[0];
			}
}

//cout<<"Err="<<sumerror<<"\n";

*error=sumerror;//
//*error=jack.output[0]-output[0];					
									nmsefile.close();
									datfile.close();
									errfile.close();
									outfile.close();
										      	   

if(iterate_>0){				//If using Backprop
 double **weights_;
 int wgtsize_=jack.wgtsize();
 weights_ = new double*[wgtsize_];
 jack.getwgts(weights_);
 save_weights(*weights_,"Data/WeightArray_tanh.dat",wgtsize_);    //Include prefix to Weights File - Title
 		}
 }
	}
	
	
int main(){

double *weights;
double error;
double **input;
int islast=1,wgtsize_;

wgtsize_=xor_net_init();
xor_net(weights,&error,islast);
cout<<error<<"Error\n";
cout<<wgtsize_<<"Wgtsize\n";
}	

