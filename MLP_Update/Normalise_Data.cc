#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;
/* Routines to save and load from Yahoo ftse data - compute moving average, exponential moving average, MACD, rate of change */
/* Scaling and Unscaling - function to normalise data and then convert back to original range */
/*Moving average and scaled inputs to 

/* Load data from csv or directly from yahoo */

/* Begin with Loading data and normalising to within range 0,1 */

void load_csv(double ** values,int * get_size_,const char *fname,int pos){
char c,*l;
int lines=0;
int count=0,i;
int get_size;
char x[10],y[10],z[10];
double *x_,*y_,*z_;
double *values_;
char buff[1000];

FILE *fh;
fh=fopen(fname, "r");
if(fh == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}

get_size=0;

get_size=600;

x_ = new double[get_size];
y_ = new double[get_size];
z_ = new double[get_size];

count=-1;

while(fscanf(fh,"%s\n",buff)==1){
count++;
if(count>1){
sscanf(buff,"%*[^','],%[^','],%*[^','],%*[^','],%[^','],%*[^','],%s",x,y,z);

x_[count]=atof(x);
y_[count]=atof(y);
z_[count]=atof(z);
}

}
get_size=count;

count=-1;


for(int i=0;i<get_size;i++){
//cout<<i<<" "<<x_[i]<<" "<<y_[i]<<" "<<z_[i]<<"\n";
}


*get_size_ = get_size;
*values = z_;
fclose(fh);

}

void moving_average(double ** data_out, double *data, int get_size,int * dat_size,int k){
double sum=0;
int count=-1;
double *data_;
data_ = new double[get_size-k];

for(int i=k+1;i<get_size;i++){
count++;
sum=0;
for(int j=i-k;j<i;j++){
sum+=data[j];
			}
data_[count]=sum/k;
//cout<<count<<" "<<data_[count]<<" "<<data[i]<<"\n";
			}

*data_out = data_;
*dat_size = count;					           }

void feature_scale(double ** data_out, double *data, int get_size,int * dat_size){
double *data_;
data_ = new double[get_size];
double max=data[0],min=data[0];
for(int i=0;i<get_size;i++){
if(data[i]>max){
max=data[i];
}
if(data[i]<min){
min=data[i];
}			    }
for(int i=0;i<get_size;i++){
data_[i]=(data[i]-min)/(max-min);
}

*data_out = data_;
*dat_size = get_size; 								 }

void standardise(double ** data_out,double *data,int get_size,int * dat_size){
double *data_;
data_ = new double[get_size];
double mean_,std_;
double sum=0;
for(int i=0;i<get_size;i++){
sum+=data[i];
}
sum=0;
mean_=sum/get_size;
for(int i=0;i<get_size;i++){
sum+=pow(data[i]-mean_,2);
}
std_=sqrt(sum/get_size);
for(int i=0;i<get_size;i++){
data_[i] = sqrt(pow((data[i]-mean_)/std_,2));
}

*data_out = data_;
*dat_size = get_size;
										}


int main(){

int k_=10;
double *values;
double *values_;
double *values__;
double *values___;
int get_size,dat_size;
load_csv(&values,&get_size,"Ftse_Data/acl.l.csv",1);
int count=-1;


feature_scale(&values__,values,get_size,&dat_size);
standardise(&values_,values,get_size,&dat_size);
moving_average(&values___,values__,get_size,&dat_size,k_);

cout<<"sizeof Data:"<<sizeof(values___)<<"\n";

for(int i=0;i<dat_size-k_;i++){

//cout<<i<<" "<<values__[i]<<"\n";
//cout<<i<<" "<<values[i]<<" "<<100*values_[i]<<" "<<100*values__[i]<<"\n";
//cout<<i<<" "<<100*values_[i]<<" "<<100*values__[i]<<" "<<100*values___[i]<<"\n";
				}


		}


/*
if(i<k){
sum+=data[i];
}else{
count++;
data_[count]=sum+data[i]-data[i-k]
}

*/
