//to create a neuron container for use with a network container

/*neuron centered archetecture*/
/*test with xor and later adapt to threading neurons in each layer for faster speed and ability to add/remove neurons and links*/
/*problem - doesnt work on xor needs rewrite?*/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <time.h>
/*#include "../neuron/mackayglass/mackey.cc"*/

using namespace std;

double function_x_noisy(double sum,int type, double y);
double function_x(double sum,int type, double y);
double function_y(double sum,int type, double y);
double function_z(double sum,int type, double y);
double function_norm(double sum, int type, double y);
double function_input(double sum, int type, double y);
double function_bisigmoid(double sum,int type, double y);
double function_bent(double sum,int type,double y);
double function_gauss(double sum,int type,double y);

/*Globals*/  
double temp,oldtemp;
/*Stats*/
double min__,max__,mean__,std__;

#define PI 3.142

class neuron

{

private:

int   nnodes;
double **neurodata;

public:
double act;
double error;
double *delta;
double *prevdelta;
double *prevweight;
double Gain__;
double Alpha,PreAlpha; 			//Used for Parametric Activation function
//context weights
double cinweight;
double coutweight;

double (*m_pointertofunction)(double,int,double);



void init(int nodes, double (*pointertofunction) (double,int,double))
{

delta = new double[nodes];
prevdelta = new double[nodes];
prevweight = new double[nodes];

neurodata = new double*[nodes];
for(int i = 0; i < nodes; ++i)
neurodata[i] = new double[5];
nnodes = nodes;

m_pointertofunction = pointertofunction;

}


void eraser()
{

delete[] neurodata;
delete[] delta;
delete[] prevdelta;
delete[] prevweight;

}

void erase()
{
delete[] neurodata;

neurodata =0;
nnodes =0;

}

double& operator()(int nIndex,int y)
{
  //assert(nIndex >= 0 && nIndex < nnodes);
  return neurodata[nIndex][y];
  
  
}

int getlength(){ return nnodes; }



double activation(){

double sum;
sum=0;
for(int i=0;i<nnodes;i++){

sum += neurodata[i][0]*neurodata[i][1];

			}
	
sum = sum*Gain__;			//Include Gain for each Neuron

return m_pointertofunction(sum,1,Alpha);

}




};



class net

{

public:

int nlayers;
int *nnodes;
neuron **node;
neuron *thisnode;
neuron *context;


int numnodes;
int input_nodes;
int output_nodes;
int context_layer;
int count;
double Gain_;
double *output;
double lrate,lrate_;
double momentum;
double decay;
double *sumerr;
double errdelta;
double neterror;
double prevdelt,ddelt,sumneterr,sumerrdelt;
double min_,max_,std_,mean_;
double **data,*input_,*output_;
double **data__,*data_;
int dlrate;
//double *input;

void init(int layers, int *netnode, int contextlayer){
dlrate=1;			//switch for dynamic lrate	
lrate_=0.5;
lrate=0.5;
momentum =0.01;
 decay=0.0001;
 Gain_=1;
 count=0;
 srand(468768);			//time(NULL) for random seed
 
 //input = new double[netnode[0]];

 output_nodes = netnode[layers-1];
 
 input_nodes = netnode[0];
 
 sumerr = new double[output_nodes];
 
 output = new double[netnode[layers-1]];  //output array
 
 node = new neuron*[layers]; 
 
 
 if(contextlayer<layers&&contextlayer!=0){ //make sure the context is valid
 printf("context layer=%d\n",contextlayer);
 context = new neuron[netnode[contextlayer]];
 context_layer=contextlayer;
			  }else{context_layer=0;}


			  
 for(int y=0;y<layers;y++){
 
 node[y] = new neuron[netnode[y]];
 }


  //memory for net

nnodes = new int[layers];

nnodes = netnode;

int layer=0,p=0;
numnodes=0;
for(int layer=0; layer<layers;layer++){
for(int x=0; x<nnodes[layer]; x++){
numnodes++;
				}}
				
 				
//neuron *thisnode;

thisnode = new neuron[numnodes];



//cout << input_nodes << "<--Number of inputs";

for(int layer=0; layer<layers;layer++){

for(int x=0; x<nnodes[layer]; x++){		//connect neurons here adapt later
p++;
thisnode[p]=node[layer][x];			//index to each neuron
node[layer][x].Gain__=Gain_;
if(layer==0){
node[layer][x].init(1,function_input);  //use linear function to transmit inputs

}
else if(layer==layers-1){

node[layer][x].init(nnodes[layer-1],function_input); //use output function to transmit outputs
						//output computed from final layer
}
else{
node[layer][x].init(nnodes[layer-1],function_x);
	}		
	
				   }
					}
				

					
if(context_layer!=0){					
for(int l=0;l<nnodes[context_layer];l++){
									
context[l].init(1,function_y);		//initialise each context neuron with only 1 input

}		}		
				

nlayers = layers;
}



neuron& operator()(int layer,int x){
	
	return node[layer][x]; 
}



void eraser(){

delete[] sumerr;
delete[] output;
delete[] node;
delete[] nnodes;
delete[] thisnode;
delete[] context;

}


int wgtsize(){
int c=0;
for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){
c++;
				}
				}
				}
return c;
}

int sethid(double *weights){
int totalc=0;

for(int i=0;i<nlayers;i++)
for(int j=0;j<nnodes[i];j++)
for(int k=0;k<node[i][j].getlength();k++)
totalc++;

return (int)weights[totalc+1]; //retrieve topology 
}

void emap_wgtupdt(double *weights,int wgt1,int wgt2){
int c=1;
int totalc=0;

for(int i=0;i<nlayers;i++)
for(int j=0;j<nnodes[i];j++)
for(int k=0;k<node[i][j].getlength();k++)
totalc++;

int _rand,div,point=1;
div=totalc/point;
_rand=rand()%point+1;
point=(div)*_rand;

for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){

//cout<<i<<","<<j<<"---"<<node[i][j].getlength()<<"number of connections for neuron\n";

//if(c>point||rand()%100>39){
if(rand()%100>75){
node[i][j](k,0) = weights[c];	
		    }

//				}
c++;
				}
				}
				}


//cout<<nnodes[1]<<","<<c<<"----num weights\n";
}

void wgtupdt(double *weights){
int c=1;
int totalc=0;

for(int i=0;i<nlayers;i++)
for(int j=0;j<nnodes[i];j++)
for(int k=0;k<node[i][j].getlength();k++)
totalc++;

int _rand,div,point=1;
div=totalc/point;
_rand=rand()%point+1;
point=(div)*_rand;

for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){

//cout<<i<<","<<j<<"---"<<node[i][j].getlength()<<"number of connections for neuron\n";

//if(c>point||rand()%100>39){

node[i][j](k,0) += weights[c];	

//				}
c++;
				}
				}
				}


//cout<<nnodes[1]<<","<<c<<"----num weights\n";
}

void getwgts(double ** weights){
int c=-1;
double *weights_;
weights_ = new double[wgtsize()];
for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){
c++;
weights_[c] = node[i][j](k,0);

}}}
*weights = weights_;

}

void setwgts(double *weights){
int c=-1;
for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){
c++;
node[i][j](k,0) = weights[c];
}}}

//for(int i=0;i<c;i++){
//cout<<"Loading weights:"<<weights[i]<<"\n";
//}


}



void update(double *input, int type){  //feedforward

double res;


//set all weights to random
if(type==1){


for(int i=0;i<nlayers;i++){

for(int j=0;j<nnodes[i];j++){

for(int k=0;k<node[i][j].getlength();k++){

res = node[i][j](k,0) = (double)  1/(rand()%10+1); //(0.7 *(rand()%1000 /500.0)) - 0.7;

//sub for random initialise weights

node[i][j].delta[k] = 0;//(double)   1/(rand()%10+1); //initialise deltas

node[i][j].prevdelta[k] = 0;//(double) 1/(rand()%10+1);

node[i][j].prevweight[k] = 0;//(double) 1/(rand()%10+1);

node[i][j].Alpha = (double) 1/(rand()%100+1);    //use Gaussian Noise Initialisation NOT Uniform Random?

node[i][j].act = (double) 1/(rand()%10+1);
node[i][j](k,1) = (double) 1/(rand()%10+1);


//cout << res << "\n"; 
					}
			}
			}
			
			
			
//initialise context weights and activations
if(context_layer!=0){			
for(int l=0;l<nnodes[context_layer];l++){
context[l].coutweight=-0.5 + double(rand()/(RAND_MAX + 1.0));//(double)  1/(rand()%10+1);
context[l].cinweight=0.5;				
context[l].act=(double)  1/(rand()%10+1);
					}
		    }	
	
		}
		
//Gather normalisation data from Inputs
max_=input[0],min_=input[0];
double sum;
for(int i=0;i<input_nodes;i++){
sum +=input[i];
if(input[i]>max_){
max_=input[i];
}
if(input[i]<min_){
min_=input[i];
}

}

//Average and Std deviation
mean_=sum/input_nodes;
sum=0;
for(int i=0;i<input_nodes;i++){
sum+=pow(input[i]-mean_,2);
}
std_ =sqrt(sum/input_nodes);
//Set to Globals//
mean__=mean_;
std__=std_;
min__=min_;
max__=max_;

	
//set input layer to input values		

for(int j=0;j<input_nodes;j++){

//cout<< node[0][j].getlength()<<"inputs\n";

for(int k=0;k<node[0][j].getlength();k++){

node[0][j](k,1) = input[j];
node[0][j](k,0) = 1;             //set weight to 1 for 1-1 mapping of inputs

//cout <<"node" <<j<< "act:" << node[0][j](k,1) << " ";
//cout << "weight:"<< node[0][j](k,0) << "\n";

					}			
				}

				
int current_layer;								

//calc activations from input layer by layer

for(int i=1;i<nlayers;i++){

current_layer = i-1;

//cout<<"layer:"<<i<<"\n";

for(int j=0;j<nnodes[i];j++){

//cout<< "\nnode " << i<<","<<j<<" length:" <<node[i][j].getlength()<<"\n";

for(int k=0;k<node[i][j].getlength();k++){

//no. neurons in previous layer i-1 = k 

//cout << "node act:" << i-1 <<","<<k<<": "<< node[i-1][k].activation()<<"\n";

node[i-1][k].act = node[i][j](k,1) = node[i-1][k].activation();
					

if(current_layer==context_layer&&context_layer!=0){

//node[i-1][k].act = node[i][j](k,1) += context[k].act*context[k].coutweight;

//*not correct needs to be included in activation() function as additional act*weight*/

//cout<< "context_layer:"<<i-1<<" Act:"<<k<<"::"<<node[i-1][k].act<<"\n";
//cout<< "context:"<<i-1<<" Act:"<<k<<"::"<<context[k].act<<"\n";


				        }					
					
					}
if(i==nlayers-1){

//cout << "weights:" << node[i][j](0,0)<<","<<node[i][j](1,0)<<","<<"\n";

//cout << "node act:" << i <<","<<j<<": "<< node[i][j].activation()<<"\n";

node[i][j].act = output[j] = node[i][j].activation();


//cout<<node[i][j].activation()<< "++" << output[j]   <<"<--activation1\n";
		
		}
//node[i][j].activation()

				}
				}
				
				
//cout<<node[nlayers][0].activation()<<"--"<<node[nlayers][0].act<<"--"<<output[0]<<"\n";
						
//cout<<node[1][1].act<<"--\n";	

//copy activations to context neuron
//update context neurons with hidden layer activations
if(context_layer!=0){
for(int j=0;j<nnodes[context_layer];j++){

context[j].act = context[j].cinweight*node[context_layer][j].act;

//cout<< "node_layer:"<<context_layer<<" Act:"<<j<<"::"<<node[context_layer][j].act<<"\n";

//cout<< "context_memory:"<<context_layer<<" Act:"<<j<<"::"<<context[j].act<<"\n";



}
			}		
			
			}



void learn(){				//later use pointer to learning function

//update();

//backprop();


}


void backprop(double *d,int type){

//compute output error
double sum;
double err,serr=0;			//output error
double wt;
//double sumerr;

double a;			//activation value for neuron

double (*func)(double,int,double);		       //activation function for neuron

neuron *y = node[nlayers-1];

errdelta=0;

for(int i=0;i<output_nodes;i++){

errdelta += sumerr[i]; //use previous sumerr to calc delta error

func = y[i].m_pointertofunction;

a = output[i];  

//cout <<"output"<<output[i] <<":" <<d[i]<<"\n";   

if(type==1){
err = d[i]-a;

//cout<<i<<"rawerror"<<err<<"\n";

	   }else{
err = d[i];	   
	   }

serr += sumerr[i] =pow(err,2);

						//err = error_func(err);
//err = 0.5*pow(err,2);

//err = 1/(1+(exp(-err)));

y[i].error = err;//func(err,-1,a)*Gain_;

//cout<<i<<"error"<<y[i].error<<"\n";

				}
				
//cout <<"error"<<err;
				

errdelta = serr/2 - neterror;

neterror = serr/2;        //sum of squared error


//feedback the errors


	for(int i=nlayers-2;i>=0;i--){

    for(int j=0;j<nnodes[i];j++){
    
    sum=0;
    for(int k=0;k<nnodes[i+1];k++){
    
    err = node[i+1][k].error;
    //cout<<i+1<<":"<<k<<" "<<err<<"err\n";
    
    wt = node[i+1][k](j,0);		//weight connecting neuron j to k
    sum += err*wt;
    

   				   }
	//sum = 0.5*pow(sum,2);
				   
	func = node[i][j].m_pointertofunction;
	 
	 a = node[i][j].act;
					
	 node[i][j].error = func(sum,-1,a)*Gain_;
	
	//cout<<node[i][j].error<<"--"<<i<<":"<<j<<" error\n";
	//if(i==1){cout<<node[i][j].error<<"---"<<i<<"---"<<j<<"--MLPERROR\n";}
	
					}
					   }
					   
		   
//errors been feedback now change weights using lrate and momentum					   
	
double dlta;
double prvdlta;
	
					   
	for(int i=1;i<nlayers;i++){
	
	for(int j=0;j<nnodes[i];j++){
	
 for(int k=0;k<node[i][j].getlength();k++){
 
 	node[i][j].PreAlpha = lrate*node[i][j].error*node[i][j].Alpha + momentum*node[i][j].PreAlpha;
	
 	node[i][j].delta[k] = lrate*node[i][j].error*node[i][j](k,1) + momentum*node[i][j].prevdelta[k] - decay*node[i][j].prevweight[k];		   
					   
	node[i][j].prevweight[k] = node[i][j](k,0) += node[i][j].delta[k];
	
	node[i][j].prevdelta[k] = node[i][j].delta[k];
	
	node[i][j].Alpha += node[i][j].PreAlpha; 
	
	
	//cout<<"Weight update:"<<i<<"---"<<j<<"---"<<k<<"--"<<node[i][j].delta[k]<<"\n";
	//cout<<"Alpha:"<<node[i][j].Alpha<<"\n";
						}
						
	
					}
					} 
					
	//cout<<node[2][0](0,1)<<","<<node[1][0].act<<"\n";				
					
	//update weights coming from context layer (coutweight)				
	/*				
	for(int j=0;j<nnodes[context_layer];j++){
	

 
 	context[j].delta[0] = lrate*node[context_layer][j].error*node[context_layer][j].act + momentum*context[j].prevdelta[0] -
	decay*context[j].prevweight[0];		   
					   
	context[j].prevweight[0] = context[j].coutweight += context[j].delta[0];
	
	context[j].prevdelta[0] = context[j].delta[0];
	
		
	
					}				
	*/				
					
					
					
					
	//cout<<"error5\n";				
					   
					   
//cout << "weight(1,2):" << node[1][2](1,0) <<"\n";
//cout << "error (1,2):" << node[1][2].error <<"\n";
 }
 
 
double error_func(double dif){

 if ((-0.1 < dif) && (dif < 0.1))
	return (0.0);
      else 
	return (dif);
	/*
if(err < -.9999999)
return (-17.0);
else if (err > .99999)
return (17.0);
else return ( log ((1.0+err)/(1.0-err)) );*/

} 

//use for online training rl



void run_net(double *input, int train, double *feedback, int iter,int itermax){
//max_no iter

double shift = 55/100;
int c,min=0,max=0;

int i= iter,tempshift=650;

if(train==1&&dlrate==1){

if(i>shift*itermax){
lrate -= 0.000000001;
}else{

lrate += 0.0000001;
}


if(max==1){lrate += 0.000001;}	
if(min==1){lrate -= 0.000001;}   


if( ((sumerrdelt-prevdelt)>0&&ddelt<0)||((sumerrdelt-prevdelt)<0&&ddelt>0)  ){

if(ddelt>0){
max=1;min=0;}else{
max=0;min=1;}

}




ddelt=sumerrdelt-prevdelt;

if(i>count){				//reset neterr
prevdelt=sumerrdelt;
sumneterr=0;sumerrdelt=0;
count=i;
	    }
	    

sumneterr+=neterror;			//sum neterr over iterations
sumerrdelt+=errdelta;

					//boltzman temperature
oldtemp=temp;					
if(iter%tempshift==1){
//temp += 0.0025*neterror;			//(double(rand()/(RAND_MAX + 1.0))/iter);
temp += double(rand()/(RAND_MAX + 1.0));
temp -= double(rand()/(RAND_MAX + 1.0));

if(neterror<0.0025&&temp>0){
temp -= iter*0.0000025;			//(1-  double(rand()/(RAND_MAX + 1.0))/iter);


			}
		}
		if(temp<0)temp=oldtemp;					
		//temp=1;			
					
					
					
					
					}



//lrate=lrate_;
//lrate=0.005;
//momentum =0.0001;
 //decay=0.0001;



if(i==0&&train<2){
//cout<<i<<"  initialising";
update(input,1);			//initialise values


}else{
			  		// new net updated with env data

update(input,0);

//cout<<output[0]<<"<--\n";


if(train<2){
backprop(feedback,train);		      //backprop feedback is error  1 = desired  0 = reinforce 2 =no train

	    }
	    
if(train==3){				      //train net with ga providing weights vector
//update weights vector



}	    

}

//cout<<output[0]<<"<--\n";
//cout << lrate << "\n";


	

}

		






void print_nodes(){

for(int x=0;x<nlayers;x++){
cout << "\nlayer:" << x << "\n";



for(int y=0;y<nnodes[x];y++){
cout << "node:" << y << " "; 


for(int z=0;z<node[x][y].getlength();z++){

//cout << "weight:"<< node[x][y](z,0) << "";
						}

				}

			}

cout << "\n\n";
		}

void save_weights(const char *fname){
FILE *fout;
fout=fopen(fname, "w");

int wgtsize_;
wgtsize_ = wgtsize();
double *weights;
weights = new double[wgtsize_];
getwgts(&weights);

//for(int i=0;i<wgtsize_;i++){
//cout<<"Saving weights:"<<weights[i]<<"\n";
//			  }

if(fout == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
for(int i=0;i<wgtsize_;i++){
fprintf(fout,"%f\n",weights[i]);
}

//fclose(fout);
}

void load_weights(const char *fname){
char c,*l;
int lines=0;
int count=0,i;
float x;
double *weights_;
int wgtsize_;
wgtsize_ = wgtsize();
weights_ = new double[wgtsize_];
FILE *fh;
fh=fopen(fname, "r");
if(fh == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
while  ((c=fgetc(fh)) != EOF&&count<wgtsize_)   {

fscanf(fh,"%f",&x);
weights_[count]=(double)x;
count++;
	   
				}


setwgts(weights_);

fclose(fh);
}



};


double function_norm(double sum, int type, double y){
double sum_;

//sum_ = (1/(1+exp(-sum)));
//sum_ =  pow(((sum-125)/255),2)/1000;		//Normalised to scale 0,1 for images
//sum_ = sqrt(pow((sum_-mean__)/std__,2))/100;
  sum_ = (sum-min__)/(max__-min__);


if(type==-1){return 0;}else{

return sum_;}
}



double function_input(double sum, int type, double y){

//sum = sqrt(pow(sin(sum),2));

if(type==1){
return sum;
	}
	if(type==-1){
	
	return 1-sum;
	}

}


double function_x(double sum,int type,double y){  //activation function for all hidden layer neurons

double sigmoid,temp;

if(type==1){

if(sum>0.5){					//this is actually hardlim not sigmoid
 sigmoid =  0.5;}else
 if(sum<-0.5){
 sigmoid = 0;}
 else
 sigmoid = 1/(1+exp((double) -sum));


		}
if(type==-1){


temp = 1/(1+exp((double) -y));

//sigmoid = (0.25-(temp*temp) )* sum;
sigmoid =  y*(1-y) * sum;		//derivative sigmoid
//sigmoid = (1-pow(tanh(y),2))*sum;


		}
return sigmoid; 


}

double function_x_noisy(double sum,int type,double y){  //activation function for all hidden layer neurons
double rand_=0;
double sigmoid,temp;

if(type==1){
rand_ = ((double)(rand()%2000))/10000;		//Sub with Gaussian Noise
if(sum>0.5){					//this is actually hardlim not sigmoid
 sigmoid =  0.5;}else
 if(sum<-0.5){
 sigmoid = 0;}
 		else{
 if(sum<0){					//Add the Parametric Alpha
 rand_*=y;
 }
 sigmoid = 1/(1+exp((double) -sum));
 	  }
 
 sigmoid+=rand_;

		}
	
if(type==-1){


temp = 1/(1+exp((double) -y));

//sigmoid = (0.25-(temp*temp) )* sum;
sigmoid =  y*(1-y) * sum;		//derivative sigmoid
//sigmoid = (1-pow(tanh(y),2))*sum;


		}
return sigmoid; 


}

double function_y(double sum,int type, double y){  //input function
double rand_=0;
//sum=sum/1000;

if(type==1){
//rand_ = ((double)(rand()%1000))/10000;

sum = tanh(sum)+rand_;
//sum = (2/(1+exp(-2*sum)))-1;		//Kick Ass


}
if(type==-1){

sum = (1-pow(tanh(y),2))*sum;

}

return sum;

}

double function_z(double sum, int type, double y){	//output function

double out;
double sum_;
double prob;
int hardlim=1;

//sum_ = sqrt(pow(sin(sum),2));			//sum_ = 1/(1+exp(-sum));
prob = (double)(rand()%100)/100;

if(hardlim==1){
sum_=sum;
prob=0;
}

if(sum>prob){out=1;}
else {out=0;}


if(type==-1){

						//out = y*(1-y) * sum;
return (1-y);
}else{


return out;
}
}

double function_bisigmoid(double sum, int type,double y){

double alpha=0.5;

if(type==1){

sum = (1/1+exp(-alpha * sum))-2;
sum = sqrt(pow(sum,2));
return sum;
}else{


//sum = (2* alpha * exp(-alpha * sum ))/pow((1+ exp(-alpha * sum)),2);
sum = ((2* alpha * exp(-alpha * y ))/pow((1+ exp(-alpha * y)),2))*sum;

return sum;
}


}

double function_bent(double sum,int type,double y){
double temp;

if(type==1){
temp = sqrt(pow(sum,2)+1)-1/2;
sum += temp;


return sum;
}else{
temp = y/2*sqrt(pow(y,2)+1);
sum  = (temp+1)*sum;

return sum;
}

}

double function_gauss(double sum,int type,double y){
double temp;

if(type==1){
sum = exp(-pow(sum,2));

return sum;
}else{
sum = -2*y*exp(-pow(y,2))*sum;

return sum;
}


}
