#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <time.h>
#include <pthread.h>
#include <vector>
#include <fstream>
#include <iostream>

#define offset 0
#define cross_def 0.6
#define mutat_def 0.06
#define max_def 1
#define _popsize 150
#define _chromolen 16
#define chromsize 75 //init size of individual to handle variable 
#define _infosize 5   //phenotype info			//topology
#define _objects 2    //number of objectives			
#define _endtop 4     //Up to 4 neurons selected with probability 0 - 1
#define _endfunc 2
#define _fixedtop 1  //Fixed topology e.g topology = 4 hidden neurons
#define topo_start 1
#define _hidval 3    //Value for Fixed Topology
#define _phenosize 5 //Amount of information stored in phenotype for each chromo e.g = _endtop
#define _wgtupdt 1   //Switch on/off weight update
using namespace std;

void * thread_function(void *p);

class ga

{

public:

typedef struct
{
double *weights;
double error;
}net;

typedef struct 
{
double* output;
double  fitness,fitscale;
double*  objfit;
char**   chromosome;
double   data;
double*  graphdata;  //Store Energy LAndscape data ie Error, Wgt1, Wgt2
double** phenotype; //Store Genome expression data e.g probability's etc
int*	 info;  //Additional information about phenotype
//double Temp;    //Used for Annealing
}entity;


typedef struct 
{
double  mean,nmean;
double stability;
double best,bestvalue,ascale,bscale;
double* bestscores;
int 	size;
int num_chromosomes;
int len_chromosomes;
entity** entity_array;
double Temp;    //Used for Annealing
int evaltop;    //Used for Annealing top entity.
}population;

net *mynet;
population *xpop,*ypop,*currentpop;
int thread,*index,whichthread,_wgtsize;
std::vector<int> thread_index;
pthread_mutex_t mutexdata;
ofstream outfile;
ofstream errplot;

int replace,bscore,positer,iterations,mutlog,crosslog,clonelog,xover_type,selector,mutator,swap_mut,flip_mut,len_binary;
double cross,mut,previousbest;


void (*m_pointertofunction)(double*,double *,double**);
void (*d_pointertofunction)(double*,double *,double**);

ga(double **input_array,int *config,void (*pointertofunction) (double*,double *,double**))
{


int i,j,k;
outfile.open("outfile.dat",ios::out);
errplot.open("errplot.dat",ios::out);

//mynet = new net;		//setup weights array later
//mynet->error=6.9996;
//mynet->weights = new double[5];

//replace with config array
_wgtsize=config[0];

 replace=0,bscore=0,positer=0,iterations=150,mutlog=0,crosslog=0,clonelog=0,xover_type=1,selector=0,mutator=1,swap_mut=1,flip_mut=0;
cross=cross_def,mut=mutat_def,previousbest=100;


xpop = new population;
ypop = new population;

init(xpop);init(ypop);

//printf("var = %d\n",len_binary);


m_pointertofunction = pointertofunction;
d_pointertofunction = pointertofunction;


}

void getwgtsize(int wgtsize){
_wgtsize = wgtsize; 
}

void run(){
int iter,newfit;
double lim=0.99,bestfit=100;

for(iter=0;iter<iterations;iter++){

xpop->Temp=100;
while(1){
if(xpop->Temp<lim){xpop->Temp=rand()%100;}
xpop->Temp-=0.99;
//eval_thread(xpop);
eval_top(xpop);   //evaluate the top individual with topology from varying temp

if(xpop->entity_array[0]->objfit[1]<bestfit){bestfit=xpop->entity_array[0]->objfit[1];newfit=1;break;}else{if(xpop->Temp<lim){newfit=0;break;}}
}


//Exclude bestfit individual's topology found in sim annealing from being altered 
if(newfit==1){
//Record Topology at Temp in Phenotype or erase Phenotype - preserve Genotype by assigning probabilitys of 1 to Chromosomes for topology
//sharing and cloning ie Chromo 1 = 11111111 -> Prob =1  this will ensure the fittest individuals topology is preserved in the gene pool
//If Bestfit then keep Phenotype if not then delete and do not store
cout<<"Bestfit:"<<bestfit<<"\n";
xpop->entity_array[0]->info[2]=1;
		}

//cout<<"Is outside Sim Anneal\n";

eval_thread(xpop);
//cout<<"errrrrrrooor\n";
//rank_scale(xpop);

//if(iter==0||iter==iterations-1)
stats(xpop);

if(selector==1)			{
if(iter>0)select(ypop,xpop);		//select from xpop into ypop for x-over
copy(xpop,ypop);			//copy ypop to xpop
eval_thread(xpop);	

				}

adapt(xpop,ypop);			//adapt from xpop to create new ypop
copy(xpop,ypop);			//copy population from ypop to xpop
			//apply fitness scale		
		
				}
		

errplot.close();
outfile.close();
}

void init(population *newpop){
srand(time(NULL));
int i,j,k;
//newpop = new population;
newpop->size=_popsize;
newpop->num_chromosomes = chromsize+4; //3;
len_binary = newpop->len_chromosomes = _chromolen;

newpop->bestscores = new double[iterations];
for(i=0;i<iterations;i++)newpop->bestscores[i]=100;

newpop->entity_array = (entity**) calloc(newpop->size,sizeof(entity*));
    

    
for(i=0; i<newpop->size; i++){
newpop->entity_array[i] = (entity*) calloc(newpop->size,sizeof(entity));

//Genotype
newpop->entity_array[i]->chromosome =(char**) calloc(newpop->num_chromosomes,sizeof(char*));
//Phenotype
newpop->entity_array[i]->phenotype =(double**) calloc(newpop->num_chromosomes,sizeof(double*));

for(j=0; j<newpop->num_chromosomes; j++){
newpop->entity_array[i]->chromosome[j] =(char*) calloc(newpop->len_chromosomes,sizeof(char));	
newpop->entity_array[i]->phenotype[j] =(double*) calloc(_phenosize,sizeof(double));												
												}

//Phenotype (Use for Memetic Algo)
newpop->entity_array[i]->info =(int*) calloc(_infosize,sizeof(int));
newpop->entity_array[i]->objfit =(double*) calloc(_objects,sizeof(double));
newpop->entity_array[i]->output =(double*) calloc(_objects,sizeof(double));
newpop->entity_array[i]->graphdata =(double*) calloc(3,sizeof(double));

//for(j=0; j<_infosize; j++){
//newpop->entity_array[i]->info[j] =(int*) calloc(_infosize,sizeof(int));
//					}
}



//initialise

for(i=0; i<newpop->size; i++){
for(j=0; j<newpop->num_chromosomes; j++){
for(k=0; k<newpop->len_chromosomes; k++){
if (rand()%10>5)
newpop->entity_array[i]->chromosome[j][k]='1';
else
newpop->entity_array[i]->chromosome[j][k]='0';

					}

				
				}
			}
	
			


}

  int bin2int_thread(char *binary){
  
  int result=0;
  for (int i=0; i<len_binary; ++i){
  //if(binary[i]==1)result++;
  result=result*1+(binary[i]=='1'?1:0);
				  }
//cout<<result<<"bin2int--------\n";
  return result;
  }

  float bin2dec(population *pop, char *binary,int numdec){
  
  float result;
  int decimal=0;
  
  for (int i=0; i<pop->len_chromosomes; ++i)
	{
	
		decimal=decimal*2+(binary[i]=='1'?1:0);
	}
  
  
  result = (float)(decimal/(6553.5*2-10));	//have to engineer some way to set range auto
  						//using nbit, and range
  return result;
  
  //divide by 65535 = value of 16bit move dec point to give range of 100 * 2 -100 to give 
  //range -100:100
 
  }  
  
   float bin2dec_thread(char *binary,int numdec){
  
  float result;
  int decimal=0;
  //int len_binary = 16;
  //printf("var = %d %d\n",len_binary,mutator);
  
  
  
  for (int i=0; i<len_binary; ++i)
	{
	
		decimal=decimal*2+(binary[i]=='1'?1:0);
	}
  
  
  result = (float)(decimal/(6553.5*2));	//have to engineer some way to set range auto
  
  
  
  						//using nbit, and range
  return result;
  
  //divide by 65535 = value of 16bit move dec point to give range of 100 * 2 -100 to give 
  //range -100:100
 
  }  
  
  

 double func(double A,double B){
  double t1,t2,t3,t4;
	
  //schaefer
       t1 = sin(sqrt(pow(A,2) + pow(B,2)));
       t2 = 1 + 0.001*(A*A + B*B);
       t3 = 0.5 + (t1*t1 - 0.5)/(t2*t2);
  
  //trig 
       //t4 = 1/(pow(B,2)+sqrt(A+pow(A,2)));         //gets nan
       //t4 = 1/(pow(B,2)+A+pow(A,2));  
       //t4 = 1/(5+2*A+pow(A,2)*pow(B,2));    
       //t4 = (pow(A,2)+pow(B,2));
       //t4 = 1-pow(A,2)*pow(B,2);
       //t4 = pow((1 - sin(A)),2);
         t4 = 5+sin(A);
       //t4 = (1-sin(pow(A,2)*pow(B,2)))-2;
       //t4 = pow(A,2);     
       t4 = 5 - (pow(A,2) + pow(B,2));
       //implement count zero's  
       //t4 = pow(A,2)+pow(B,2);
       //t4 = 1/(1-sin(A)*cos(B));
       //t4 = A*sin(A)*cos(B);
  return t4;
  }  

void eval_top_nothreads(population *pop){

entity *ent = pop->entity_array[0]; //get top individual
int topology=1;
double fit1,thisfit;


//get topology
for(int i=0;i<_endtop;i++){
topology += (1/(1+exp((bin2dec_thread(ent->chromosome[i+_endfunc],1)/this->xpop->Temp)))*100>rand()%100);
			  }
//cout<<topology<<"--------\n";

//evaluate
double **input_array;
mynet = new net;       //create a mynet array
int getsign,divider;
if(bin2int_thread(ent->chromosome[_endtop])>8){getsign = -1;}else{getsign = 1;}

divider=10;//bin2dec_thread(ent->chromosome[_endtop+1],1)*1000; 

int toposize=200; //turn into array for more layers
if(_fixedtop==1){
topology=_hidval;	}
//calculate toposize
int in=2,out=1,hid=topology,nlayers=1,type=1,conn=0; 
					//get values from config
					//type 1 = all connect


for(int i=0;i<nlayers;i++){
conn+=hid*in+i*hid*hid;
}
conn+=out*in+nlayers*out*hid;
toposize=conn;

mynet->weights = new double[toposize+_endtop];
mynet->weights[0] = topology;
_wgtsize = toposize;

//cout<<topology<<"-----"<<toposize<<"\n";

//Get Weights
for(int i=1;i<toposize;i++){
//mynet->weights[i] = bin2dec_thread(ent->chromosome[i+_endtop+_endfunc+2],1);
mynet->weights[i] = getsign*bin2dec_thread(ent->chromosome[i+_endtop+_endfunc+2],1)/divider;
//cout<<i<<","<<mynet->weights[i]<<"\n";
}


//cout<<"No Weights:"<<toposize<<"\n";

d_pointertofunction(mynet->weights,&mynet->error,input_array);

//Minimise Error
if(mynet->error>0)
	   fit1=1/(1/mynet->error);     
           else
	   fit1=1/mynet->error;

ent->output[1] = mynet->error;
//thisfit += mynet->error;
thisfit = sqrt(pow(fit1,2));
ent->fitness += thisfit;  
ent->objfit[1] = thisfit;

//cout<<"Is Simulated Annealing\n";
}

void* thread_function(void *p)
{
	
	pthread_mutex_lock (&mutexdata);

	int currentthread = this->whichthread;//this->thread_index.back();//this->index;
	this->whichthread+=1;

	//cout<<"Is Inside GA Thread:"<<this->whichthread<<"\n";
	
	//printf("current thread: %d",currentthread);
	
   	 //entity *ent = (entity *) p;
	
	entity *ent = this->currentpop->entity_array[currentthread];
	
	//entity *ent = this->xpop->entity_array[currentthread];

//****Objective 1*****//

	double A=0,B=0,C=0,D=0,E=0,F=0;
	int i,j,k,l;
	double fit1,fit2,t4,sumfit;
	
	//fflush(stdout);
	//printf("here");
	


         A = bin2dec_thread(ent->chromosome[0],1);
         B = bin2dec_thread(ent->chromosome[1],1);

    t4 = func(A,B);
       
       //printf("thread :%d --->%f  = func(%f,%f) %s\n",currentthread,t4,A,B,ent->chromosome[0]);
       
       if(max_def==0){
       //minimize
	   if(t4>0)
	   fit1=1/(1/t4);     
           else
	   fit1=1/t4;     
           	}else{
	//maximize	
	   if(t4>0){
	   fit1=1/t4;				//printf("greater than zero t4=%f\n",t4);
	   }else{
	   fit1=1/(1/(t4-5));			//printf("less than zero t4=%f\n",t4);
	   }
	         }
	
double thisfit=0;	 
        ent->output[0]   = t4;
        ent->objfit[0]  = sqrt(pow(fit1,2)); 
	//ent->fitness = sqrt(pow(fit1,2));
    	thisfit = sqrt(pow(fit1,2));
	ent->fitness = thisfit;

//**** Objective 2 *****//

double **input_array;
mynet = new net;		//setup weights array later
int topology=topo_start;
//mynet->weights = new double[_wgtsize+4];   
double prob,temp_,val_,sum_=0,min_=0.1,max_=1;
  int sel_;
  temp_ = this->currentpop->Temp; 

  for(int i=0;i<_endtop;i++){
  val_ = bin2dec_thread(ent->chromosome[i+_endfunc],5)/temp_;
  if(val_>max_){max_=val_;}
  if(val_<min_){min_=val_;}
  sum_+=bin2dec_thread(ent->chromosome[i+_endfunc],5);
  }


//Additional data for number of neurons in hidden layer
for(int i=0;i<_endtop;i++){
  //topology += (1/(1+exp((bin2dec_thread(ent->chromosome[i+_endfunc],1)/this->xpop->Temp)))*100>rand()%100);
   val_  = bin2dec_thread(ent->chromosome[i+_endfunc],5);
   prob  = ((val_)-min_)/(max_-min_);
   prob  = 1/(1+exp(prob));
   sel_  = prob*100>rand()%100;
   topology += sel_;
//double prob_=(1/(1+exp(bin2dec_thread(ent->chromosome[i+_endfunc],1)/this->xpop->Temp )))*100;
//double temp_=this->xpop->Temp;

//cout<<"Topology:"<<topology<<"Probability:"<<prob<<","<<temp_<<"\n";
   ent->phenotype[i][0] = val_;
   ent->phenotype[i][1] = prob;
   ent->phenotype[i][2] = sel_;
   ent->phenotype[i][3] = temp_;
   			  }
			  
   ent->phenotype[_endtop][0] = topology;			  
if(ent->info[2]==1&&this->currentpop->evaltop==0){			//preserved from sim annealing
cout<<"evaltop preserved topology rejected:"<<topology<<",Topology preserved:"<<ent->info[0]<<"\n";
topology=ent->info[0];
}
ent->info[0]=topology;

//if(topology == 0 ){topology = 4;}
//if(topology > 16 ){topology = 16;}

int getsign,divider;
if(bin2int_thread(ent->chromosome[_endtop])>8){getsign = -1;}else{getsign = 1;}

divider=10;//bin2dec_thread(ent->chromosome[_endtop+1],1)*1000; 

int toposize=200; //turn into array for more layers
if(_fixedtop==1){
topology=_hidval;	}
//calculate toposize
int in=2,out=1,hid=topology,nlayers=1,type=1,conn=0; 
					//get values from config
					//type 1 = all connect

for(int i=0;i<nlayers;i++){
conn+=hid*in+i*hid*hid;
}
conn+=out*in+nlayers*out*hid;

//hidden layer 1
//hid*in
//hidden layer 2
//hid*hid+hid*in
//hidden layer 3
//hid*hid+hid*hid+hid*in
//output layer
//out*hid+out*in

toposize=conn;

mynet->weights = new double[toposize+_endtop];

mynet->weights[0] = topology;
_wgtsize = toposize;

//Record Topology in Phenotype
ent->info[0]=topology;
ent->info[1]=toposize;

//Get Weights
for(int i=1;i<toposize;i++){
//mynet->weights[i] = bin2dec_thread(ent->chromosome[i+_endtop+_endfunc+2],1);
if(_wgtupdt==1){
mynet->weights[i] = getsign*bin2dec_thread(ent->chromosome[i+_endtop+_endfunc+2],1)/divider;
		}else{
mynet->weights[i]=0;		
		}
//cout<<i<<","<<mynet->weights[i]<<"\n";
}
	
m_pointertofunction(mynet->weights,&mynet->error,input_array);

//Gather data for energy landscape

ent->graphdata[0] = mynet->error;
ent->graphdata[1] = mynet->weights[1];
ent->graphdata[2] = mynet->weights[2];

//Minimise Error
if(mynet->error>0)
	   fit1=1/(1/mynet->error);     
           else
	   fit1=1/mynet->error;

ent->output[1] = mynet->error;
//thisfit += mynet->error;
thisfit = sqrt(pow(fit1,2));
ent->fitness += thisfit;  
ent->objfit[1] = thisfit;

//cout<<"Error="<<mynet->error<<"\n";
//cout<<"Weight Size="<<_wgtsize<<"\n";


    //printf("fitness = %f\n",ent->fitness);
     
	pthread_mutex_unlock (&mutexdata);
   	

     				}

static void* staticthreadproc(void* p){

//printf("%d\n",len_binary);
return reinterpret_cast<ga*>(p)->thread_function(p);

}

void eval_top(population *pop){
pop->evaltop=1;
eval_thread(pop);
//stats(pop);
pop->evaltop=0;
}

void eval_thread(population *pop){

//use multithreading to evaluate individuals in parallel 
//not to be used with selector ?

int i,j,k;
int nthreads = pop->size;
if(pop->evaltop==1){
nthreads = 1;}
entity *entity;
//this->index = new int[nthreads];
int index[nthreads];
pthread_t threads[nthreads];

//create threads
pthread_mutex_init(&this->mutexdata,NULL);
this->thread_index.clear();
this->whichthread=0;
this->currentpop = pop;
for(i=0;i<nthreads;i++){
index[i]=i;

pthread_mutex_lock (&mutexdata);
this->thread_index.push_back(i);
pthread_mutex_unlock (&mutexdata);

//pthread_create(&threads[i],NULL,&ga::staticthreadproc,(void*) pop->entity_array[i]);
pthread_create(&threads[i],NULL,&ga::staticthreadproc,this);
//

//printf("creating thread, %d\n",i);
}


//join threads
for(i=0;i<nthreads;i++){

pthread_join(threads[i],NULL);

}

/*printf("After"); 
for(i=0;i<pop->size;i++){

printf("Fitness %d = %f\n",i,pop->entity_array[i]->fitness);
} 
*/

//pointer_to_entity[2]->fitness = 1.009992;


//printf("After Pointer = %f Array = %f\n",pointer_to_entity[2]->fitness,pop->entity_array[2]->fitness);

     double sumfit=0;
      
      for(i=0;i<pop->size;i++){
      sumfit+=pop->entity_array[i]->fitness;
      pop->entity_array[i]->fitscale=0;
      				}
     
      //printf("Sumfit=%f\n",sumfit);
      double percent=0;
      
      for(i=0;i<pop->size;i++){
      //ent->fitscale=
      
      percent+=pop->entity_array[i]->fitscale= (pop->entity_array[i]->fitness/sumfit)*100;
      				
				}
				
     //average
     
     pop->nmean=((sumfit/pop->size)/sumfit)*100;
				
    //printf("\ntotal percent:%f nmean = %f\n",percent,pop->nmean);	




}



void eval(population *pop){

	double A=0,B=0,C=0,D=0,E=0,F=0;
	int i,j,k,l;
	double fit1,fit2,t4,sumfit;
	entity *entity;

	//entity = new entity;


   for(i=0;i<pop->size;i++){
  
  
  
        entity=pop->entity_array[i];
   
   
         A = bin2dec(pop,entity->chromosome[0],1);
         B = bin2dec(pop,entity->chromosome[1],1);
      // C = bin2dec(pop,entity->chromosome[2],1);
      // D = bin2dec(pop,entity->chromosome[3],1);
      // E = bin2dec(pop,entity->chromosome[4],1);
      // F = bin2dec(pop,entity->chromosome[5],1);
   
       t4 = func(A,B);
       
       if(max_def==0){
       //minimize
	   if(t4>0)
	   fit1=1/(1/t4);     
           else
	   fit1=1/t4;     
           	}else{
	//maximize	
	   if(t4>0){
	   fit1=1/t4;				//printf("greater than zero t4=%f\n",t4);
	   }else{
	   fit1=1/(1/(t4-5));			//printf("less than zero t4=%f\n",t4);
	   }
	         }
	
	 
      entity->output[0]   = t4;
      entity->fitness  = sqrt(pow(fit1,2)); 
      				
				
				}
  
      
      
      
      for(i=0;i<pop->size;i++){
      sumfit+=1/pop->entity_array[i]->fitness;
      pop->entity_array[i]->fitscale=0;
      				}
     
      //printf("Sumfit=%f\n",sumfit);
      double percent=0;
      
      for(i=0;i<pop->size;i++){
      pop->entity_array[i]->fitscale=percent+= ((1/pop->entity_array[i]->fitness)/sumfit)*100;
      				
				}
				
      printf("\ntotal percent:%f\n",percent);		       
      




}

void adapt(population *pop1,population *pop2){
// call x-over and mutate on pop1 to create pop2
int full=-1,total=-1,mother=1,father=2,mutate=1;
double prob;

while(total<pop2->size-2){

//select parents
//roullete_select(pop1,&mother,&father);
tournament(pop1,&mother,&father);
//random_select(pop1,&mother,&father);

prob = rand()%100;

if(mutate==1&&prob<mutat_def*100){
full+=1;
mutation(pop1,pop2,mother,father,full);			//includes cloning
total+=1;
mutlog++;			   
			    }else
			    	
if(prob<cross_def*100){
full+=2;
crossover(pop1,pop2,mother,father,full);
total+=2;
crosslog++;
		            }
	
else {
full+=1;
reproduce(pop1,pop2,mother,father,full);		//cloning
total+=1;
clonelog++;
}



}
//printf("total new individuals : %d\n",full);


}

void copy(population *pop1,population *pop2){
int i,j,k;

for(i=0;i<pop1->size;i++){

mem_copy(pop2,pop1,i,i);
}

}

void select_old(population *pop1,population *pop2){
//select from pop2 into pop1
int i,j,k,c=0,a=0;
population *temppop;
entity *entity1;
entity** newent;


do{
for(a=0;a<pop2->size;a++){
entity1 = pop2->entity_array[a];

if(entity1->fitscale<35&&c<pop1->size-1){

if(entity1->fitscale<15&&c<pop1->size-2){
mem_copy(pop2,pop1,c,a);c++;
mem_copy(pop2,pop1,c,a);c++;
}else{

mem_copy(pop2,pop1,c,a);c++;
		   
		  		   }   }


}			}while(c<pop1->size-1);

						}















void select(population *pop1,population *pop2){
//select from pop2 into pop1
int i,j,k,c=0,a=0,alpha,omega,roullete=1;
population *temppop;
entity *entity1;
entity** newent;

int a_array[pop2->size+1],count=0,selection;
double spin,fitsum;



do{

spin=(double) rand() / (RAND_MAX);
printf("spin=%f\n",spin);

for(a=0;a<pop2->size;a++){
entity1 = pop2->entity_array[a];

if(c<pop1->size-1){

if(entity1->fitscale<spin){

mem_copy(pop2,pop1,c,a);c++;
a_array[count]=a;
count++;

}else{

if(count>0){

selection=rand()%count;
//printf("this selection=%d\n",selection);
mem_copy(pop2,pop1,c,a_array[selection]);c++;

}else{

if(roullete==1){
roullete_select(pop2,&alpha,&omega);
mem_copy(pop2,pop1,c,alpha);c++;
a_array[count]=alpha;
count++;
}else{
selection=rand()%pop2->size;
mem_copy(pop2,pop1,c,selection);c++;
a_array[count]=selection;
count++;
	}
}
		   
		  		}    }  


}


			}while(c<pop1->size-1);

						}


void reproduce(population *pop1,population *pop2,int m,int f, int full){
entity *clone;
int i,j,k,l;

clone = pop2->entity_array[full];


if(rand()%100>50)
mem_copy(pop1,pop2,full,m);		//clone mother
else
mem_copy(pop1,pop2,full,f);           //clone father



//mutate
if(mutator==1){
//int prob = rand()%100;

//if(prob<mutat_def*100){mutate(pop2,0,clone);}
//mutate(pop2,0,clone);
}
}

void crossover(population *pop1,population *pop2,int m, int f,int full){
entity *mother,*father;
entity *son, *daughter;
int i,j,k,l;

mother  =pop1->entity_array[m];
father  =pop1->entity_array[f];
son     =pop2->entity_array[full];
daughter=pop2->entity_array[full-1];

//crossover (single-point)

int xpoint=-1,ypoint=-1,temp;
int npos = pop1->len_chromosomes-1;

if(xover_type==0){

for(k=0;k<pop1->num_chromosomes;k++){
xpoint=rand()%npos;
for(l=0;l<pop1->len_chromosomes;l++)  {
if(l>=xpoint)
{
son->chromosome[k][l]=mother->chromosome[k][l];
daughter->chromosome[k][l]=father->chromosome[k][l];
}else{
son->chromosome[k][l]=father->chromosome[k][l];
daughter->chromosome[k][l]=mother->chromosome[k][l];
}

					}
					
					}
		}
					
//crossover (two-point)
int prob;
					
if(xover_type==1){
					
for(k=0;k<pop1->num_chromosomes;k++){

do{
xpoint=(rand()%npos);
ypoint=(rand()%npos);
}while(xpoint==ypoint);

//printf("xpoint=%d  ypoint=%d",xpoint,ypoint);
prob=100;rand()%100;

if(xpoint>ypoint){temp=xpoint;xpoint=ypoint;ypoint=temp;}

for(l=0;l<pop1->len_chromosomes;l++)  {

if(prob>50){

if(ypoint>=l>=xpoint)
{
son->chromosome[k][l]=father->chromosome[k][l];
daughter->chromosome[k][l]=mother->chromosome[k][l];
}else{
son->chromosome[k][l]=mother->chromosome[k][l];
daughter->chromosome[k][l]=father->chromosome[k][l];
}
}else{
if(ypoint>=l>=xpoint)
{
son->chromosome[k][l]=mother->chromosome[k][l];
daughter->chromosome[k][l]=father->chromosome[k][l];
}else{
son->chromosome[k][l]=father->chromosome[k][l];
daughter->chromosome[k][l]=mother->chromosome[k][l];
}
}
					
					
					
					
					}
					
					}					
					
			}		
					
//mutate
if(mutator==1){
//int prob = rand()%100;

//if(prob<mutat_def*100){
mutate(pop2,0,son);mutate(pop2,0,daughter);
//}
}

									}


void mutate(population *pop,int type,entity *individual){

//entity *individual;
//individual = pop->entity_array[ent];
int i,j,bit,swap;
char temp;

for(i=0; i<pop->num_chromosomes; i++){

swap = rand()%(pop->len_chromosomes-1);

for(j=0; j<pop->len_chromosomes; j++){

if(rand()%100<mutat_def*100){
bit=j;
}

//bit flipping 1/len_chromosome chance bit is flipped!!
if(j==bit&&flip_mut==1){

if(individual->chromosome[i][j]=='1'){
individual->chromosome[i][j]='0';}else{
individual->chromosome[i][j]='1';
}


					}

if(j==bit&&swap_mut==1){
temp=individual->chromosome[i][j];
individual->chromosome[i][j]=individual->chromosome[i][swap];
individual->chromosome[i][swap]=temp;
				
					}
					
					
						}
					
						}
					}



void mutation(population *pop1,population *pop2,int m, int f,int full){
entity *mother,*father;
entity *son, *daughter;
int i,j,k,s=full,d=full-1,bit,sex;

mother  =pop1->entity_array[m];
father  =pop1->entity_array[f];

sex=rand()%100/50;

if(sex>50){
mem_copy(pop1,pop2,s,f);}else{
mem_copy(pop1,pop2,s,m);}

son     =pop2->entity_array[s];
//daughter=pop2->entity_array[d];


//if(sex>50){
mutate(pop2,0,son);
//}else{
//mutate(pop2,0,daughter);
//}

						}




									
									
									
void roullete_select(population *pop,int *mother,int *father){

double* prob;
prob = new double[pop->size];
int i,j,k,l,c=0,mchoice=0,fchoice=0;
double psum=0,sum=0,throw1,throw2,probability;


sorted(pop);
//calc prob
for(i=pop->size-1;i>=0;i--)
psum+=1/pop->entity_array[i]->fitness;

for(i=pop->size-1;i>=0;i--){

//for(i=0;i<pop->size-1;i++){
prob[i]=sum+=probability=(1/(pop->entity_array[i]->fitness))/psum;
//c++;
//printf("Individual =%d Sum = %f Prob= %f Fitness =%f\n",i,prob[i],probability,pop->entity_array[i]->fitness);
}

int ft,mt,nochoice=0,count=0;
 

 do{
 count++;
  throw1 =(double) (rand()%1000)/1000;
  throw2 =(double) (rand()%1000)/1000;
  mchoice=0;
  fchoice=0;
 
 
 
 for(k=pop->size-1; k>=0; k--){
 if (throw1<=prob[k]&&mchoice==0){
 
 mt=k;mchoice=1;}
 }
 
 for(l=pop->size-1; l>=0; l--){
 if (throw2<=prob[l]&&fchoice==0){	
 ft=l;fchoice=1;}
 }
 
 //printf("fchoice = %d mchoice = %d\n",fchoice,mchoice);
 
 //if(count>pop->size*10){nochoice=1;break;}
 
 }while(ft==mt||fchoice==0||mchoice==0);
 
 *mother=mt;*father=ft;
 
 
 //printf("mother %d father %d\nfitness of mother = %f with prob %f\nfitness of father = %f with prob %.4f\n",*mother,*father,pop->entity_array[*mother]->fitness,prob[*mother],pop->entity_array[*father]->fitness,prob[*father]);
//printf("%.4f           %.4f      %.4f          %.4f      \n",1-(pop->entity_array[*mother]->fitness/100),1-(pop->entity_array[*father]->fitness/100),prob[*mother],prob[*father]);
 }
 
 
void tournament(population *pop, int *mother, int *father)
    {
	
int m1,m2,f1,f2;

do{
m1=rand()%(pop->size);
m2=rand()%(pop->size);
f1=rand()%(pop->size);
f2=rand()%(pop->size);
}while(m1==m2||m2==f1||m2==f2||f1==m1||f1==f2);
//printf("*************************************M1%d  M2%d  F1%d   F2%d\n",m1,m2,f1,f2);
if ((pop->entity_array[m1]->fitness)<(pop->entity_array[m2]->fitness)){
*mother=m1;}
else 
*mother=m2;

if ((pop->entity_array[f1]->fitness<pop->entity_array[f2]->fitness)){
*father=f1;}
else
*father=f2;

}


void random_select(population *pop, int *mother, int *father)
{ 


  
  *mother=(int )rand()%pop->size;   
    
    
  
do{
  *father=(int )rand()%pop->size; 
   }while(father==mother);     
}


void rank_scale(population *pop){

int i,n;
//convert all fitness to ranked fitness
sorted(pop);

n=pop->size;
for(i=0;i<pop->size;i++){
n--;
pop->entity_array[i]->fitness = (double) 1/(sqrt(n+1));
}
 }
 
void sorted(population *pop)

{

entity *clone[pop->size];
int i,j,sorted,array[pop->size],c=0,temp;

for(i=0;i<pop->size;i++){
array[i]=i;
			}



do{
 sorted=1;
for(i=0; i<pop->size-1; i++){
if (pop->entity_array[array[i]]->fitness > pop->entity_array[array[i+1]]->fitness){

temp = array[i];
array[i] = array[i+1];
array[i+1] = temp;

sorted=0;
			}
			
			
		}
		
	} while (sorted==0);
	
for(i=0;i<pop->size;i++)clone[i] = pop->entity_array[i];

for(i=0;i<pop->size;i++)pop->entity_array[i]=clone[array[i]];

//turn sort upside down


//for(i=0,j=pop->size-1;i<pop->size;i++,j--)pop->entity_array[i]=clone[array[j]];



	
//for(j=0; j<pop->size; j++){
//printf("fitness of %d = %.2f\n",j,pop->entity_array[j]->fitness);
//}	


		} 

									
void mem_copy(population *pop1,population *pop2, int copy, int orig)
{

int i,k;
for(k=0; k<pop1->num_chromosomes; k++){

for(i=0; i<pop1->len_chromosomes; i++){

pop2->entity_array[copy]->chromosome[k][i]=pop1->entity_array[orig]->chromosome[k][i];

}
}

pop2->entity_array[copy]->fitness=pop1->entity_array[orig]->fitness;
pop2->entity_array[copy]->fitscale=pop1->entity_array[orig]->fitscale;


}									
									
 void stats(population *pop){
  
  system("clear");
  double std,sum,A,B,t4;
  int i,j,k,x,size=iterations/2,count=0;
  
  sorted(pop);
  
  for(i=0;i<pop->size;i++){
 //printf("entity %d fitness = %f fitscale = %f\n",i,pop->entity_array[i]->fitness,pop->entity_array[i]->fitscale);
  
  }
  
  double prob,temp_,val_,sum_=0,min_=0.1,max_=1;
  int sel_;
  
  for(int i=0;i<_endtop;i++){
  val_ = bin2dec_thread(pop->entity_array[0]->chromosome[i+_endfunc],5);
  if(val_>max_){max_=val_;}
  if(val_<min_){min_=val_;}
  
  sum_+=bin2dec_thread(pop->entity_array[0]->chromosome[i+_endfunc],5);
  }
  if(pop->evaltop==1){
  printf("Annealing !!\n");
		      }
  printf("Most fit individual has: %f Least fit has: %f Output : %f\n",pop->entity_array[0]->objfit[1],pop->entity_array[pop->size-1]->objfit[1],pop->entity_array[0]->output[0]);
  printf("Most fit Fitscale = %f Least fit Fitscale = %f\n",pop->entity_array[0]->fitscale,pop->entity_array[pop->size-1]->fitscale);  
  printf("Most fit individual has: %d Neurons in Hidden layer and %d Weights changed.\n",pop->entity_array[0]->info[0],pop->entity_array[0]->info[1]);
  printf("Binary Encoded Topology:\n");
  printf("Min=%f Max=%f     _|Value   Prob    Select?    Temp\n",min_,max_);
  for(int i=0;i<_endtop;i++){
  
  val_=pop->entity_array[0]->phenotype[i][0];
  prob=pop->entity_array[0]->phenotype[i][1];
  sel_=pop->entity_array[0]->phenotype[i][2];
  temp_=pop->entity_array[0]->phenotype[i][3];
   
   
  //val_  = bin2dec_thread(pop->entity_array[0]->chromosome[i+_endfunc],5);
  //temp_ = pop->Temp; 
  //prob = (val_-min_)/(max_-min_);
  //sel_ = prob*100>rand()%100;
  
  //prob  = 1/(1+exp(((bin2dec_thread(pop->entity_array[0]->chromosome[i+_endfunc],5))/pop->Temp)));
  //sel_  = (1/(1+exp(((bin2dec_thread(pop->entity_array[0]->chromosome[i+_endfunc],5))/pop->Temp))))*100>rand()%100;
  
  
  printf("Chromo %d =                  %f,| %f, | %d, | %f\n",i,val_,prob,sel_,temp_);
  //printf("Chromo %d = %d",i,bin2int_thread(pop->entity_array[0]->chromosome[i+_endfunc]));
}
  printf("Topology=%d",(int)pop->entity_array[0]->phenotype[_endtop][0]);
  printf("\n");
			
    A = bin2dec(pop,pop->entity_array[0]->chromosome[0],1);
    B = bin2dec(pop,pop->entity_array[0]->chromosome[1],1);
    t4 = func(A,B);
    printf("\n%f = FUNC(%f,%f)\n",t4,A,B);    
   
  pop->best = pop->entity_array[pop->size-1]->objfit[1]; 
   
  //find best overall score
 
  if(pop->best<previousbest){
  positer++;
 
 if (positer>size){
  positer=0;
  }
  previousbest=pop->bestscores[positer-1]=pop->best;
  }
  
  
 for(j=0;j<positer;j++){
  //printf("best score %d = %f\n",j,bestscores[j]);
  count=0;
  for(i=0;i<positer;i++){
  if(pop->bestscores[j]<pop->bestscores[i]){
  count++;
  }
  }
  if(count>bscore){
  pop->bestvalue=pop->bestscores[j];
  bscore=count;
  }
  
  	}
  
 
  //compute average
  
  for(x=0;x<pop->size;x++){
  sum+=pop->entity_array[x]->objfit[1];
  }
  
  for(x=0;x<pop->size;x++){
  std+=pow(pop->entity_array[x]->objfit[1]-(sum/pop->size),2);
  }
  
  std = sqrt(std/pop->size);
  
  
  printf("\nMean fitness = %f Stdeviation = %f Popsize = %d Sum = %f\n",sum/pop->size,std,pop->size,sum);
  
  pop->mean = sum/pop->size;
 
/* 
for(int i=0;i<pop->size;i++){
printf("Fitness for %d = %f",i,pop->entity_array[i]->objfit[1]);
}
printf("\n");
*/

//int i,j,k;  
for(j=0; j<pop->num_chromosomes; j++){
for(k=0; k<pop->len_chromosomes; k++){
if(j<10){
printf("%c",pop->entity_array[0]->chromosome[j][k]);
}
}
if(j<10){
printf("\n");
	}
}
double x_,y_,z_;

x_ = pop->entity_array[0]->graphdata[2];
y_ = pop->entity_array[0]->graphdata[1];
z_ = pop->entity_array[0]->graphdata[0];

printf("crossed : %d mutted : %d cloned : %d\n",crosslog,mutlog,clonelog);
  
printf("Energy Data: x ,   y ,   z\n");
printf("            %f    %f    %f\n",z_,y_,x_);  
outfile<<y_<<" "<<x_<<" "<<z_<<"\n"; 
errplot<<pop->entity_array[0]->objfit[1]<<"\n";  


  }




/*
for(i=0; i<pop1->num_chromosomes; i++){

bit=rand()%pop1->len_chromosomes;

for(j=0; j<pop1->len_chromosomes; j++){


//actually need to change this to bit flipping 1/len_chromosome chance bit is flipped!!

if(sex>50){
if(j==bit){

if(son->chromosome[i][j]=='1'){
son->chromosome[i][j]='0';}else{
son->chromosome[i][j]='1';
}


}
	}else{
if(j==bit){
if(daughter->chromosome[i][j]=='1'){
daughter->chromosome[i][j]='0';}else{
daughter->chromosome[i][j]='1';
}
}	
	
}

}}
*/


		};  //end class ga











