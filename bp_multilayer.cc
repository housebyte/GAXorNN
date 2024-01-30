#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#define L1 18		//Layer 1 CNN
#define Filt 6		//Filter size
#define Strd 1		//Stride
#define L2 (L1-Filt)/Strd+1 //Layer 2 CNN
#define isa_cnn 0	//Are we using the CNN
#define reg 0
#define outreg 0
#define t_  1000             //Weight Update Divider
#define t2_ 0.001            //Inputs Divider
#define t3_ 1000           //Bellman Divider
#define t4_ 1000    //Random decay Multiplyer
using namespace std;

const int show_err = 0;
const int show_rnd =0;
int randcy = 1;
int seed =1;
const int numInputs = 7;        // Input nodes, plus the bias input.
const int numOutputs = 7;
const int numPatterns = 4;      // Input patterns for XOR experiment.

const int numHidden =L1;	//Layer 1 for CNN
const int numhidlayers =2;	//Number of hidden layers
double thisseed;
double LR_IH = 0.75;       // Learning rate, input to hidden weights.
double LR_HO = 0.75;      // Learning rate, hidden to output weights.
double LR_HH = 0.75;      // Learning rate, hidden to output weights.
const double Mntem=0.00175,dcay=0.000000065,Gain_=1;
int t__=0;
double rnd_decay=1;
int patNum = 0;
double errThisPat = 0.0;
double outPred = 0.0;                   // "Expected" output values.
double RMSerror = 0.0;                  // Root Mean Squared error.

struct RLnet{
double *outVal;
double *hiddenVal;    // Hidden node outputs.
double *hiddenVal_;   //Layer 2 for CNN

double *ErrOut,*errH1,*errH2,*errIn;
double **weightsIH; // Input to Hidden weights.
double **prewgtIH;
double **predltIH;

double **weightsHH;	//Hiiden to Hidden
double **prewgtHH;
double **predltHH;

double *weightsCNN;	//CNN Layer 1 to CNN Layer 2
double *prewgtCNN;	//FieldSize = NumNeurons Layer 1
double *predltCNN;	//

double **weightsHO;    // Hidden to Output weights.
double **prewgtHO; 
double **predltHO;
double *InputsNow;			//RL Input values at time t

double (*act_l1)(double value,int diff,double y); //hid1 act
double (*act_l2)(double value,int diff,double y); //hid2 act
double (*act_l3)(double value,int diff,double y); //Output act


};

int numEpochs = 1000;
double **trainInputs;
double  **trainOutput;           // "Actual" output values.
//ofstream err;


// Function Prototypes.
void initWeights(RLnet* net );
void calcNetRL(RLnet* net,double * currentState,double action,double * BellmanQ);
void WeightChangesHO(RLnet* net);
void WeightChangesHH(RLnet* net);
void WeightChangesIH(RLnet* net);
void calcOverallError();
void displayResults();
double getRand();
double relu(double value,int diff,double y);
double sigma(double value,int diff,double y);
double tanh_(double value,int diff,double y);
double ident(double value,int diff,double y);
double invsqr(double value,int diff,double y);
double softplus(double value,int diff,double y);
double bent(double value,int diff,double y);



void initWeights(RLnet* net){

net->outVal = new double[numOutputs];
net->ErrOut = new double[numOutputs];
net->errH1 = new double[L2];
net->errH2 = new double[L2];
net->errIn  = new double[numInputs];
net->weightsIH = new double*[numInputs];
net->prewgtIH = new double*[numInputs];
net->predltIH = new double*[numInputs];
for(int i=0;i<numInputs;i++){
net->weightsIH[i] = new double[numHidden];
net->prewgtIH[i] = new double[numHidden];
net->predltIH[i] = new double[numHidden];
}
net->weightsHH = new double*[L1];	//Hiiden to Hidden
net->prewgtHH = new double*[L1];
net->predltHH = new double*[L1];
for(int i=0;i<L1;i++){
net->weightsHH[i] = new double[L1];	//Hiiden to Hidden
net->prewgtHH[i] = new double[L1];
net->predltHH[i] = new double[L1];
}

net->hiddenVal  = new double[numHidden];
net->hiddenVal_ =new double[L2];
net->weightsCNN = new double[Filt];
net->prewgtCNN = new double[Filt];
net->predltCNN = new double[Filt];
net->weightsHO = new double*[L1];
net->prewgtHO = new double*[L1];
net->predltHO = new double*[L1];
for(int i=0;i<L1;i++){
net->weightsHO[i] = new double[numOutputs];
net->prewgtHO[i] = new double[numOutputs];
net->predltHO[i] = new double[numOutputs];
}
net->InputsNow = new double[2];

net->act_l1 = sigma;
net->act_l2 = tanh_; //Not used for single layer
net->act_l3 = sigma;

trainInputs = new double*[numPatterns];
trainOutput = new double*[numPatterns];
for(int i=0;i<numPatterns;i++){
trainInputs[i] = new double[numInputs]; 
trainOutput[i] = new double[numOutputs]; 
}

// Initialize weights to random values.
int count=-1,endFilt=0;
    for(int j = 0; j < numHidden; j++){

 for(int l=0;l<numOutputs;l++){
 
	net->predltHO[j][l] = (getRand() - 0.5) / 2;
	net->prewgtHO[j][l] = (getRand() - 0.5) / 2;
        net->weightsHO[j][l] = (getRand() - 0.5) / 2;
	
	//cout << "Ouput Weight="<<weightsHO[j]<<"\n";
        					}
	
	for(int i = 0; i < numInputs; i++){
	    
	    net->predltIH[i][j] = (getRand() - 0.5) / 5;
	    net->prewgtIH[i][j] = (getRand() - 0.5) / 5;
            net->weightsIH[i][j] = (getRand() - 0.5) / 5;

            //cout << "Input Weight = " << weightsIH[i][j] << endl;
        }


int n;
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

	for(int k = 0; k < n; k++){
	

	    net->predltHH[j][k] = (getRand() - 0.5) / 5;
	    net->prewgtHH[j][k] = (getRand() - 0.5) / 5;
	    net->weightsHH[j][k] = (getRand() - 0.5) / 5;

if(isa_cnn==1){	
count++;if(count==Filt){count=0;endFilt=1;}
if(endFilt==0){
net->weightsCNN[count]= (getRand() - 0.5) / 5;	    
		}

	    //cout << "CNNWeight"<<count<<" = " << net->weightsCNN[count] << endl;
}else{
	    //cout << "Hidden Weight = " << net->weightsHH[j][k] << endl;
	}	
}


    }



}

void save_seed(RLnet* net,const char *fname)
{

FILE *fout;
fout=fopen(fname, "a");
if(fout == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
//Save random seed
fprintf(fout,"%f\n",thisseed);

}

void load_seed(RLnet* net,const char *fname,int seedline){
double x;
int count=0;
char line[100];
FILE *fh;
fh=fopen(fname, "r");
if(fh == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
while( fgets( line,100,fh ) ){
if( 1==sscanf(line,"%lf",&x) ){
count++;
if(seedline==0){
thisseed=x;
}else{
if(count==seedline){
thisseed=x;
}						}

					}
						}
}


void save_weights(RLnet* net,const char *fname)
{

FILE *fout;
fout=fopen(fname, "w");
if(fout == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
//Save random seed
fprintf(fout,"%f\n",thisseed);

//Hidden to output weights:   (Change for struct layers)
for(int j = 0; j < numHidden; j++){
for(int l=0;l<numOutputs;l++){
fprintf(fout,"%f\n",net->weightsHO[j][l]);
						}
					    }
//Input to hidden:					    
for(int j = 0; j < numHidden; j++){
for(int i=0;i<numInputs;i++){
fprintf(fout,"%f\n",net->weightsIH[i][j]);
						}
					}

int n,count=0;	
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

//Change for struct layer ***

for(int j = 0; j < numHidden; j++){
for(int k = 0; k < n; k++){
	if(isa_cnn==1){
	
	count++;if(count==Filt){}else{
	fprintf(fout,"%f\n",net->weightsCNN[count]);
}
}else{
	
	fprintf(fout,"%f\n",net->weightsHH[j][k]);
	    
      }	
	}
	
						}
}

void load_weights(RLnet* net,const char *fname){
char c;
double x;
int count=0;
int nH;
int cnn_cnt=0;
if(isa_cnn==1){
nH = L2;
}else{
nH = numHidden;
}


FILE *fh;
fh=fopen(fname, "r");
if(fh == NULL){
printf("Unable top open file %s\n",fname);
exit(1);
}
int HO = numHidden*numOutputs;
int IH = HO + numInputs*numHidden;
int i=0,j=0;
char line[100];
int scannedseed=0;
while( fgets( line,100,fh ) ){
if( 1==sscanf(line,"%lf",&x) ){
if(count==0&&scannedseed==0){
thisseed=x;
count=-1;
scannedseed=1;
}else{
if(count==0){
i=0;
j=0;
}
if(count<HO){
if(i==numOutputs){i=0;j+=1;}
net->weightsHO[j][i]=x;
//cout<<"Weights HO "<<j<<","<<i<<" "<<x<<"\n";
i++;
}
if(count==HO){
i=0;
j=0;
		      }
if(count>=HO&&count<IH){
if(i==numInputs){i=0;j+=1;}
net->weightsIH[i][j]=x;
//cout<<"Weights IH "<<j<<","<<i<<" "<<x<<"\n";
i++;
}

if(count==IH){
i=0;
j=0;
		      }
if(count>=IH){
if(i==numHidden){i=0;j+=1;}
if(isa_cnn==1){
	
	cnn_cnt++;if(cnn_cnt==Filt){}else{
	net->weightsCNN[count]=x;
							}
}else{
net->weightsHH[j][i]=x;
}
i++;
}


//cout<<"Count:"<<count<<"Value:"<<i<<"\n";
					}//Seed
count++;
						}

							}

fclose(fh);
}


void printWeights(RLnet* net){
// Initialize weights to random values.
int count=0;

    for(int j = 0; j < numHidden; j++){

for(int l=0;l<numOutputs;l++){

        cout << "Hidden 2 Layer:"<<j<<" Ouput Weight="<<net->weightsHO[j][l]<<"\n";
}	
        for(int i = 0; i < numInputs; i++){

            
            cout <<"Input:"<<i<<"Hidden L1:"<<j<< " Weight = " << net->weightsIH[i][j] << endl;
        }
int n;	
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

if(numhidlayers>1){

	for(int k = 0; k < n; k++){
	if(isa_cnn==1){
	
	count++;if(count==Filt){count=0;}
	cout <<"Hidden 1 Layer :" <<j<<"Hidden 2 Layer :"<<k<< "CNNWeight****** = " << net->weightsCNN[count] << endl;
}else{
	    cout <<"Hidden 1 Layer :" <<j<<"Hidden 2 Layer :"<<k<< "Hidden Weight = " << net->weightsHH[j][k] << endl;
      }	
	}
	
	}
    }
}

void calcNetRL(RLnet* net,double * StateArray,double action,double * BellmanQ){
// Calculates values for Hidden and Output nodes.

//Load Inputs
for(int i=0;i<numInputs;i++){
net->InputsNow[i] = StateArray[i];
}

//Compute First Hidden Layer Act

    for(int i = 0; i < numHidden; i++){
	net->hiddenVal[i] = 0.0;
	for(int j=0;j<numInputs;j++){
	net->hiddenVal[i] = net->hiddenVal[i] + ((net->InputsNow[j]/t2_)* net->weightsIH[j][i]);
 						}
	net->hiddenVal[i] = net->act_l1(net->hiddenVal[i],0,0);
		
    }
   
   
int count =0;    
int n;
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

//Compute Later Hidden Layer Act


if(numhidlayers>1){

    for(int i = 0; i < n; i++){
	  net->hiddenVal_[i] = 0.0;

    for(int j = 0; j < numHidden; j++){
    if(isa_cnn==1){
	count++;if(count==Filt){count=0;}
        net->hiddenVal_[i] = net->hiddenVal_[i] + (net->hiddenVal[j] * net->weightsCNN[count]);
	
}else{    
	net->hiddenVal_[i] = net->hiddenVal_[i] + (net->hiddenVal[j] * net->weightsHH[j][i]);
      } 
        			       }

       
	net->hiddenVal_[i] = net->act_l2(net->hiddenVal_[i],0,0);
    }
   
   		}

    //Compute Output
    
    outPred = 0.0;

if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}
   
double sum;
for(int j=0;j<numOutputs;j++){
sum =0;
 for(int i = 0; i < n; i++){

if(numhidlayers==1){
	sum += net->hiddenVal[i] * net->weightsHO[i][j];
}else{ 
        sum += net->hiddenVal_[i] * net->weightsHO[i][j];
	}				
					}
         net->outVal[j] =sum;
		}

errThisPat=0;
double err;
for(int j=0;j<numOutputs;j++){
net->outVal[j] = net->act_l3(net->outVal[j],0,0);
err = BellmanQ[j] - net->outVal[j];
net->ErrOut[j] = net->act_l3(err,1,net->outVal[j]);
//net->ErrOut[j] = sqrt(pow(BellmanQ[j]/t3_ - net->outVal[j],2));
errThisPat+=0.5*(pow(net->ErrOut[j],2));

if(show_err==1){
cout<<j<<" Err"<<net->ErrOut[j]<<"  Target"<<BellmanQ[j]<<" Out"<<net->outVal[j]<<"\n";
		}
							}
if(show_err==1){
cout<<"_____________________________\n";  
			}
outPred = net->outVal[0];   

}

void WeightChangesHO(RLnet* net){  

//Decaying Noisy Act function proportional to error 
t__+=1;
rnd_decay = exp((-.8765/t4_)*(1/sqrt(pow(errThisPat,2)))*(t__)); 
//Make this proportional to error *** 
if(show_rnd==1){
cout<<rnd_decay<<" = decay **** " <<1/sqrt(pow(errThisPat,2))<<"  1/err\n";
			}

//Adjust the Hidden to Output weights.

int n;
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

double sum;
    
for(int k = 0; k < n; k++){

//Calc error for Layer before Output
sum=0;
for(int h=0;h<numOutputs;h++){    
sum+= net->ErrOut[h] * net->weightsHO[k][h];       
}
	
	if(numhidlayers==1){			//Using only L1 and L3
	net->errH1[k] = net->act_l1(sum,1,net->hiddenVal[k])*Gain_;
	}else{
	net->errH2[k] = net->act_l2(sum,1,net->hiddenVal_[k])*Gain_;
	}
//cout<<"Err h1 = "<<net->errH1[k]<<"\n";

}

for(int k = 0; k < n; k++){

//Do Weight change
for(int h=0;h<numOutputs;h++){
double weightChange=0;
double delta=0;

	if(numhidlayers==1){
	delta = weightChange = LR_HO * net->ErrOut[h] * net->hiddenVal[k];
	}else{
	weightChange = LR_HO * net->ErrOut[h] * net->hiddenVal_[k];
	}
	weightChange = (weightChange + Mntem*net->predltHO[k][h] - dcay*net->prewgtHO[k][h])/t_;
        net->weightsHO[k][h] = net->weightsHO[k][h] + weightChange;

        // Regularization of the output weights.
if(outreg==1){
        if (net->weightsHO[k][h] < -0.5){
            net->weightsHO[k][h] = -0.5;
        }else if (net->weightsHO[k][h] > 0.5){
            net->weightsHO[k][h] = 0.5;
        }
		}
	net->prewgtHO[k][h] = net->weightsHO[k][h];
	net->predltHO[k][h] = weightChange;

		}
    }

}

void WeightChangesHH(RLnet* net){    					//Optional CNN 
// Adjust the Input to Hidden weights.
int count =-1;
int n;
if(isa_cnn==1){
n = L2;
}else{
n = numHidden;
}

//Calc errH1
double sum=0;
count=0;
     for(int i = 0; i < numHidden; i++){

        for(int k = 0; k < n; k++){
	
if(isa_cnn==1){
	count++;if(count==Filt){count=0;}

sum+= net->errH2[k]*net->weightsCNN[count];	
	}else{

sum+= net->errH2[k]*net->weightsHH[i][k];	
	       }
							}
				
net->errH1[i] = net->act_l1(sum,1,net->hiddenVal[i])*Gain_; //Include Activation of Neuron in H1
	
	}
count=0;
    for(int i = 0; i < n; i++){

        for(int k = 0; k < numHidden; k++){
	double x=0;   
	   if(isa_cnn==1){
	count++;if(count==Filt){count=0;}
	x = net->errH2[i] * LR_HH;
	}else{
	x = net->errH2[i] * LR_HH;
           }
	   x = x * net->hiddenVal[k];
	   double weightChange = x;

if(isa_cnn==1){
	count++;if(count==Filt){count=0;}
        weightChange = (weightChange + Mntem*net->predltCNN[count] - dcay*net->prewgtCNN[count])/t_;
        net->weightsCNN[count] = net->weightsCNN[count] + weightChange;
 if(reg==1){       
  if (net->weightsCNN[count] < -0.5){
            net->weightsCNN[count] = -0.5;
        }else if (net->weightsCNN[count] > 0.5){
            net->weightsCNN[count] = 0.5;
        }
}
	net->prewgtCNN[count] = net->weightsCNN[count];
	net->predltCNN[count] = weightChange;


}else{
	    weightChange = weightChange + Mntem*net->predltHH[k][i] - dcay*net->prewgtHH[k][i];
            net->weightsHH[k][i] = net->weightsHH[k][i] - weightChange;
if(reg==1){        
	    if (net->weightsHH[k][i] < -0.5){
            net->weightsHH[k][i] = -0.5;
        }else if (net->weightsHH[k][i] > 0.5){
            net->weightsHH[k][i] = 0.5;
        }
}
	    net->prewgtHH[k][i] = net->weightsHH[k][i];
	    net->predltHH[k][i] = weightChange;
     }

	}
    }
}

void WeightChangesIH(RLnet* net){
// Adjust the Input to Hidden weights.

double sum=0;
double x=0;
    for(int i = 0; i < numInputs; i++){

        for(int k = 0; k < numHidden; k++){
    
	    x =   net->errH1[k] * LR_IH;
	    x = x * net->InputsNow[i];
	   double weightChange = x;
	    weightChange = (weightChange + Mntem*net->predltIH[i][k] - dcay*net->prewgtIH[i][k])/t_;
            net->weightsIH[i][k] = net->weightsIH[i][k] + weightChange;
     
     if(reg==1){        
	    if (net->weightsIH[i][k] < -0.5){
            net->weightsIH[i][k] = -0.5;
        }else if (net->weightsIH[i][k] > 0.5){
            net->weightsIH[i][k] = 0.5;
        }
}
        
	net->prewgtIH[i][k] = net->weightsIH[i][k];
	net->predltIH[i][k] = weightChange;
	
	}
    }
}


double relu(double value,int diff,double y){
double act;
if(diff==0){

//if(value<0){act=0;}
if(value<0){act=0.1*value;}
//if(value<0){act=0.01*(exp(value)-1);}
if(value>=0){act=value;}

	    }else{
	    
//if(value<0){act=0;}
if(value<-5.5){act=0.01;}
if(value>=0){act=1;}	    
	    }
return act;

}

double invsqr(double value,int diff,double y){
double act;
double rnum;
rnum=(double)(rand()%1000)/1000;
//rnum =1.647;
if(diff==0){
if(value<0){act=value/sqrt(1+rnum*pow(value,2));}
if(value>=0){act=value;}
}else{
if(value<0){act=pow(1/sqrt(1+rnum*pow(value,2)),3);}
if(value>=0){act=1;}

}

return act;
}

//Record random number at end of decay
//Set this as the value for rnum during recall
//Implement Srelu with x,y,z all being random numbers found using above method
//Theory after training and saving with different seeds can we recall different solutions
//From the same net

double sigma(double value,int diff,double y){
double act;
double rnum;
rnum=(double)(rand()%1000)/1000;
rnum = 1-(rnum*rnd_decay)/100;
if(rnum<0.99999&&randcy==1){thisseed=rnum;}
//cout<<rnum<<"\n";
if(randcy==0){
rnum=1;
		  }
if(seed==1){
thisseed=rnum=1-(double)(rand()%10)/100000;
}
if(seed==2){
rnum=thisseed;
}
if(diff==0){

act = (1/(1+exp(-value*rnum)));

	    }else{

act = y*(1-y)*value;	    
//act = 1-(1/(1+exp(-value)));
	    
	    }
return act;

}

double tanh_(double value,int diff,double y){
double act;
if(diff==0){

act = tanh(value);

	    }else{
	    
act = (1-pow(tanh(y),2))*value;
	    
	    }
return act;

}
double ident(double value,int diff,double y){
if(diff==0){
return value;
}else{
return 1;
}
}
double softplus(double value,int diff,double y){
double act;
double rnum;
rnum=(rand()%200)/100;
if(diff==0){
log(1+exp(value));
}else{
act = 1/(1+exp(-value));
}


return act;
}


double bent(double value,int diff,double y){
double act;
double rnum;
rnum=(rand()%200)/100;
if(diff==0){
act = (sqrt(pow(value,2)+1)-1)/2;
act = act + value;
}else{
act =(value/(2*sqrt(pow(value,2)+1)))+1;
}
return act;
}




double getRand(){
    return double(rand() / double(RAND_MAX));
}

/*******************************************Train Xor*****************************************/

void calcOverallError(RLnet* net){
    RMSerror = 0.0;

    for(int i = 0; i < numPatterns; i++){
         patNum = i;
          calcNetRL(net,trainInputs[i],5,trainOutput[i]);
         RMSerror = RMSerror + (errThisPat * errThisPat);
    }

    RMSerror = RMSerror / numPatterns;
    RMSerror = sqrt(RMSerror);
}

void displayResults(RLnet* net){
    for(int i = 0; i < numPatterns; i++){
        patNum = i;
        calcNetRL(net,trainInputs[i],5,trainOutput[i]);
        cout << "pat = " << patNum + 1 << 
                " actual = " << trainOutput[patNum][0] << 
                " neural model = " << outPred << endl;
    }
}

void initData(){
    // The data here is the XOR data which has been rescaled to 
    // the range -1 to 1.

    // An extra input value of 1 is also added to act as the bias.

    // The output must lie in the range -1 to 1.

    trainInputs[0][0]   =  1;
    trainInputs[0][1]   = 0;
    trainInputs[0][2]   =  1; // Bias
    trainOutput[0][0]      =  0.5;

    trainInputs[1][0]   = 0;
    trainInputs[1][1]   =  1;
    trainInputs[1][2]   =  1; // Bias
    trainOutput[1][0]      =  0.5;

    trainInputs[2][0]   =  1;
    trainInputs[2][1]   =  1;
    trainInputs[2][2]   =  1; // Bias
    trainOutput[2][0]      = -0.5;

    trainInputs[3][0]   = 0;
    trainInputs[3][1]   = 0;
    trainInputs[3][2]   =  1; // Bias
    trainOutput[3][0]      = -0.5;
}

/*
int main(){

err.open("errfile.dat");

    srand((unsigned)time(0));   // Seed the generator with system time.
        
	RLnet *net;
	net = new RLnet;

    	initWeights(net);

	    initData();

printWeights(net);

cout<<"Training about to commence\n";

    // Train the network.
    for(int j = 0; j <= numEpochs; j++){

        for(int i = 0; i < numPatterns; i++){

            //Select a pattern at random.
            patNum = rand() % numPatterns;
	    
            //Calculate the output and error for this pattern.
            calcNetRL(net,trainInputs[patNum],5,trainOutput[patNum]);

            //Adjust network weights.
            WeightChangesHO(net);
	    if(numhidlayers>1){
	    WeightChangesHH(net);
	    }
            WeightChangesIH(net);
        }

        calcOverallError(net);

        //Display the overall network error after each epoch
        cout << "epoch = " << j << " RMS Error = " << RMSerror << endl;
	err<<RMSerror<<"\n";
    }
    //Training has finished.
//    printWeights(net);
    displayResults(net);

err.close();

    return 0;
}

*/
