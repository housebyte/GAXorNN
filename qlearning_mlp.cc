
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include "bp_multilayer.cc"
//#include "backprop_state_to_act.cc"
//#include "backprop_xor.cc"

using namespace std;

const int qSize = 7;
const double gAmma = 0.8;
const int iterations = 278;
const int memsize = 78;
const int show_move = 0;
const int use_R = 0;
const double rnd_explor =0.755;
double cum_reward=0; //Cumultive reward
double spc = 0.45;
/*
int initialStates[qSize] = {1, 3, 5, 2, 4, 0};

double R[qSize][qSize] =  {{-1, -1, -1, -1, spc, -1},
			{-1, -1, -1, spc, -1, 1},
			{-1, -1, -1, spc, -1, -1},
			{-1, spc, -1, -1, spc, -1},
			{-1, -1, -1, spc, -1, 1},
			{-1, spc, -1, -1, spc, 1}};
			
*/

int initialStates[qSize] = {1, 3, 5, 2, 4, 0, 6};

double R[qSize][qSize] =  {

			{-1, -1, -1, -1, spc, -1,spc},
			{-1, -1, -1, spc, -1, 1,-1},
			{-1, -1, -1, spc, -1, -1,1},
			{-1, spc, -1, -1, spc, -1,-1},
			{-1, -1, -1, spc, -1, 1,-1},
			{-1, spc, -1, -1, spc, 1,spc},
			{-1, spc, -1, -1, spc, 1,-1},
			};


int training=1,recall=0,discover=0;

const int goal_state = 5;			
const int goal_act = 5;
const double visited_input = 0.0;
const double reward_multp = 1;
const double visited_reward = 0.75*reward_multp;
int currentState;
int trained;

struct Qlearner{
double **Q;
double **Q_check;
double BellmanQ,*BellQ_;
double *cstate;
int **tlist;         //Records state,action to experience
double ***rlist;   //Records the reward history of game wiped after game over
double **tqact;  //Record all the q values for experience
double **tstate; //Record the cstate for experience
double **R_;
double visited; //Visited reward

int tcount;
int tcomplete;
			};
ofstream err,outpt,cumrew;

void episode(RLnet* net,Qlearner*  qprog,int initialState);
void chooseAnAction(RLnet* net,Qlearner* qprog);
int getRandomAction(Qlearner* qprog,int upperBound, int lowerBound);
void initialize(Qlearner* qprog);
double maximum(Qlearner* qprog,int state, bool returnIndexOnly);
double netmax(RLnet* net,Qlearner* qprog,int state, bool returnIndexOnly);
double netmax_(RLnet* net,Qlearner* qprog,int state, bool returnIndexOnly);
double reward(Qlearner* qprog,int action);

int main(){

	int newState;
	
err.open("errfile.dat");
outpt.open("out.dat");
cumrew.open("cumrew.dat");

Qlearner *qprog;
qprog = new Qlearner;

	
	initialize(qprog);
	
	
	//Init Net
	
	//srand(71179874);   // Seed the generator with system time. (Training Seed = RNG seed)
	srand(time(NULL));
	RLnet *net;
	net = new RLnet;
	
    	initWeights(net);
	
	//load_weights(net,"weights2.txt");
	thisseed=(double)(rand()%1000)/1000;	///				   (Recall seed = Activation seed)
	//thisseed=0.74;
	if(training==1){seed=0;randcy=0;}else{seed=2;randcy=1;}
	
	if(training==1){save_seed(net,"seeds.txt");}
	if(recall==1){load_seed(net,"seeds.txt",0);randcy=0;}  //Include Annealed learning**

int discovery=rand()%1000;
//discovery=848;//262;
cout<<"Discover seed="<<discovery<<"\n";

if(training==1||discover==1){srand(discovery);}

trained=0;

if(training){ //If training
	
qprog->tcomplete=0;
    //Perform learning trials starting at all initial states.
    for(int j = 0; j <= (iterations - 1); j++){
        for(int i = 0; i <= (qSize - 1); i++){
		         
		episode(net,qprog,initialStates[i]);

	
		} // i
		
		
	} // j
}	

//Print weights
cout<<"Ended Training\n";
//printWeights(net);
cout<<"This seed="<<thisseed<<"\n";
    if(training==1){save_weights(net,"weights2.txt");}
    if(recall==1||discover==1){load_weights(net,"weights2.txt");}

  //Print out Q matrix.
 if(0){   
    for(int i = 0; i < qSize ; i++){
        for(int j = 0; j < qSize; j++){
            cout  << qprog->Q[i][j];
			if(j < qSize - 1){
				cout << ",";
			}
		} // j
        cout << "\n";
	} // i
    cout << "\n";
 }
trained=1;
int count;
int oldState;
	//Perform tests, starting at all initial states.
	for(int i = 0; i <= (qSize - 1); i++){
        currentState = initialStates[i];
        newState = 0;
	oldState = 0;
	count=0;
		do {
            //outpt << "First state="<<currentState << ", "<<"\n";
	    newState = netmax_(net,qprog,currentState, true);
	    //cout << currentState << ", ";
	    oldState = currentState;
	    if(R[currentState][newState]!=-1){
            currentState = newState;}
	    count++;
        } while(R[oldState][currentState]!=1 && count<10);
        //cout << endl;
	} // i

cout<<"Q test output\n";
//Perform tests, starting at all initial states.
count=0;
	for(int i = 0; i <= (qSize - 1); i++){
        currentState = initialStates[i];
	cout << currentState << ",";
        newState = 0;
		do {
		count++;
            newState = maximum(qprog,currentState,true);
            
            currentState = newState;
        cout << currentState << ",";
	} while(currentState != goal_state&&count<100);
        cout << endl;
	} // i

cout<<"Q Neural test output\n";
//Perform tests, starting at all initial states.
count=0;
	for(int i = 0; i <= (qSize - 1); i++){
        currentState = initialStates[i];
	cout << currentState << ",";
        newState = 0;
		do {
		count++;
            newState = netmax_(net,qprog,currentState, true);
 
            currentState = newState;
	     cout << currentState << ",";
        } while(currentState != goal_state&&count<100);
        cout << endl;
	} // i
	
err.close();
outpt.close();
cumrew.close();

	return 0;

}

void episode(RLnet* net,Qlearner* qprog,int initialState){

    currentState = initialState;
//cout<<"Inititial State "<<initialState<<"\n";
    //Travel from state to state until goal state is reached.
	do {
        chooseAnAction(net,qprog);
	} while(currentState != goal_state);

    //When currentState = 5, run through the set once more to
    //for convergence.
    for(int i = 0; i <= (qSize - 1); i++){
        chooseAnAction(net,qprog);
	} // i
	
for(int i=0;i<qSize;i++){
qprog->cstate[i]=0;
}

}

void randomUpdate(RLnet* net,Qlearner* qprog){
//Additional training to NN from Random states and actions
int act,state,list[25][2];
int current=0;
int repeat=0;
for(int i=0;i<25;i++){
while(1){
repeat=0;
act = rand()%qSize;
state = rand()%qSize;
for(i=0;i<current;i++){
if(list[i][0]==act&&list[i][1]==state){repeat=1;}
}
if(repeat==0){break;}

}

list[current][0]=act;list[current][1]=state;
current++;

//Current Q values now randomly sampled train the NN
 for(int i=0;i<qSize;i++){qprog->cstate[i]=0;}
	qprog->cstate[state]=1;
 
 calcNetRL(net,qprog->cstate,act,qprog->BellQ_);
       err<<errThisPat<<"\n";
       //Adjust network weights.
            WeightChangesHO(net);
	    WeightChangesHH(net);
            WeightChangesIH(net);
				
			}


}

void chooseAnAction(RLnet* net,Qlearner* qprog){

	int possibleAction;
	//Cstate
for(int i=0;i<qSize;i++){if(qprog->cstate[i]==0.5){         //Where weve been
qprog->cstate[i]=visited_input;				
				}				}
	qprog->cstate[currentState] = 0.5;

	if(qprog->tcount<memsize-2){
	qprog->tcount+=1; }else{
	qprog->tcount=0;
	qprog->tcomplete=1;
	//cout<<"Tcomplete experience can now train\n";
	}
int netmoved=0;	 

    //Randomly choose a possible action connected to the current state.
 
if(show_move==1){   
cout<<"Net moved from";
   				}
				
if(qprog->tcomplete==1){

   //possibleAction = maximum(qprog,currentState,true);
    possibleAction = netmax_(net,qprog,currentState,true);
	if(rand()%100<20){										/* New feature - false moves are tolerated during training for negative rl */
	if(R[currentState][possibleAction] > -1){
   	netmoved=1;    
		}else{
	possibleAction = getRandomAction(qprog,qSize, 0);		
	netmoved=0;
	}	 }else{
	netmoved=1;}
	
	if(show_move==1){	
cout<<currentState<<" to "<<possibleAction<<"\n";
			   }
				}		
if(show_move==1){cout<<"Random moved from";}
	  
	  double this_rnd = (double)(rand()%1000)/1000;
	  
	  if(this_rnd>rnd_explor||qprog->tcomplete!=1){					      	
	  possibleAction = getRandomAction(qprog,qSize, 0);
	         
	    netmoved=0;
if(show_move==1){	    
cout<<currentState<<" to "<<possibleAction<<"\n";    
 			   }

							}			

if(show_move==1){

	for(int i=0;i<qSize;i++){
	cout<<"\n";
	for(int j=0;j<qSize;j++){
	if(i==currentState&&j==possibleAction){
	if(netmoved==1){
	cout<<"X";}else{cout<<"R";}
	
	}else{
	
	if(R[i][j]==-1){
	cout<<"B";
	}else{
	cout<<"0";}}
	
	
	
	}
	}									
	
	//cout << "\033[2J\033[1;1H";		
	system("clear");
	//sleep(0.1);
//cout<<"Error***\n";	
	}

	if(R[currentState][possibleAction] >= -1){

//Create Q Matrix using Bellman Equation
	       
       if(rand()%10>8){
       if(R[currentState][possibleAction] >= 0){
       qprog->Q[currentState][possibleAction] = qprog->Q[currentState][possibleAction] + 0.5*( reward(qprog,possibleAction) - qprog->Q[currentState][possibleAction]);
       								}else{
	qprog->Q[currentState][possibleAction] = qprog->Q[currentState][possibleAction] + 0.5*( -100 - qprog->Q[currentState][possibleAction]);															}
 				}

//Create Experience epoch 


	qprog->tlist[qprog->tcount][0]=currentState;
	qprog->tlist[qprog->tcount][1]=possibleAction;
if(netmoved==1){
	cum_reward+=qprog->rlist[qprog->tcount][currentState][possibleAction];
			}
//Load the map
	
	if(qprog->tcount>1){
	if(qprog->tlist[qprog->tcount-1][4]!=1){ //Remember last Rmap
qprog->rlist[qprog->tcount]=qprog->rlist[qprog->tcount-1];	
								}
					}else{
qprog->rlist[qprog->tcount]=qprog->R_;				
					}
/*Note Valid move could include the goal state,act */					
	if(R[currentState][possibleAction]==spc){ //If valid move Mark Rmap
qprog->rlist[qprog->tcount+1][currentState][possibleAction]=qprog->visited;
								}
	
	if(currentState==goal_state&&possibleAction==goal_act){
	cumrew<<cum_reward<<"\n";			//Save cumulitive reward
	qprog->tlist[qprog->tcount][4]=1;
	qprog->rlist[qprog->tcount]=qprog->R_;		//Reset history
	//cum_reward=0; //reset cumultive reward
	}else{
	qprog->tlist[qprog->tcount][4]=0;
	}

//View the map
/*
cout<<"******************\n";
for(int i=0;i<qSize;i++){
cout<<"\n";
for(int j=0;j<qSize;j++){
cout<<qprog->rlist[qprog->tcount][i][j]<<",";

}}*/

//train from previos net q values option completely unreliant on Q real vals

	//Cstate
for(int i=0;i<qSize;i++){if(qprog->cstate[i]==0.5){       //Where weve been
qprog->cstate[i]=visited_input;				
				}				}
				
	qprog->cstate[possibleAction]=0.5;   //Where we are
	qprog->tstate[qprog->tcount]=qprog->cstate;
        
	if(netmoved==0){
	for(int i=0;i<qSize;i++){
        qprog->tqact[qprog->tcount][i]=qprog->Q[currentState][i];				
				  }
				  }else{
	qprog->tqact[qprog->tcount]=net->outVal;			  
				  
				  }
netmoved=0;

/*Need to include cstate on tlist to provide complete state input to net*/

//Update net using random sample from tlist
//Belman equation used to calculate error for neural network

if(qprog->tcomplete==1){
int pos=rand()%(memsize-1);
//cout<<"Updating from experience\n";

int nextAct;
int thisState=qprog->tlist[pos][0];	//Sampled from Experience
int nextState=qprog->tlist[pos][1];    //Sampled from Experience

//Create BellQ_ (NN Target) from predicted values
netmax_(net,qprog,thisState,false);
for(int i=0;i<qSize;i++){
qprog->BellQ_[i]= net->outVal[i];
				}

//Calculate predicted nextAct from nextstate use to provide Q_sa
int reward_;
if(use_R==1){
reward_ = R[thisState][nextState];
}else{
reward_ = qprog->rlist[pos][thisState][nextState];
}

if(reward_  >= 0){

nextAct = netmax_(net,qprog,nextState,false);

//Insert Q_sa into Bellman Equation to create BellQ_ Target ****Add full Bellman Equation****

qprog->BellQ_[nextState] = qprog->BellQ_[nextState] + 0.5*(reward_ + gAmma*net->outVal[nextAct]-qprog->BellQ_[nextState]);
			
					  }else{
qprog->BellQ_[nextState] = qprog->BellQ_[nextState] + qprog->BellQ_[nextState] + 0.5*(-100-qprog->BellQ_[nextState]);
				  
					  }					  
					  
							
if(qprog->tlist[pos][4]==1){	//If game is over Reward is Max (I think this is the problem gets rewarded for cheating)
qprog->BellQ_[nextState] = R[thisState][nextState];
}


				
if(show_move==1){
cout<<"Randomly Sampled Epoch:\n";
cout<<"ThisState  NextState  Reward \n";
cout<<thisState<<"           "<<nextState<<"         "<<qprog->rlist[pos][thisState][nextState]<<"\n";				
			    }
				

//Update the net using BellQ_ target
calcNetRL(net,qprog->tstate[pos],0,qprog->BellQ_);
err<<errThisPat<<"\n";
if(show_move==1){
cout<<"**********************Error="<<errThisPat<<"\n";
	   		    }
	    WeightChangesHO(net);
	    WeightChangesHH(net);
            WeightChangesIH(net);
								}


		//cout<<"currentState="<<currentState<<"possibleAct="<<possibleAction<<"QValue="<<Q[currentState][possibleAction]<<"BellmanQ="<<BellmanQ/1000<<"NetOuput="<<outPred<<"\n\n";
	


        if(R[currentState][possibleAction]!=-1){
	
	currentState = possibleAction;
	
					}

	}
}

int getRandomAction(Qlearner* qprog,int upperBound, int lowerBound){

	int action;
	bool choiceIsValid = false;
	int range = (upperBound - lowerBound) + 1;
    
    //Randomly choose a possible action connected to the current state.
    do {
    
        //Get a random value between 0 and 6.
        action = rand()%qSize;
	//action = lowerBound + int(range * rand() / (RAND_MAX + 1.0));
		if(R[currentState][action] > -2){
            choiceIsValid = true;
	    
		}
    } while(choiceIsValid == false);
    return action;
}

void initialize(Qlearner* qprog){
	
	qprog->BellQ_ = new double[qSize];
	qprog->Q = new double*[qSize];
	qprog->Q_check = new double*[qSize];
	qprog->cstate = new double[qSize];
	qprog->tlist = new int*[memsize];
	qprog->rlist = new double**[memsize];
	qprog->tstate = new double*[memsize];
	qprog->tqact = new double*[memsize];
	qprog->R_ = new double*[qSize];
	for(int i=0;i<qSize;i++){
	qprog->R_[i]=new double[qSize];
	}
	for(int i=0;i<qSize;i++){
	for(int j=0;j<qSize;j++){
	qprog->R_[i][j]=R[i][j]*reward_multp;
				   }
	}
	
	for(int i=0;i<memsize;i++){
	qprog->tlist[i] = new int[qSize];
	qprog->rlist[i] = new double*[qSize];
	for(int j=0;j<qSize;j++){
	qprog->rlist[i][j] = new double[qSize];
	}
	qprog->Q[i] = new double[qSize];
	qprog->Q_check[i] = new double[qSize];
	qprog->tstate[i] = new double[qSize]; //States
	qprog->tqact[i] = new double[qSize];  //Actions
	}
	
	qprog->visited = visited_reward;
	//srand((unsigned)time(0));

    for(int i = 0; i < qSize ; i++){
        for(int j = 0; j < qSize; j++){
            qprog->Q[i][j] = 0;
	    qprog->Q_check[i][j] = 0;
		} // j
	} // i
	
	
}

double maximum(Qlearner* qprog,int state, bool returnIndexOnly){
// if returnIndexOnly = true, a Q matrix index is returned.
// if returnIndexOnly = false, a Q matrix element is returned.

	int winner;
	bool foundNewWinner;
	bool done = false;

    winner = 0;
    
	do {
        foundNewWinner = false;
        for(int i = 0; i <= (qSize - 1); i++){
			if((i < winner) || (i > winner)){     //Avoid self-comparison.
				if(qprog->Q[state][i] > qprog->Q[state][winner]){
                    winner = i;
                    foundNewWinner = true;
				}
			}
		} // i

		if(foundNewWinner == false){
            done = true;
		}

    } while(done = false);

	if(returnIndexOnly == true){
		return winner;
	}else{
		return qprog->Q[state][winner];
	}
}

double netmax(RLnet* net,Qlearner* qprog,int state,bool returnIndexOnly){
	int winner=-1;
	double winnet=-1;
	bool foundNewWinner;
	bool done = false;

    //winner = 0;
    for(int i=0;i<qSize;i++){qprog->cstate[i]=0;}
    qprog->cstate[state]=1;
    
	do {
        foundNewWinner = false;
        for(int i = 0; i <= (qSize - 1); i++){
	
	calcNetRL(net,qprog->cstate,i,qprog->BellQ_);
	outpt<<"Output net :\n"<<"Q value="<<outPred<<"Action="<<i<<"Bellman="<<qprog->Q[state][i]<<"\n";
			if((i < winner) || (i > winner)){     //Avoid self-comparison.
				if(outPred > winnet){
		    //cout<<"Q value="<<outPred<<" "<<"Current winner "<<winner<<"="<<winnet<<"\n";
                    winner = i;
		    winnet = outPred;  //best output from net
                    foundNewWinner = true;
		    outpt<<"Found a winner\n"<<"Q value="<<outPred<<" Action="<<winner<<"\n";
				}
			}
		} // i

		if(foundNewWinner == false){
            done = true;
		}

    } while(done = false);

//cout<<"End search\n";
outpt<<"**-----Final winner for state ----**"<<state<<" ="<<winner<<"\n";
if(returnIndexOnly == true){
		return winner;
	}else{
		return winnet;
	}
	
	
}


double netmax_(RLnet* net,Qlearner* qprog, int state,bool returnIndexOnly){
	int winner=0;
	double winnet=0;
	bool foundNewWinner;
	bool done = false;

    //winner = 0;
    for(int i=0;i<qSize;i++){if(qprog->cstate[i]==0.5){         //Where weve been
qprog->cstate[i]=visited_input;				
				}				}
qprog->cstate[state]=0.5;
     
	do {
	
        foundNewWinner = false;
	calcNetRL(net,qprog->cstate,1,qprog->BellQ_);

        for(int i = qSize-1; i  >=0; i--){

if(trained==1){
	outpt<<"Output net :\n"<<"Q value="<<net->outVal[i]<<"Action="<<i<<"Bellman="<<qprog->Q[state][i]<<"\n";
		}

			if((i < winner) || (i > winner)){     //Avoid self-comparison.
				if(net->outVal[i] > winnet){
                    winner = i;
		    winnet = net->outVal[i];  //best output from net
                    foundNewWinner = true;
			
if(trained==1){		 
		    outpt<<"Found a winner\n"<<"Q value="<<net->outVal[i]<<" Action="<<winner<<"\n";
		     }	

				}
			}
		} // i

		if(foundNewWinner == false){
            done = true;
		}

    } while(done = false);

//cout<<"End search\n";
if(trained==1){
outpt<<"**-----Final winner for state ----**"<<state<<" ="<<winner<<"\n";
		    }
		    
if(returnIndexOnly == true){
//		cout<<"Returned Winner="<<winner<<"\n";
		return winner;
	}else{
		return winnet;
	}
	
	
}

double reward(Qlearner* qprog,int action){
				
    return static_cast<double>(R[currentState][action] + (gAmma * maximum(qprog,action, false)));
}

