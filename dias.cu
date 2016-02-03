//advanced cuda system
//sped up algorithms


/*
	Code guide: first matrices are initialized. they are used to keep track of the particles, the probabilities to jump, the substrate, and the general electric potential.
	Input parameters are also taken in. Currently the code takes in one parameter. The rest of the parameters must be adjusted manually and the code must be recompiled. The general electric potential is calculated in cuda. This reduces a n^4 problem to a n^2 one. A site is picked at random at the CPU (part of the monte-carlo process) and the probabilities with the particles around it are calculated at the gpu. The probabilities are then returned to the CPU where the second part of the Monte-Carlo algorithm occurs. Here, the site which the subject particle will interact with is chosen randomly but with weights according to the probabilities. The jump is made, and the system starts over.  



*/

#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <unistd.h> /* for getpid() */
#include <time.h> /* for time() */
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cuda.h>
#include <sstream>
#include <string>

#define PI	3.1415926535897932384626433832795
#define TWOPI 	6.28318530717958647692528676655901

// construct REAL "type," depending on desired precision
// set the maximum number of threads

#ifdef DOUBLE
 #define REAL double
 #define MAXT 256
#else
 #define REAL float
 #define MAXT 512
#endif

using namespace std;

int currentCount = 0;
int countThese = 1;

typedef struct {
	REAL re;
	REAL im;
} COMPLEX;
//wrote own modulo algorithms since computer modulo (%) does negative modulo's incorrectly (-3%10 = -3 instead of 7)
 __device__ int G_mod(int a,int b)
{
        while (a < 0) {
                a = a + b;
        }
        while (a >= b) {
                a = a - b;
        }
return a;
}
// I need a version of my modulo for the gpu and for the CPU 
int C_mod(int a, int b) {
        while (a < 0) {
                a = a + b;
        }
        while (a >= b) {
                a = a - b;
        }
return a;
}

__global__ void matrixCopy(int intN, REAL * matrixIn,REAL *matrixOut){
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < intN*intN) {
                matrixOut[idx] = matrixIn[idx];

        }

}




//Here, the gpu's find the general electric potential at each lattice site. 
__global__ void findPotential(REAL *particles,REAL *potentials, double N,double L, REAL *boxR) { 
        int i,j,intx,inty,checkx,checky,distancex,distancey;
	int intN = (int) N;
        int checkRange = N/2; //(*2)
	        double changeToV = 3.6e-10; // Ke*Q/Kd 
  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

        double k,l,sum,distanceTerm;
//	double deltax,deltay;
	 if(idx<intN*intN) {
        	i = idx/intN;
                j = idx%intN;
		sum = 0;
                       for(l = 0 ; l < checkRange*2; l++) {
                                for(k = 0; k < checkRange*2; k++) {
                                        checkx = 		G_mod(i + k - checkRange,N);
                                        checky = G_mod(j + l - checkRange,N);

                                        if ((k != checkRange) || (l != checkRange)) {
//                                                deltax = (double) (k - checkRange);  
//						deltay = (double) (l - checkRange); 
//						distanceTerm = L*sqrt(deltax*deltax + deltay*deltay );
						distancex = (int) k;
						distancey = (int) l;
  						distanceTerm = boxR[i + intN*j + intN*intN*distancex + intN*intN*intN*distancey];				
	                                        intx = (int) checkx;
                                                inty = (int) checky;
						if ((intx != i) || (inty != j)) {
                                                	sum = sum +  particles[(intx) + intN*(inty)]/distanceTerm;
						}
                                        }
                                }
                        }
//		potentials[i + intN*j] = -sum*changeToV;
                potentials[i + intN*j] = sum*changeToV;

	}
}

__global__ void potOnParticles(REAL *particles,REAL *potentials,int intN, double L,REAL *boxR) {
        int i,j,intx,inty,checkx,checky,distancex,distancey;
        double N = (double) intN;
        int checkRange = N/2; //(*2)
  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	        double changeToV = 3.6e-10; // Ke*Q/Kd 
        double k,l,sum,distanceTerm;
//      double deltax,deltay;
         if(idx<intN*intN) {
           
		i = idx/intN;
                j = idx%intN;
                sum = 0;
		if (particles[i + intN * j] > 0 ) {
                       for(l = 0 ; l < checkRange*2; l++) {
                                for(k = 0; k < checkRange*2 ; k++) {
                                        checkx = G_mod(i - checkRange + k  ,N);
                                        checky = G_mod(j - checkRange + l ,N);

                                        if ((k != checkRange) || (l != checkRange)) {
                                                distancex = (int) k;
                                                distancey = (int) l;
                                                distanceTerm = boxR[i + intN*j + intN*intN*distancex + intN*intN*intN*distancey];
                                                intx = (int) checkx;
                                                inty = (int) checky;
                                                if ((intx != i) || (inty != j)) {
                                                        sum = sum +  particles[(intx) + intN*(inty)]/distanceTerm;
                                                }
                                        }
                                }
                        }
		}
                potentials[i + intN*j] = changeToV*sum*particles[i + intN*j];
//		potentials[i + intN*j] = particles[i + intN*j];
        }


}



//check for a CUDA error, use argument for identification
bool errorAsk(const char *s="n/a")
{
    cudaError_t err=cudaGetLastError();
    if(err==cudaSuccess)
        return false;
    printf("CUDA error [%s]: %s\n",s,cudaGetErrorString(err));
    return true;
};

//here the occupation states at the lattice sites are compared to find what kind of coulomb blockade is taking place
 __device__ double findBlockade(int p,int thisp,double Ec)
{
	
        if ((thisp == 1) && (p == -1 )) {
                return 0; //no blockade penalty
        }
        if ((thisp == -1) && (p == 1 )) {
                return 0;
        }
        if ((thisp == -1) && (p == 3)) {
                return 2*Ec;
        }
        if ((thisp == 3) && (p == -1 )) { //not sure about this one figured twice the electrons means twice the penalty.
                return 2*Ec;
        }
        if ((thisp == 1) && (p == 1 )) {
                return Ec;
        }
	if ((thisp == -1) && (p == -1 )) {
                return Ec;
	}  
        if ((thisp == 1) && (p == 3 )) {
                return 0;
        }
        if ((thisp == 3) && (p == 1 )) {
                return 0;
        }
	if ((thisp == 3) && (p == 3 )) { //no chance
                return 1000*Ec; 
        }

return 0; //in case something whacky happens
}


//The first half of the heart of this program. Here the probabilities are calculated based on the energy change of the system and on the localization of the electron.
__global__ void findProbabilities(int N,double xi,REAL *probabilities,REAL *particles,REAL *potentials,REAL *substrate,int x, int y, double eV,double Ec,double T,REAL *boxR,double alphaOne, double alphaTwo)
{
//	REAL number = 11;
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    int i,j,thisi,thisj,p,thisp,hyperIndex;
	double potConstant,currentPart,distancePart,blockadePart,potentialPart,substratePart;
//	double doublej, doublei,r;
	double changeToV = 3.6e-10; // Ke*Q/Kd	
//	potConstant = 1.17e-13;
//	potConstant = Ec;
	potConstant = 1;
	
	if(idx<N*N)
    {         
		i = idx/N;
                j = idx%N;
		i = i-N/2;
		j = j-N/2;

		thisi = G_mod(i + x,N);
		thisj = G_mod(j + y,N);
	
		hyperIndex = x + N*y + N*N*(idx/N) + N*N*N*(idx%N);
	
//		doublei = i;
//		doublej = j;
//		r = sqrt(doublei*doublei + doublej*doublej);	
//		distancePart = -2.000*boxR[idx];
		distancePart = -2*boxR[hyperIndex]/xi;		
//		distancePart = 0;
		p = particles[x + N*y];
		thisp = particles[thisi + N*thisj];
	
		if(particles[x + N*y] > particles[thisi + N*thisj]) {

			blockadePart = -1*findBlockade(p,thisp,Ec)/boxR[hyperIndex];
			potentialPart = -potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] - changeToV/boxR[hyperIndex]);
			substratePart = substrate[thisi+ N*thisj];
			currentPart = eV*i;

//			currentPart = 0;			
//			blockadePart = 0;
//			potentialPart= 0;
//			substratePart= 0;

			
		}

		if (particles[x + N*y] < particles[thisi + N*thisj]) {

			blockadePart = -1*findBlockade(p,thisp,Ec)/boxR[hyperIndex]; 
			potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] + changeToV/boxR[hyperIndex]);
			substratePart = -substrate[thisi + N*thisj];
			currentPart = -eV*i;

//			currentPart = 0;
//			substratePart = 0;
//			potentialPart = 0;
//			blockadePart = 0;

		}

		if ( particles[x + N*y] == particles[thisi + N*thisj] ){
/*
			if (p > 0 ) {
				currentPart  = eV*i;
			}

			else {
                                currentPart  = -eV*i;
                        }
*/

			substratePart = -substrate[thisi+ N*thisj];
			blockadePart = -1*findBlockade(p,thisp,Ec)/boxR[hyperIndex];
			potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y]);
	
			currentPart = 0;
//			substratePart = 0;
//			potentialPart = 0;
//			blockadePart = 0;

		}



	probabilities[idx] = exp(distancePart+alphaTwo*(blockadePart+potentialPart+substratePart+currentPart)/T);

//	probabilities[idx] =exp(distancePart + alphaTwo*(potentialPart + currentPart)/T);
//probabilities[idx] = distancePart+alphaTwo*(blockadePart+potentialPart+substratePart+currentPart)/T;


	if (probabilities[idx] > 1) {
		probabilities[idx] = 1;
	}

	if ((thisi==x && thisj==y )  ){
		probabilities[idx] = 1; //force probability of jumping to self to 1 (avoids 0/0 problems)
//	probabilities[idx] = 0; //1 seems arbitrary
	}
	}

};
__device__ void fillRecord(REAL *jumpRecord,REAL fillVal,int N) {
int found = 0;
int n = 0;
	while ((found == 0) && (n < N)) {
		if( jumpRecord[n] == 999) {
			found = 1;
			jumpRecord[n] = fillVal; 
			
		}
		n++;
	}
}



//figures out which way the electron jump will occur and also calculates the current or jump distance (since particle movement is also done here).
 __device__ void interaction(int grabJ,int x,int y,int newx,int newy,int N,REAL *particles,REAL *jumpRecord) {
	double current,totalCurrent = 0;
	int whichWay = 0;
	REAL dx2,dy2,fillVal;
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
//	if(idx < 1) {
        if ((particles[x + y*N] == -1 ) && ( particles[newx + newy*N] == -1 ) ) { //do i really need this?
                current = 0;
        }

        else if (particles[x + y*N] > particles[newx + newy*N] ) {

                if( (x  > N/2 && x < 3*N/4) && (newx <= N/2)  ) {
                        current = 1;
                }

                if( (x < N/2 )&& (newx >= N/2 && newx < 3*N/4  )) {
                        current = -1;
                }
		whichWay = 1;
        }

        else if (particles[x + y*N] < particles[newx + newy*N]) {

                if( (x  > N/2 && x < 3*N/4) && (newx <= N/2)  ) {
                        current = -1;
                }

                if( (x < N/2) && (newx >= N/2 && newx < 3*N/4)  ) {
                        current = 1;
                }
		whichWay = -1;

        }


	else if ((particles[x + y*N] == 1) && (particles[newx + newy*N] == 1)) {
                if( (x  > N/2 && x < 3*N/4) && (newx <= N/2)  ) {
                        current = 1;
                }

                if( (x < N/2 )&& (newx >= N/2 && newx < 3*N/4  )) {
                        current = -1;
                }


		whichWay = 1;
		

	}


	if (whichWay > 0){
		particles[x + y*N] = particles[x + y*N] - 2;
                particles[newx + newy*N] = particles[newx + newy*N] + 2;
		
	}

	else if (whichWay < 0) {
	        particles[x + y*N] = particles[x + y*N] + 2;
                particles[newx + newy*N] = particles[newx + newy*N] - 2;
	}

totalCurrent = totalCurrent + current;

		dx2 = (REAL) ((x - newx) * (x - newx));
                dy2 = (REAL) ((y - newy) * (y - newy));
		fillVal = sqrtf(dx2 + dy2);
	if(grabJ == 1) {
		fillVal = current;
	}

	if((fillVal < 50)  && (particles[x + y*N] != particles[newx + newy*N])) {
		fillRecord(jumpRecord,fillVal,10000);
	}
//}
}

//this section does the various outputs such as particle positions or general electric potential
//this one outputs how far electrons jumped
void showJump(int N,int x,int y,int newx,int newy,REAL* hereP) {
	double r,deltax,deltay;
	deltax = (x-newx);
	deltay = (y-newy);

	r = sqrt(deltax*deltax + deltay*deltay);
	
//	cout<<x<<" "<<y<<" "<<newx<<" "<<newy<<endl;
	
	cout<<r<<endl;
//	cout<<hereP[x + N*y]<<" "<<hereP[newx + N*newy]<<endl;


}

//this is for showing the electron positions
void showMove(REAL* hereP,int N) {
        int i,j;
        for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        cout<<hereP[i + N*j]<<" ";
                }
        cout<<endl;
        }


}

//sums the potentials (during relaxation this should generally decrease)
double sumEnergy(REAL* hereField,int N) {
	int i,j;
	double sum;
	sum = 0;
	for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        sum = sum +  hereField[i + N*j];
                }
        }
	
	return sum;
}

//to double check i had no particles leaking
void countParticles(REAL* hereP, int N) {
        int i,j;
        double sum;
        sum = 0;
        for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        sum = sum +  hereP[i + N*j];
                }
        }

        cout<<sum<<endl;



}

__global__ void particleJump(int grabJ, int x, int y,double randomNum,int N,REAL *reducedProb,REAL *particles,REAL *jumpRecord) {
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        double pickedValue = randomNum*reducedProb[N*N -1];
	int newx,newy,lastx,lasty;
	if ((idx > 0) && (idx < N*N)) {
		if ((reducedProb[idx - 1] < pickedValue) && (reducedProb[idx] > pickedValue)) {
			lastx = idx/N;
        		lasty = idx%N;
		        newx = G_mod(x - N/2 +  lastx,N);
		        newy = G_mod(y - N/2 +  lasty,N);
			interaction(grabJ,x,y,newx,newy,N,particles,jumpRecord);			
//fillRecord(jumpRecord,pickedValue, 10000);

		}
	}
	if (idx == 0) {
		if (pickedValue < reducedProb[0]) {
	                lastx = idx/N;
                        lasty = idx%N;
		        newx = G_mod(x - N/2 +  lastx,N);
		        newy = G_mod(y - N/2 +  lasty,N);
			interaction(grabJ,x,y,newx,newy,N,particles,jumpRecord);
		}	
	}
//	fillRecord(jumpRecord,pickedValue, 10000);
	

}

void printBoxCPU(REAL *c_array,int size, char * boxName) {
        int k,l;
        FILE    *fp1;
//        char    str1[256];
//        sprintf(str1, "box.txt");
        fp1 = fopen(boxName, "w");
        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {

                        fprintf(fp1, "%lf ",1e9*c_array[k + l*size]);
                }
        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
}


void printBoxGPU(REAL *g_array,int size, char * boxName) {
        REAL *c_array;
        c_array =  new REAL[size*size];
        int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
//        char    str1[256];
//        sprintf(str1, "box.txt");
        fp1 = fopen(boxName, "w");
        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {

                        fprintf(fp1, "%lf ",c_array[k + l*size]);
                }
        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
	delete[] c_array;
}

void printLineGPU(REAL *g_array,int size,char * lineName) {
        REAL *c_array;
        c_array =  new REAL[size];
        int k;
        cudaMemcpy(c_array,g_array,size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
 //       char    str1[256];
//        sprintf(str1, "line.txt");
        fp1 = fopen(lineName, "w");
        for (k = 0; k < size ; k++){

                        fprintf(fp1, "%lf ",c_array[k]);
			fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
	delete[] c_array;
}

REAL *loadMatrix(REAL *hereMatrix,char* fileName) {
//      infile.open (fileName, ifstream::in);
//      REAL * buffer;
//        ifstream read(fileName);
        ifstream infile(fileName);
        string line;
        int counter = 0;
        double d;
        while (getline(infile, line)) {
                istringstream iss(line);
                if (iss >> d) {
//                      cout<<d<<endl;
                        hereMatrix[counter] = d;

                        counter++;
                }


        }


        return hereMatrix;
}




//second part of the heart of this code. Here the probabilities are summed and a number is picked from 0 to that number. The code then sums through the probabilities untill it reaches that number. In this way, probabilities which are higher will have a larger chance of getting picked. 
void particleScout(REAL *reducedProb,REAL* particles,REAL* probabilities,REAL* jumpRecord,int x,int y,int N,double randomNum,int blocks, int threads,int grabJ) {
        
	thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(probabilities);
        thrust::device_ptr<REAL> g_return = thrust::device_pointer_cast(reducedProb);

        thrust::inclusive_scan(g_go, g_go + N*N, g_return); // in-place scan 
		
	particleJump<<<blocks,threads>>>(grabJ, x, y,randomNum,N,reducedProb,particles,jumpRecord);
        errorAsk("particleJump");
}


//the particles are picked here. This is also where the system is run from. (find potential, find probabilities, and move particle are done here)
void findJump(REAL* hereP,REAL* hereProb,REAL* herePot,REAL *particles,REAL *probabilities,REAL *potentials,REAL *substrate,REAL *reducedProb,REAL *jumpRecord,int N,double xi,int threads,int blocks,double eV,double Ec,double L,double T,REAL *boxR, double alphaOne, double alphaTwo,int grabJ) {
	int x,y;	
	double randomNum;
	 x = floor(drand48()*N);
         y = floor(drand48()*N);
//      Displays:

//        showMove(hereP,N);
//      showMove(hereProb,N);
//      showMove(herePot,N);
//      sumEnergy(herePot,N);
//      countParticles(hereP,N);
//      line 300 for the jump distance display

	findPotential<<<blocks,threads>>>(particles,potentials, N,L,boxR);
	errorAsk("find Potential");
	findProbabilities<<<blocks,threads>>>(N,xi,probabilities,particles,potentials,substrate,x,y,eV,Ec,T,boxR,alphaOne,alphaTwo);
	errorAsk("find probabilities"); //check for error

//        printGPU(probabilities,N);	
	
	randomNum = drand48();
	particleScout(reducedProb, particles,probabilities,jumpRecord, x, y, N,randomNum, blocks, threads,grabJ);
}

__global__ void G_stackE(REAL *particles,REAL *stacked,int intN) {
        int i,j;
        double blockade = 1.97e-5;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        i = idx/intN;
        j = idx%intN;

	if(idx < intN*intN) {
		if (particles[i + j*intN] > 1) {
			stacked[idx] = blockade;

		}
	}
}



__global__ void G_subE(REAL *substrate,REAL *particles,REAL *combined,int intN) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if(idx < intN*intN) {
	        combined[idx] = substrate[idx]*particles[idx];
	}


}
__global__ void fillSum(int index,int intN,int addSub,REAL *sumArray,REAL numToInsert) {
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
//	if(idx < 1) {
	REAL	dSign = (REAL) addSub;
	sumArray[index] = dSign*numToInsert;
//	}
}

__global__ void particleSwitch(int i,int j,int intN,REAL *particles) {

	if (particles[i + j*intN] == 0) {
		particles[i + j*intN]= 1;
	}
	else {
		particles[i + j*intN]= 0;
	}
}

__global__ void dosPut(int i,int j,int intN,REAL *Ematrix,REAL sum) {
	Ematrix[i + j*intN] =	sum;
}


 void G_dos(REAL * sumArray,REAL *extraArray,REAL *boxR,REAL *particles,REAL *substrate,REAL *Ematrix,REAL *potentials,int slices,double N,double L,int threads,int blocks) {
	int i,j,intN;//not sure about Sums
	intN = (int) N;
        thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(potentials);
	thrust::device_ptr<REAL> sumArrayPtr = thrust::device_pointer_cast(sumArray);
	thrust::device_ptr<REAL> extraArrayPtr = thrust::device_pointer_cast(extraArray);
	REAL result;
//printBoxGPU(particles,N,"pBox.txt");

        for (j = 0; j < intN; j++) {
                for (i = 0; i < intN; i++) {
			findPotential<<<blocks,threads>>>(particles,potentials, N,L,boxR);

//			potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
			result = thrust::reduce(g_go, g_go + intN*intN);        
			fillSum<<<blocks,threads>>>(0,intN,-1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(0,intN,1,sumArray,result);

			G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(1,intN,-1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(1,intN,1,sumArray,result);

			G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
                        fillSum<<<blocks,threads>>>(2,intN,-1,sumArray,result);
//			fillSum<<<blocks,threads>>>(2,intN,1,sumArray,result);
			
			particleSwitch<<<blocks,threads>>>(i,j,intN,particles);

			findPotential<<<blocks,threads>>>(particles,potentials, N,L,boxR);
//                        potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
			result = thrust::reduce(g_go, g_go + intN*intN);
                        fillSum<<<blocks,threads>>>(3,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(3,intN,-1,sumArray,result);

                        G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
                        result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(4,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(4,intN,-1,sumArray,result);

                        G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
                	result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(5,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(5,intN,-1,sumArray,result);

                        particleSwitch<<<blocks,threads>>>(i,j,intN,particles);
			
			result = thrust::reduce(sumArrayPtr, sumArrayPtr + 6); 
//			result = 0;	
			dosPut<<<blocks,threads>>>(i,j,intN,Ematrix,result);
		}
	}


}


//random substrate is created here
REAL *createSub(REAL *hereS,double muVar,int N) {
        int i,j;

        for(j = 0; j < N; j++ ) {
                for(i = 0; i < N; i++) {

                        hereS[i + N*j] = drand48()*muVar*2 - muVar;
        //              if(i > nx/2) hereS[i + ny*j] = 50000000;        
                }

        }





        return hereS;
}



// creates the variation in x & y matrices
REAL * createDiff(REAL * hereDiff, double var, int N) {
	int i,j;

	for(j = 0; j < N; j++) {
		for(i = 0; i < N; i++) {
			hereDiff[i + N*j] = drand48()* var*2 - var;
		}
	}

	return hereDiff;
}
REAL *C_zeros(double N, REAL *A) {
	int idx;
	for (idx = 0; idx < N; idx++) {
		A[idx] = 0;
	}
return A;


}
//creates and fills matrices
REAL *C_random(double N,double nparticles,REAL *A) {
        int idx,idy,count,index;
        int randx,randy;
        count = 0;


        for (idx = 0; idx < N; idx++) {
                for( idy = 0; idy < N; idy++) {
                        index = int(idy + idx*N);
                        A[index] = -1;


                }
        }

	        while(count < nparticles) {
                randx = drand48()*N;
                randy = drand48()*N;
                randx = floor(randx);
                randy = floor(randy);

                index = int(randx*N + randy);
                 if (A[index] < 2) {
                        A[index] = A[index] + 1;
                        count++;
                }
        }



return A;

}



//creates and fills matrices when filled percent > 100%
REAL *C_more(double N,double nparticles,REAL *A) {
        int idx,idy,count,index;
        int randx,randy;
        count = 0;


        for (idx = 0; idx < N; idx++) {
                for( idy = 0; idy < N; idy++) {
                        index = int(idy + idx*N);
                        A[index] = 1;


                }
        }

	


                while(count < (nparticles-N*N)) {
                randx = drand48()*N;
                randy = drand48()*N;
                randx = floor(randx);
                randy = floor(randy);

                index = int(randx*N + randy);
                 if (A[index] < 2) {
                        A[index] = A[index] + 1;
                        count++;
                }
        }



return A;

}


//creates the "distance hyper-matrix" 1/r
REAL *createR(REAL *A,REAL *diffX, REAL *diffY,double N,double L,double xi) {
	double r,doublel,doublek,deltaX,deltaY;
	double diffXThere,diffYThere,diffXHere,diffYHere;
	int i,j,k,l,intN,idx,kObs,lObs,kNew,lNew;
	
	intN = N;
	for (idx = 0; idx < N*N*N*N; idx++) {

                i = idx%(intN);
                j = (idx%(intN*intN) - idx%(intN))/intN; 
                k = (idx%(intN*intN*intN) - idx%(intN*intN))/(intN*intN) ; 
                l = (idx%(intN*intN*intN*intN) - idx%(intN*intN*intN))/(intN*intN*intN) ;

/*
                k = idx%(intN);
                l = (idx%(intN*intN) - idx%(intN))/intN;
                i = (idx%(intN*intN*intN) - idx%(intN*intN))/(intN*intN) ;
                j = (idx%(intN*intN*intN*intN) - idx%(intN*intN*intN))/(intN*intN*intN) ;
*/

                doublek = (double) k;
                doublel = (double) l;

		kNew = i + k - N/2;
		lNew = j + l - N/2;

                kObs = C_mod(kNew,N);
                lObs = C_mod(lNew,N);
		
                diffXHere = diffX[i + intN*j];
                diffXThere = diffX[kObs + intN*lObs];

                if((kNew < 0) || (kNew > N)) {
      //                  diffXHere = -diffX[i + intN*j];
                        diffXThere = -diffX[kObs + intN*lObs];
                }
                diffYHere = diffY[i + intN*j];
                diffYThere = diffY[kObs + intN*lObs];

                if((lNew < 0) || (lNew > N)) {
        //                diffYHere = -diffY[i + intN*j];
                        diffYThere = -diffY[kObs + intN*lObs];
                }

		deltaX = diffXHere - (diffXThere + L*(doublek - N/2));
		deltaY = diffYHere - (diffYThere + L*(doublel - N/2));
		
		r = sqrt(deltaX*deltaX + deltaY*deltaY);

		A[idx] = r;


	}
/*
	for (i = 0; i < N; i++) {
		for(j = 0; j < N ; j++) {
			 cout<<A[1 + intN*1 + intN*intN*i + intN*intN*intN*j]<<" ";
		}
		cout<<endl;
	}
*/

//cout<<A[26 + intN*16 + intN*intN*12 + intN*intN*intN*8]<<endl;

return A;
	
}

//clumps all of the original electrons ( to show relaxation)
REAL *C_clump(double N,double nparticles,REAL *A) {
        int idx;

	for (idx = 0;idx < N*N; idx++) {
		A[idx] = -1;
	}

	for (idx = 0; idx < nparticles; idx++) {
		A[idx] = 1;
	}
	
	
return A;
}

//electrons evenly spaced out (to try to calculate average jump distances with a general electric potential)
REAL *C_spread(double N,double nparticles,REAL *A) {
        int idx,i,j,intN;
	intN = (int) N;
	
	
        for (idx = 0;idx < N*N; idx++) {
                A[idx] = -1;
        }

        for (idx = 0; idx < N*N; idx++) {
        	i = idx/intN;
        	j = idx%intN;
		
		if((i + j)%2) {        
			A[idx] = 1;
        	}
	}


return A;
}

__global__ void particleSwap(int i,int j,int k,int l,int intN,REAL *particles) {
	int temp;
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

//	if (idx < 1) {
	temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
	particles[k + l*intN] = temp;
//	}
}

__device__ void g_particleSwap(int i,int j,int k,int l,int intN,REAL *particles){
	
        int temp;
 //        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
//	if (idx < 1) {
	temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
        particles[k + l*intN] = temp;	
//	}
}


__device__ int changeCoordinates(int intN, int x1, int x2) {

	int modulox,newCoord;
        if (x2 < intN/2) {
		modulox = x2;
	}
	else {
		modulox = x2 - intN;
	}

	newCoord = intN/2 + modulox ;

return newCoord;

}



/*
__global__ void fastSwap(int i1, int j1, int i2, int j2,int intN,REAL *particles,REAL *boxR,REAL *Ematrix, REAL *tempDos,REAL *potentials){

	int x,y;
	int xPre,yPre;
	double changeToV = 3.6e-10; // Ke*Q/Kd 
	double distance1,distance2;
	double crater,mound,newPot;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(idx < intN*intN) {
		
		if (particles[i1 + intN*j1] != particles[i2 + intN*j2]) {


                	xPre = idx/intN;
	                yPre = idx%intN;
                        x = (int) G_mod(xPre + ( intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[x + intN*y + intN*intN*i1 + intN*intN*intN*j1];//might be the other way
			if (distance1 > 0) {
				if (particles[i1 + intN*j1] == 1) {
					
					crater = -changeToV/distance1;
				}
				else {
					crater = changeToV/distance1;
				}
			}
			else {
				crater = 0;	
			}

                        x = (int) G_mod(xPre + ( intN/2 - i2),intN);
                        y = (int) G_mod(yPre + (intN/2 - j2),intN);

                        distance2 = boxR[x + intN*y + intN*intN*i2 + intN*intN*intN*j2];//might be the other way 
			if (distance2 > 0) {
                                if (particles[i2 + intN*j2] == 1) {
                                        mound = -changeToV/distance2;
                                }
                                else {
                                        mound = changeToV/distance2;
//                                	mound = 999;
				}
                        }

			else {
				mound = 0;
			}
//			newPot = 
//			tempDos[idx] = particles[idx]*(potentials[idx] + crater + mound);

			tempDos[idx] = crater;
		}
		else {
			tempDos[idx] = Ematrix[idx];
		}
	
		if (particles[i1 + intN*j1] == 0){
//			tempDos[idx] = -tempDos[idx];
		} 	
	}
}
*/

__global__ void slowSwap(int i1,int j1,int i2, int j2,int intN, REAL* tempPar,REAL *tempPot,REAL *tempDos,REAL *particles,REAL *potentials,REAL *Ematrix,REAL *boxR,REAL *substrate,REAL *watcher) {
	double distance1, distance2;
	int xPre,yPre,x,y;
        double changeToV = 3.6e-10; // Ke*Q/Kd 

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		tempPar[idx] = particles[idx];
                tempPot[idx] = potentials[idx];
                tempDos[idx] = Ematrix[idx];
		if (particles[i1 + intN*j1] != particles[i2 + intN*j2]) {
			if(particles[i1 + intN*j1] == 1) {
				tempPar[i1 + intN*j1] = -1;

//	                        xPre = idx/intN;
//	                        yPre = idx%intN;
        	                
				yPre = idx/intN;
                                xPre = idx%intN;				

				x = (int) G_mod(xPre + ( intN/2 - i1),intN);
                	        y = (int) G_mod(yPre + (intN/2 - j1),intN);
                                distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];

	                        if (distance1 > 0) {
	                                tempPot[idx] = tempPot[idx] + changeToV/distance1;
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
        	                }
				else {
//					tempPot[idx] = tempPot[idx] - substrate[idx]*particles[idx];	
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];

				}
//	probe = distance1;
				tempPar[i2 + intN*j2] = 1;
				
				distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];
				if (distance2 > 0) {
                                        tempPot[idx] = tempPot[idx] - changeToV/distance2;
					tempDos[idx] = tempPot[idx]*tempPar[idx]+ substrate[idx]*tempPar[idx];
                                }
				else {
//                                        tempPot[idx] = tempPot[idx] + substrate[idx]*particles[idx];
					
					tempDos[idx] = tempPot[idx]*tempPar[idx] - substrate[idx]*tempPar[idx];

                                }		

						
			}
			else {
                                tempPar[i1 + intN*j1] = 1;

//                                xPre = idx/intN;
//                                yPre = idx%intN;

				yPre = idx/intN;
                                xPre = idx%intN;

                                x = (int) G_mod(xPre + ( intN/2 - i1),intN);
                                y = (int) G_mod(yPre + (intN/2 - j1),intN);
                                distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
//				distance1 = boxR[x + intN*y + intN*intN*i1 + intN*intN*intN*j1];
//				watcher[idx] = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                                if (distance1 > 0) {
                                        tempPot[idx] = tempPot[idx] - changeToV/distance1;
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
                                }
				else {
//                                        tempPot[idx] = tempPot[idx] + substrate[idx]*particles[idx];
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
//					watcher[idx] = tempPot[idx]*tempPar[idx] ;
                                }
//watcher[idx] = changeToV/distance1;

                                tempPar[i2 + intN*j2] = -1;
				x = (int) G_mod(xPre + ( intN/2 - i2),intN);
                                y = (int) G_mod(yPre + (intN/2 - j2),intN);

                                distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];
                                if (distance2 > 0) {
                                        tempPot[idx] = tempPot[idx] + changeToV/distance2;
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
                                }
                                else {
//                                        tempPot[idx] = tempPot[idx] - substrate[idx]*particles[idx];
					tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
//					watcher[idx] = tempPot[idx]*tempPar[idx] ;
				}
			watcher[idx] = substrate[idx]*tempPar[idx];
			}
//		tempDos[idx] = probe;
		}

	        else {
                        tempDos[idx] = Ematrix[idx];
                }
	}
}	

__global__ void subAdd(int intN, REAL *particles,REAL *potentials,REAL *substrate){

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		potentials[idx] = potentials[idx] + substrate[idx]*particles[idx];
	}

}

__global__ void potAdd(int i1, int j1, int intN,REAL *particles,REAL *boxR,REAL *potentials){
        int x,y;
        int xPre,yPre;
        double changeToV = 3.6e-10; // Ke*Q/Kd
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
                if (particles[i1 + intN*j1] == 1) {

		        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
 		        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                       		potentials[idx] = potentials[idx] -.5*changeToV/distance1; // .5 since Im coming from neutral
                        }
			
			
		}
		
		if (particles[i1 + intN*j1] == -1) {

                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                                potentials[idx] = potentials[idx] + .5*changeToV/distance1;
                        }


                }

		

	}
}


__global__ void potSub(int i1, int j1, int intN,REAL *particles,REAL *boxR,REAL *potentials){
        int x,y;
        int xPre,yPre;
        double changeToV = 3.6e-10; // Ke*Q/Kd
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
                if (particles[i1 + intN*j1] == -1) {

                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                                potentials[idx] = potentials[idx] + changeToV/distance1;
                        }


                }
                if (particles[i1 + intN*j1] == 1) {


 	                yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                                potentials[idx] = potentials[idx] - changeToV/distance1;
                        }


                }
	



        }
}

__global__ void potChange(int i1, int j1, int intN,REAL *particles,REAL *boxR,REAL *potentials,REAL* Ematrix){
        int x,y;
        int xPre,yPre;
        double changeToV = 3.6e-10; // Ke*Q/Kd
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {


                if (particles[i1 + intN*j1] == 1) {
			particles[i1 + intN*j1] = 0;
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                                potentials[idx] = potentials[idx] + changeToV/distance1;
//				potentials[idx] = 999;
                        }


                if(particles[idx] == 0) {
                        Ematrix[idx] = potentials[idx];
//                	if (distance1 == 0) {
//				Ematrix[idx] = -potentials[idx];
//			}
		}
                else {
                        Ematrix[idx] = -potentials[idx];
//			if (distance1 == 0) {
//				Ematrix[idx] = potentials[idx];
//                        }
                }


                }
				
		else {
			particles[i1 + intN*j1] = 1;
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = (int) G_mod(xPre + (intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                        if (distance1 > 0) {
                                potentials[idx] = potentials[idx] - changeToV/distance1;
                      }

                if(particles[idx] == 0) {
                        Ematrix[idx] = potentials[idx];
//			if (distance1 == 0) {
//                                Ematrix[idx] = -potentials[idx];
//                        }
                }
                else {
                        Ematrix[idx] = -potentials[idx];
//			if (distance1 == 0) {
//                                Ematrix[idx] = potentials[idx];
//                        }
                }



                }



	}

}


__global__ void findE(int intN, REAL *particles,REAL *Ematrix,REAL *potentials,REAL *substrate) {
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
			Ematrix[idx] = particles[idx]*potentials[idx] + particles[idx]*substrate[idx];
//			Ematrix[idx] = particles[idx]*potentials[idx];
	}


}


__global__ void dosChange(int intN, REAL *particles,REAL *Ematrix,REAL *potentials) {
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		if(particles[idx] == 0) {                        
			Ematrix[idx] = potentials[idx];
		}
		else {
			Ematrix[idx] = -potentials[idx];
		}
        }


}

__global__ void particleDrop(int intN,int i ,int j,int newParticle,REAL *particles){
	particles[i + intN*j] = newParticle;
}

__global__ void potSwap(int i1, int j1, int i2, int j2,int intN,REAL *particles,REAL *boxR,REAL *potentials){ 
        int x,y;
	int xPre,yPre;
        double changeToV = 3.6e-10; // Ke*Q/Kd
//	double changeToV = 3.6e-1; // Ke*Q/Kd 
	REAL distance1,distance2;
//	REAL before,after;

	 int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	 if(idx<intN*intN) {
//		before = Ematrix[idx];
	        if (particles[i1 + intN*j1] != particles[i2 + intN*j2]) {

			yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
			

                        x = (int) G_mod(xPre + ( intN/2 - i1),intN);
                        y = (int) G_mod(yPre + (intN/2 - j1),intN);

			distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
			if (distance1 > 0) {
                                if (particles[i1 + intN*j1] == 1) {
					potentials[idx] = potentials[idx] + changeToV/distance1;
				}
                                else {
					potentials[idx] = potentials[idx] - changeToV/distance1;
				}
                       }




			x = (int) G_mod(xPre + ( intN/2 - i2),intN);
                        y = (int) G_mod(yPre + (intN/2 - j2),intN);
			
                        distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];//might be the other way

			if (distance2 > 0) {
                                if (particles[i2 + intN*j2] == 1) {
					potentials[idx] = potentials[idx] + changeToV/distance2;
				 }
                                else {
					potentials[idx] = potentials[idx] - changeToV/distance2;
		                }
			}
		
		}

        }
}

void C_particleForce(REAL* substrate,REAL* potentials, REAL * boxR, REAL *Ematrix,REAL *particles,int intN, int i1, int j1,int i2,int j2,int threads, int blocks) {
			
			potSwap<<<blocks,threads>>>(i1, j1, i2, j2,intN,particles,boxR,potentials);
			particleSwap<<<blocks,threads>>>(i1, j1, i2,j2,intN,particles);
			findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);


}

void C_particlePick(REAL *substrate,REAL* potentials,REAL * results, REAL * boxR, REAL *Ematrix,REAL *particles,int intN, int i, int j,int threads, int blocks) {
/*
//printBoxGPU(particles,intN,"pBox.txt");
cout<<i<<" "<<j<<" "<<endl;
for (int q = 0; q < 5; q++) {
      cout<<results[q]<<endl;
}
*/
        if ((results[0] < results[1] ) ||(results[0] < results[2] ) ||(results[0] < results[3] ) ||(results[0] < results[4] ) ) {
        
	int iPrev,jPrev,iPost,jPost;
        iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
        jPost = C_mod(j + 1,intN);

                if ((results[1] > results[2] ) &&(results[1] > results[3] ) &&(results[1] > results[4] )  ) {
			
			potSwap<<<blocks,threads>>>( i, j, iPrev,j,intN,particles,boxR,potentials);	
			particleSwap<<<blocks,threads>>>(i, j, iPrev,j,intN,particles);
			findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

//cout<<iPrev<<" "<<j<<endl;
		}

                else if ((results[2] > results[3] ) &&(results[2] > results[4] )) {
                	potSwap<<<blocks,threads>>>( i, j, i,jPrev,intN,particles,boxR,potentials);
			particleSwap<<<blocks,threads>>>(i, j, i,jPrev,intN,particles);
			findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

//cout<<i<<" "<<jPrev<<endl;
		}

                else if (results[3] > results[4]) {
   	            	potSwap<<<blocks,threads>>>( i, j, iPost,j,intN,particles,boxR,potentials);
			particleSwap<<<blocks,threads>>>(i, j, iPost,j,intN,particles);
			findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

//cout<<iPost<<" "<<j<<endl;
		}

                else {
			potSwap<<<blocks,threads>>>( i, j, i,jPost,intN,particles,boxR,potentials);
               		particleSwap<<<blocks,threads>>>(i, j, i,jPost,intN,particles);
			findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

//cout<<i<<" "<<jPost<<endl;
		 }
        }
}


void fastTest(REAL *watcher,REAL* substrate,REAL* potentials,REAL *Ematrix, REAL *tempDos,REAL *tempPar,REAL *tempPot, REAL *particles, REAL *boxR,int i, int j, int intN,int threads, int blocks) {
        
	int iPrev,jPrev,iPost,jPost;
        iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
        jPost = C_mod(j + 1,intN);
	REAL result;
	thrust::device_ptr<REAL> p_tempDos = thrust::device_pointer_cast(tempDos);
	thrust::device_ptr<REAL> p_Ematrix = thrust::device_pointer_cast(Ematrix);	
	REAL results[5];

	result = thrust::reduce(p_Ematrix, p_Ematrix + intN*intN);
	results[0] = result;	

	slowSwap<<<blocks,threads>>>( i,j, iPrev, j, intN, tempPar,tempPot,tempDos,particles,potentials,Ematrix,boxR,substrate,watcher);
	result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
	results[1] = result;	

        slowSwap<<<blocks,threads>>>( i,j, i, jPrev, intN, tempPar,tempPot,tempDos,particles,potentials,Ematrix,boxR,substrate,watcher);
        result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
	results[2] = result;

        slowSwap<<<blocks,threads>>>( i,j, iPost, j, intN, tempPar,tempPot,tempDos,particles,potentials,Ematrix,boxR,substrate,watcher);
        result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
	results[3] = result;	

        slowSwap<<<blocks,threads>>>( i,j, i, jPost, intN, tempPar,tempPot,tempDos,particles,potentials,Ematrix,boxR,substrate,watcher);
        result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
	results[4] = result;	

	C_particlePick(substrate,potentials, results,  boxR, Ematrix,particles, intN,  i,  j, threads,  blocks);	


}



void spiral(int index,double L,int intN,int blocks, int threads,REAL *particles,REAL *potentials,REAL *boxR,REAL *sumArray,REAL *Ematrix, REAL *tempDos,REAL* tempPar,REAL *tempPot,REAL *substrate,REAL *watcher) { 
	int xMod,yMod,nLevels,xStart,yStart,ringLevel,ringLength,xNow,yNow,xCount,yCount;
	nLevels = 5;
	xStart = index%intN;
        yStart = index/intN;
	

	for (ringLevel = 1; ringLevel < nLevels; ringLevel++) {
		ringLength = ringLevel * 2 +1; 
		xNow = xStart + ringLevel;
	        yNow = yStart + ringLevel;

		for (xCount = 1; xCount < ringLength; xCount++) {
			xNow = xNow - 1;
			yNow = yNow;
			xMod = C_mod(xNow,intN);
			yMod = C_mod(yNow,intN);
			fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, xMod, yMod,  intN, threads, blocks);
		}
		for (yCount = 1; yCount < ringLength; yCount++) {
			xNow = xNow;
			yNow = yNow - 1;
			xMod = C_mod(xNow,intN);
                        yMod = C_mod(yNow,intN);
			fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, xMod, yMod,  intN, threads, blocks);
		}
                for (xCount = 1; xCount < ringLength; xCount++) {
                        xNow = xNow + 1;
                        yNow = yNow;
                        xMod = C_mod(xNow,intN);
                        yMod = C_mod(yNow,intN);
                	fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, xMod, yMod,  intN, threads, blocks);
		}
		for (yCount = 1; yCount < ringLength; yCount++) {
                        xNow = xNow;
                        yNow = yNow + 1;
                        xMod = C_mod(xNow,intN);
                        yMod = C_mod(yNow,intN);
			fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, xMod, yMod,  intN, threads, blocks);
                }



	}

}


__global__ void checkRange(int index,REAL *rangeMatrix,int intN) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < intN*intN) {

	        int i,j,k,l;
	        double di,dj,dk,dl,r,dx,dy;
		i = index/intN;
		j = index%intN;
	        k = idx/intN;
	        l = idx%intN;

	
	        dk = (double) k;
	        dl = (double) l;
	    	di = (double) i;
       		dj = (double) j;

	        dx = dk - di;
        	dy = dl - dj;

	        r = sqrt(dx*dx + dy*dy);
        	rangeMatrix[idx] = 0;
		if (r < 10) {
			rangeMatrix[idx] = 1;
        	}
	}

}

int checkStable(REAL *watcher,REAL *substrate,REAL* tempPar,REAL *tempPot, REAL *tempDos,REAL *potentials,REAL *boxR,REAL * Ematrix, REAL * particles,int c_stable,REAL min_value,REAL max_value,int min_offset,int max_offset,int intN,int blocks,int threads){
	int i1,i2,j1,j2;
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

//	if(idx < 1) {	
//	if (min_value > max_value ) {
        if (min_value + max_value < 0) {
	
/*
		i1 = min_offset/intN;
		j1 = min_offset%intN;
		i2 = max_offset/intN;
		j2 = max_offset%intN;	
 */
                i1 = min_offset%intN;
                j1 = min_offset/intN;
                i2 = max_offset%intN;
                j2 = max_offset/intN;		


//cout<<i1<<" "<<j1<<" "<<i2<<" "<<j2<<endl;
//		slowSwap<<<blocks,threads>>>( i1,j1, i2, j2, intN, tempPar,tempPot,tempDos,particles,potentials,Ematrix,boxR,substrate,watcher); dont think i need this

                potSwap<<<blocks,threads>>>(i1, j1, i2, j2,intN,particles,boxR,potentials);
		particleSwap<<<blocks,threads>>>(i1, j1, i2,j2,intN,particles);
		findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

	
		c_stable = 0;
		
	}
	else c_stable = 1;
//	}
	return c_stable;
}

int highsToLows(int max_offset,int min_offset,REAL max_value,REAL min_value,int c_stable,REAL * sumArray,REAL *boxR, REAL *particles,REAL *potentials,REAL *rangeMatrix, REAL *Ematrix, REAL *tempDos,REAL *tempPar,REAL *tempPot,REAL *substrate,REAL* watcher, int intN,double L,int blocks,int threads) {
	c_stable = checkStable(watcher,substrate,tempPar,tempPot,tempDos,potentials,boxR, Ematrix,  particles, c_stable, min_value, max_value, min_offset,max_offset,intN, blocks,threads);

	if (c_stable == 0) {
		
		spiral(max_offset, L, intN,blocks, threads,particles,potentials,boxR,sumArray,Ematrix,tempDos,tempPar,tempPot,substrate,watcher);
		spiral(min_offset, L, intN,blocks, threads,particles,potentials,boxR,sumArray,Ematrix,tempDos,tempPar,tempPot,substrate,watcher);
	}
		
	return c_stable;
}

__global__ void grabPositives(REAL* particles,REAL *extraArray,REAL* Ematrix,int N,int posNeg) {
 int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (idx < N*N) {
		extraArray[idx] = 0;
		if(particles[idx] == posNeg) {
			extraArray[idx] = Ematrix[idx];
		}
	}


}

void __global__ lastFlip(int intN,REAL *invertedDos,REAL *particles) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < intN*intN) {
		if (particles[idx] == -1) {
			invertedDos[idx] = -invertedDos[idx];
		}
	}
}

void dosInvert (int intN,int threads,int blocks,REAL *invertedDos,REAL* particles,REAL *potentials,REAL *boxR,REAL* tempDos,REAL *tempPar,REAL * tempPot,REAL *Ematrix) {//should work for nParticles > 1
	int i,j;
	double  result1,result2;
	thrust::device_ptr<REAL> g_go =  thrust::device_pointer_cast(tempDos);

	

	for(j = 0; j < intN; j++) {
		for (i = 0; i < intN; i++) {

//i = 20;
//j = 20;

			matrixCopy<<<blocks,threads>>>(intN, potentials ,tempPot);
			matrixCopy<<<blocks,threads>>>(intN, particles , tempPar);
			matrixCopy<<<blocks,threads>>>(intN, Ematrix ,tempDos);

			potChange<<<blocks,threads>>>(i, j,  intN,tempPar,boxR,tempPot,tempDos);
//			dosChange<<<blocks,threads>>>(intN, tempPar,tempDos,tempPot);
			result1 = thrust::reduce(g_go, g_go + intN*intN);

			potChange<<<blocks,threads>>>(i, j,  intN,tempPar,boxR,tempPot,tempDos);
//			dosChange<<<blocks,threads>>>(intN, tempPar,tempDos,tempPot);
			result2 = thrust::reduce(g_go, g_go + intN*intN);

			dosPut<<<blocks,threads>>>( i, j,intN,invertedDos, result2 - result1);
			
//printBoxGPU(tempPot,intN,"dBox.txt");
//printBoxGPU(tempPar,intN,"pBox.txt");
				
		}
	}
//lastFlip<<<blocks,threads>>>(intN,invertedDos,tempPar);
//printBoxGPU(tempPar,intN,"pBox.txt");


}
void switcharoo(int c_stable,REAL *sumArray,REAL *rangeMatrix,REAL *substrate,REAL *extraArray,REAL *boxR,REAL *Ematrix, REAL *particles,REAL *potentials,REAL *tempDos,REAL *tempPar,REAL *tempPot,REAL *invertedDos,REAL *watcher,int intN, double L,int slices,int threads, int blocks) {
	int counter = 0;
	int min_offset,max_offset;
	REAL min_value,max_value;
	  thrust::device_ptr<REAL> g_ptr =  thrust::device_pointer_cast(Ematrix);
//	  thrust::device_ptr<REAL> extra_ptr =  thrust::device_pointer_cast(extraArray);
		thrust::device_ptr<REAL> inverted_ptr =  thrust::device_pointer_cast(extraArray);
//	dosInvert (intN,threads,blocks,invertedDos, particles,potentials,boxR, tempDos); 
        while (c_stable == 0) {
	
//		dosInvert (intN,threads,blocks,invertedDos, particles,potentials,boxR, tempDos,tempPar,tempPot,Ematrix);	     

		grabPositives<<<blocks,threads>>>(particles,extraArray,Ematrix,intN,1);

		min_offset = thrust::min_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr;
		min_value = *(inverted_ptr + min_offset);
//cout<<min_value<<endl;
	
		grabPositives<<<blocks,threads>>>(particles,extraArray,Ematrix,intN,-1);		

              max_offset = thrust::min_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr; //grabbing the smallest positive number
              max_value = *(inverted_ptr + max_offset);
//cout<<max_value<<endl;
//		max_offset = thrust::max_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr;
//		max_value = *(inverted_ptr + max_offset);
//cout<<max_value<<endl;

//	potentialse = *(g_ptr + max_offset);
	
		c_stable = highsToLows( max_offset,min_offset, max_value,min_value,c_stable, sumArray,boxR,particles,potentials,rangeMatrix, Ematrix, tempDos,tempPar,tempPot,substrate,watcher, intN, L, blocks,threads);
                if (counter >= 200) {
                        c_stable = 1;
                }
                else {
                        counter++;
                }

	}
	
}

void glatzRelax(int threads,int blocks,double L,double N,REAL* potentials,REAL *substrate, REAL *particles,REAL *boxR,REAL *Ematrix,REAL *tempDos,REAL *tempPar,REAL *tempPot,REAL *invertedDos,REAL *watcher) {

//        int sizeShared = 512*sizeof(REAL)/blocks;
        REAL *rangeMatrix,*extraArray,*sumArray,*hereSum;
	int i,j,intN,slices;
	intN = (int) N;
        slices = (intN*intN)/512 + 1;
	int c_stable = 0;
	int sizeSum = 6;
	hereSum = new REAL[sizeSum];
        hereSum = C_zeros(sizeSum,hereSum);

        cudaMalloc(&extraArray,N*N*sizeof(REAL));
	cudaMalloc(&rangeMatrix,N*N*sizeof(REAL));
	cudaMalloc(&sumArray,sizeSum*sizeof(REAL));
	cudaMemcpy(sumArray,hereSum,sizeSum*sizeof(REAL),cudaMemcpyHostToDevice);
	
//	G_dos(sumArray,extraArray,boxR,particles,substrate,Ematrix,potentials, slices,N, L, threads,blocks) ;

	for(j = 0; j < intN; j++) {
		for(i = 0; i < intN; i++) {
			potAdd<<<blocks,threads>>>( i, j,  intN,particles,boxR,potentials);

		}
	}
//	subAdd<<<blocks,threads>>>(intN,particles,potentials,substrate);
	
	findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate); 



for (int t = 0; t < 1; t++) {		
//original pair exchange
	for(j = 0; j < N; j++) {
	       	for(i = 0; i < N; i++) {
//			cout<<i<<" "<<j<<endl;
			fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, i, j,  intN, threads, blocks);
		}
	}
}


/*
particleDrop<<<blocks,threads>>>(intN, 16 ,15,1,particles);
potAdd<<<blocks,threads>>>( 16, 15,  intN,particles,boxR,potentials,substrate);
findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

particleDrop<<<blocks,threads>>>(intN, 1 ,16,1,particles);
potAdd<<<blocks,threads>>>( 1, 16,  intN,particles,boxR,potentials,substrate);
findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);
*/


//C_particleForce(substrate,potentials,boxR, Ematrix,particles,intN, 16, 16,16,15,threads, blocks);

/*
	i = 1;
	j = 1;

fastTest(watcher,substrate,potentials,Ematrix, tempDos,tempPar,tempPot, particles, boxR, i, j,  intN, threads, blocks);
*/


/*
//move all of i and the first half of j to the second half of j
int i1,i2,j1,j2;
-1or ( i1 = 0; i1 < intN;i1++) {
//for ( i1 = intN-1; i1 >= 0;i1--) {

	for (j1 = 0; j1 < intN/2; j1++) {
		i2 = i1;
		j2 = j1 + intN/2;
		C_particleForce(substrate,potentials, boxR, Ematrix,particles,intN, i1, j1,i2,j2,threads, blocks);
	}
}
*/



//C_particleForce(substrate,potentials,boxR, Ematrix,particles,intN, 5, 5,25,20,threads, blocks);

	errorAsk("pair exchange");




//highs to lows
	switcharoo(c_stable,sumArray,rangeMatrix,substrate,extraArray,boxR,Ematrix, particles,potentials,tempDos,tempPar,tempPot,invertedDos,watcher, N,  L,slices,threads, blocks);
	errorAsk("switching highs to lows");

/*
cout<<"artificial relaxation"<<endl;
particleDrop<<<blocks,threads>>>(intN, 1 , 8 ,0,particles);
potSub<<<blocks,threads>>>( 1, 8,  intN,particles,boxR,potentials);
findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);

particleDrop<<<blocks,threads>>>(intN, 1 ,24,1,particles);
potAdd<<<blocks,threads>>>( 1, 24,  intN,particles,boxR,potentials,substrate);
findE<<<blocks,threads>>>(intN, particles,Ematrix,potentials,substrate);
*/
//        switcharoo(c_stable,sumArray,rangeMatrix,substrate,extraArray,boxR,Ematrix, particles,potentials,tempDos,tempPar,tempPot,invertedDos, N,  L,slices,threads, blocks);
	dosInvert ( intN,threads,blocks,invertedDos, particles,potentials,boxR, tempDos,tempPar,tempPot,Ematrix);



	cudaFree(extraArray);
	cudaFree(rangeMatrix);


}

__global__ void	jumpFill(REAL* jumpRecord,int N) {
          int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	  if (idx < N) {
		jumpRecord[idx] = 999;
		
		
	}

}

__device__ REAL findMin(REAL d1, REAL d2, REAL d3, REAL d4) {
	
	if (d1 < d2 && d1 < d3 && d1 < d4) {
		return d1;
		
	}
        if (d2 < d1 && d2 < d3 && d2 < d4) {
		return d2;
		
        }
        if (d3 < d1 && d3 < d2 && d3 < d4) {
		return d3;
		
        }
        if (d4 < d1 && d4 < d2 && d4 < d3) {
		return d4;
		
        }

	return d1; //default
}

__global__ void aMaker(REAL *aMatrix,REAL *boxR,int N) {
        int i,j,iPrev,jPrev,iPost,jPost;
	REAL distanceUp,distanceDown,distanceLeft,distanceRight, minDistance;
        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < N*N) {
	
                i = idx%(N);
                j = (idx%(N*N) - idx%(N))/N;
		

	        iPrev = G_mod(i - 1,N);
	        jPrev = G_mod(j - 1,N);
        	iPost = G_mod(i + 1,N);
	        jPost = G_mod(j + 1,N);
	
		distanceUp = boxR[i + N*j + N*N*i + N*N*N*jPost];
		distanceDown = boxR[i + N*j + N*N*i + N*N*N*jPrev];
		distanceLeft = boxR[i + N*j + N*N*iPrev + N*N*N*j];
		distanceRight = boxR[i + N*j + N*N*iPost + N*N*N*j];

		minDistance = findMin(distanceUp,distanceDown,distanceLeft,distanceRight);
		aMatrix[idx] = minDistance/2;

        }

}

__global__ void subCombine(REAL *aMatrix,REAL *substrate,REAL L, int N) {

        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < N*N) {
	substrate[idx] = substrate[idx]*aMatrix[idx]/L;

	}

}

int main(int argc,char *argv[])
{
	int threads,blocks;
	int N,t,tSteps,nParticles,relax,grabJ;
	double xi,muVar,xVar,yVar,eV,Ec,L,T,alphaOne,alphaTwo;
cudaDeviceReset();
cudaSetDevice(0);
cudaDeviceSynchronize();
cudaThreadSynchronize();

	srand48(time(0));

//	N = 32;
	N = 100;
//	N = 256;
	muVar = 0;
//	muVar = 1e-5;
	
//	eV = .05;
	eV = 0;
	Ec = 16000;
//	Ec = 1.6e-5;
//	Ec = 1;
//	T = 1;
	alphaOne = 1; // technically combined with density of states
//	alphaTwo = 1e7; // technically combined with e^2 and epsilon
	alphaTwo = 1.16e4; //C/Kb
	T = 25;
//	nParticles = input;
	nParticles = .5*N*N;
//	nParticles = 1;
//	L = 7e-6;	
	L = 1e-8; //10 nm
//	tSteps = 1000000; //for statistically accurate runs
//	tSteps = 10000; //for potential runs
	tSteps = 1; // for seeing the fields
//	tSteps = 0;
//	relax = 1;
	relax = 0; 
	grabJ=0;	

	REAL *reducedProb,*particles,*probabilities,*potentials,*substrate,*hereP,*hereProb,*herePot,*hereS,*boxR,*hereBoxR,*hereXDiff,*hereYDiff,*Ematrix,*jumpRecord,*tempDos,*tempPar,*tempPot,*invertedDos,*watcher,*aMatrix;
	xi = L;
	xVar = 0;
	yVar = 0;

//	char *lineName;
//       char *boxName;
	char boxName[256];
	char lineName[256];
	sprintf(lineName, "line.txt");
	sprintf(boxName, "box.txt");


int intVal;
REAL realVal;
ifstream is_file(argv[1]);
string line;
while( getline(is_file, line) )
{
 istringstream is_line(line);
 string key;
  if( getline(is_line, key, '=') )
  {
    string value;
    if( getline(is_line, value) )
//      store_line(key, value);
        if(key == "Temp") {
                realVal = atof(value.c_str());
                T = realVal;
        }

        if(key == "muVar") {
                realVal = atof(value.c_str());
                muVar = realVal;
        }


	if(key == "XYvar") {
                realVal = atof(value.c_str());
                xVar = realVal*L;
		yVar = realVal*L;
        }
	
        if(key == "tSteps") {
                intVal = atoi(value.c_str());
                tSteps = intVal;
        }

        if(key == "L") {
                realVal = atof(value.c_str());
                L = realVal;
        }

	if(key == "eV") {
                realVal = atof(value.c_str());
                eV = realVal;
        }

	if(key == "relax") {
                intVal = atoi(value.c_str());
                relax = intVal;
        }
	
	if(key == "grabJ") {
                intVal = atoi(value.c_str());
                grabJ = intVal;
        }

	if(key == "lineName") {
                 sprintf(lineName,  value.c_str());

//                lineName = value.c_str();
        }

        if(key == "boxName") {
                 sprintf(boxName,  value.c_str());
//                boxName = value.c_str();
        }

  }
}


//	xi = 1; // xi/a	
	clock_t begin = clock();
	

	threads=MAXT;
	blocks=N*N/threads+(N*N%threads==0?0:1);
	cudaMalloc(&watcher,N*N*sizeof(REAL));
        cudaMalloc(&reducedProb,N*N*sizeof(REAL));
	cudaMalloc(&particles,N*N*sizeof(REAL));
	cudaMalloc(&probabilities,N*N*sizeof(REAL));
	cudaMalloc(&potentials,N*N*sizeof(REAL));
	cudaMalloc(&substrate,N*N*sizeof(REAL));
	cudaMalloc(&Ematrix,N*N*sizeof(REAL));
        cudaMalloc(&tempDos,N*N*sizeof(REAL));
	cudaMalloc(&tempPar,N*N*sizeof(REAL));
	cudaMalloc(&tempPot,N*N*sizeof(REAL));
        cudaMalloc(&invertedDos,N*N*sizeof(REAL));
        cudaMalloc(&jumpRecord,N*N*sizeof(REAL));
	cudaMalloc(&aMatrix,N*N*sizeof(REAL));
	cudaMalloc(&boxR,N*N*N*N*sizeof(REAL));
	

	herePot =  new REAL[N*N];
	herePot = C_zeros(N, herePot); 
	hereProb = new REAL[N*N];	
	hereProb = C_random(N,0,hereProb);
	hereP = new REAL[N*N];
//	hereP = C_clump(N,nParticles,hereP);//test relaxation
	hereP = C_spread(N,nParticles,hereP); //test general potential
//	hereP = C_random(N,nParticles,hereP);	
//	hereP = C_random(N,0,hereP); //empty system
//        hereP = C_more(N,nParticles,hereP);
	hereXDiff = new REAL[N*N];
	hereYDiff = new REAL[N*N];
	hereXDiff = createDiff(hereXDiff, xVar, N);
	hereYDiff = createDiff(hereYDiff, yVar, N);



	hereS = new REAL[N*N];
	hereS = createSub(hereS,muVar,N);
	hereBoxR = new REAL[N*N*N*N];
	hereBoxR = createR(hereBoxR,hereXDiff,hereYDiff,N,L,xi);
	cudaMemcpy(watcher,herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(potentials,herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(Ematrix,herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);//just filling it with 0s
        cudaMemcpy(substrate,hereS,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(boxR,hereBoxR,N*N*N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(particles,hereP,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
	aMaker<<<blocks,threads>>>(aMatrix,boxR,N);
	subCombine<<<blocks,threads>>>(aMatrix,substrate, L, N);
/*	
	char    nameP[256];
        sprintf(nameP, "line.txt");
	hereP = loadMatrix(hereP,nameP);
 */       

	jumpFill<<<blocks,threads>>>(jumpRecord,10000);
//system is run but results arent output for the relaxation phase
        if (relax == 1) {
		glatzRelax(threads, blocks, L, N, potentials,substrate, particles, boxR,Ematrix,tempDos,tempPar,tempPot,invertedDos,watcher);
	}
	
	
//find the DoS

//      Ematrix = dosFind(hereP, hereS,herePot,Ematrix, particles,potentials,boxR, N, L, threads, blocks);  
//	showMove(hereXDiff,N);  



	for(t = 0; t < tSteps ; t++) {
		countThese = 1;
		findJump(hereP,hereProb,herePot,particles,probabilities,potentials,substrate,reducedProb,jumpRecord, N, xi, threads, blocks,eV,Ec,L,T,boxR,alphaOne,alphaTwo,grabJ);
	}

//	sprintf(str1, "line.txt");
//	printBoxCPU(hereXDiff,N,boxName);
	lastFlip<<<blocks,threads>>>(N,Ematrix,particles);
//	printBoxGPU(particles,N,boxName);
	printBoxGPU(Ematrix,N,boxName);
//        printBoxGPU(particles,N,lineName);

	printLineGPU(jumpRecord,10000,lineName);
/*
        cudaMemcpy(hereP,jumpRecord,N*N*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
//        char    str1[256];
  //      sprintf(str1, "particles.txt");
        fp1 = fopen(fileName, "w");
	for (int k = 0; k < N*N ; k++){
                fprintf(fp1, "%lf ",hereP[k]);
        }
//cleanup
	fclose(fp1);


*/
	
        delete[] herePot;
        delete[] hereProb;
        delete[] hereP;
        delete[] hereS;
	delete[] hereBoxR;
	cudaFree(particles);
	cudaFree(probabilities);
        cudaFree(potentials);
        cudaFree(substrate);
	cudaFree(boxR);
	cudaFree(Ematrix);
	cudaFree(jumpRecord);
	
	clock_t end = clock();
//  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

//cout<<currentCount<<endl;
//cout<<"this took "<<elapsed_secs<<" seconds"<<endl;
}
