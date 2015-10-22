//advanced cuda system
//sped up algorithms
//dosMott5: like dosMott4 but with reduce instead of scan


/*
	Code guide: first matrices are initialized. they are used to keep track of the particles, the probabilities to jump, the substrate, and the general electric potential.
	Input parameters are also taken in. Currently the code takes in one parameter. The rest of the parameters must be adjusted manually and the code must be recompiled. The general electric potential is calculated in cuda. This reduces a n^4 problem to a n^2 one. A site is picked at random at the CPU (part of the monte-carlo process) and the probabilities of interaction with the particles around it are calculated at the GPU. The probabilities are then returned to the CPU where the second part of the Monte-Carlo algorithm occurs. Here, the site which the subject particle will interact with is chosen randomly but with weights according to the probabilities. The jump is made, and the system starts over.  



*/

#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <unistd.h> /* for getpid() */
#include <time.h> /* for time() */
#include <math.h>
#include <assert.h>
#include <iostream>
#include <ctime>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cuda.h>

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



//Here, the gpu's find the general electric potential at each lattice site. 
__global__ void findPotential(REAL *particles,REAL *potentials, double N,double L, REAL *boxR) { 
        int i,j,intx,inty,checkx,checky,distancex,distancey;
	int intN = (int) N;
        int checkRange = 50; //(*2)
  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

        double k,l,sum,distanceTerm;
//	double deltax,deltay;
	 if(idx<intN*intN) {
        	i = idx/intN;
                j = idx%intN;
		sum = 0;
                       for(l = 0 ; l < checkRange*2; l++) {
                                for(k = 0; k < checkRange*2; k++) {
                                        checkx = G_mod(i + k - checkRange,N);
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
                potentials[i + intN*j] = sum;

	}
}

__global__ void potOnParticles(REAL *particles,REAL *potentials,int intN, double L,REAL *boxR) {
        int i,j,intx,inty,checkx,checky,distancex,distancey;
        double N = (double) intN;
        int checkRange = N/2; //(*2)
  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

        double k,l,sum,distanceTerm;
//      double deltax,deltay;
         if(idx<intN*intN) {
           
		     i = idx/intN;
                j = idx%intN;
                sum = 0;
		if (particles[i + intN * j] > 0 ) {
                       for(l = 0 ; l < checkRange*2; l++) {
                                for(k = 0; k < checkRange*2; k++) {
                                        checkx = G_mod(i + k - checkRange,N);
                                        checky = G_mod(j + l - checkRange,N);

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
                potentials[i + intN*j] = sum*particles[i + intN*j];

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
	

        if ((thisp == 1) && (p == 0 )) {
                return 0; //no blockade penalty
        }
        if ((thisp == 0) && (p == 1 )) {
                return 0;
        }
        if ((thisp == 0) && (p == 2 )) {
                return 2*Ec;
        }
        if ((thisp == 2) && (p == 0 )) { //not sure about this one figured twice the electrons means twice the penalty.
                return 2*Ec;
        }
        if ((thisp == 1) && (p == 1 )) {
                return Ec;
        }
        if ((thisp == 1) && (p == 2 )) {
                return 0;
        }
        if ((thisp == 2) && (p == 1 )) {
                return 0;
        }
	if ((thisp == 2) && (p == 2 )) { //no interaction
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
	
	potConstant = 1.17e-13;
//	potConstant = Ec;
//	potConstant = 0;
	
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
			potentialPart = -potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y]);
			substratePart = substrate[thisi+ N*thisj];
			currentPart = eV*i;

//			currentPart = 0;			
//			blockadePart = 0;
//			potentialPart= 0;
//			substratePart= 0;

			
		}

		if (particles[x + N*y] < particles[thisi + N*thisj]) {

			blockadePart = -1*findBlockade(p,thisp,Ec)/boxR[hyperIndex]; 
			potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y]);
			substratePart = -substrate[thisi + N*thisj];
			currentPart = -eV*i;

//			currentPart = 0;
//			substratePart = 0;
//			potentialPart = 0;
//			blockadePart = 0;

		}

		if ( particles[x + N*y] == particles[thisi + N*thisj] ){

			if (p > 0 ) {
				currentPart  = eV*i;
			}

			if (p == 0 ) {
                                currentPart  = -eV*i;
                        }


			substratePart = -substrate[thisi+ N*thisj];
			blockadePart = -1*findBlockade(p,thisp,Ec)/boxR[hyperIndex];
			potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y]);
	
//			currentPart = 0;
//			substratePart = 0;
//			potentialPart = 0;
//			blockadePart = 0;

		}


//	probabilities[idx] = exp(distancePart+(blockadePart+potentialPart+substratePart+currentPart)/T);
	probabilities[idx] = exp(distancePart+alphaTwo*(blockadePart+potentialPart+substratePart+currentPart)/T);

//        probabilities[idx] = exp(distancePart+(substratePart+currentPart)/T);
//	probabilities[idx] = distancePart+(blockadePart+potentialPart+substratePart+currentPart)/T;	
//	probabilities[idx] = potentialPart*alphaTwo;	
	if (probabilities[idx] > 1) {
		probabilities[idx] = 1;
	}

	if ((thisi==x && thisj==y )  ){
		probabilities[idx] = 1; //force probability of jumping to self to 1 (avoids 0/0 problems)
	}
	}

};

//figures out which way the electron jump will occur and also calculates the current or jump distance (since particle movement is also done here).
 __device__ void interaction(int x,int y,int newx,int newy,int N,REAL *particles) {
	double current,totalCurrent = 0;
	int whichWay = 0;
        if ((particles[x + y*N] == 0 ) && ( particles[newx + newy*N] == 0 ) ) {
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
		particles[x + y*N] = particles[x + y*N] - 1;
                particles[newx + newy*N] = particles[newx + newy*N] + 1;
		
	}

	else if (whichWay < 0) {
	        particles[x + y*N] = particles[x + y*N] + 1;
                particles[newx + newy*N] = particles[newx + newy*N] - 1;
	}


totalCurrent = totalCurrent + current;
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

__global__ void particleJump(int x, int y,double randomNum,int N,REAL *reducedProb,REAL *particles) {
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        double pickedValue = randomNum*reducedProb[N*N -1];
	int newx,newy,lastx,lasty;
	if (idx > 0) {
		if ((reducedProb[idx - 1] < pickedValue) && (reducedProb[idx] > pickedValue)) {
			lastx = idx/N;
        		lasty = idx%N;
		        newx = G_mod(x - N/2 +  lastx,N);
		        newy = G_mod(y - N/2 +  lasty,N);
					
		}
	}
	if (idx == 0) {
		if (pickedValue < reducedProb[0]) {
	                lastx = idx/N;
                        lasty = idx%N;
		        newx = G_mod(x - N/2 +  lastx,N);
		        newy = G_mod(y - N/2 +  lasty,N);
		}	
	}

	interaction(x,y,newx,newy,N,particles);	

}

//second part of the heart of this code. Here the probabilities are summed and a number is picked from 0 to that number. The code then sums through the probabilities untill it reaches that number. In this way, probabilities which are higher will have a larger chance of getting picked. 
void particleScout(REAL *reducedProb,REAL* particles,REAL* probabilities,int x,int y,int N,double randomNum,int blocks, int threads) {
        
	thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(probabilities);
        thrust::device_ptr<REAL> g_return = thrust::device_pointer_cast(reducedProb);

        thrust::inclusive_scan(g_go, g_go + N*N, g_return); // in-place scan 
		
	particleJump<<<blocks,threads>>>( x, y,randomNum,N,reducedProb,particles);
        errorAsk("particleJump");
}


void printGPU(REAL *g_array,int size) {
	REAL *c_array;
        c_array =  new REAL[size*size];
	int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        char    str1[256];
        sprintf(str1, "particles.txt");
        fp1 = fopen(str1, "w");
        for (k = 0; k < size ; k++){
		for(l = 0; l < size; l++) {

                	fprintf(fp1, "%lf ",c_array[k + l*size]);
		}
	fprintf(fp1,"\n");
        }

//cleanup
        fclose(fp1);
}

//the particles are picked here. This is also where the system is run from. (find potential, find probabilities, and move particle are done here)
void findJump(REAL* hereP,REAL* hereProb,REAL* herePot,REAL *particles,REAL *probabilities,REAL *potentials,REAL *substrate,REAL *reducedProb,int N,double xi,int threads,int blocks,double eV,double Ec,double L,double T,REAL *boxR, double alphaOne, double alphaTwo) {
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
	particleScout(reducedProb, particles,probabilities, x, y, N,randomNum, blocks, threads);
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

	sumArray[index] = addSub*numToInsert;

}

__global__ void particleSwitch(int i,int j,int intN,REAL *particles) {
	if (particles[i + j*intN] == 0) particles[i + j*intN]= 1;
	else particles[i + j*intN]= 0;
}

__global__ void dosPut(int i,int j,int intN,REAL *dosMatrix,REAL sum) {
	dosMatrix[i + j*intN] =	sum;
}

 void G_dos(REAL * sumArray,REAL *extraArray,REAL *boxR,REAL *particles,REAL *substrate,REAL *reducedSum,REAL *dosMatrix,REAL *potentials,REAL *g_temp,int slices,double N,double L,int threads,int blocks) {
	int i,j,intN;//not sure about Sums
	intN = (int) N;
        thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(potentials);
        thrust::device_ptr<REAL> g_return = thrust::device_pointer_cast(reducedSum);
	thrust::device_ptr<REAL> sumArrayPtr = thrust::device_pointer_cast(sumArray);
	thrust::device_ptr<REAL> extraArrayPtr = thrust::device_pointer_cast(extraArray);
	REAL result;

        for (j = 0; j < N; j++) {
                for (i = 0; i < N; i++) {
			potOnParticles<<<threads,blocks>>>(particles,potentials, N,L,boxR);
			result = thrust::reduce(g_go, g_go + intN*intN);        
			fillSum<<<blocks,threads>>>(0,intN,-1,sumArray,result);

			G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(1,intN,-1,sumArray,result);

			G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(2,intN,-1,sumArray,result);
			
			particleSwitch<<<blocks,threads>>>(i,j,intN,particles);

                        potOnParticles<<<threads,blocks>>>(particles,potentials, N,L,boxR);
			result = thrust::reduce(g_go, g_go + intN*intN);
			fillSum<<<blocks,threads>>>(3,intN,1,sumArray,result);

                        G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
                        result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(4,intN,1,sumArray,result);

                        G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
                	result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(5,intN,1,sumArray,result);

                        particleSwitch<<<blocks,threads>>>(i,j,intN,particles);
			
			result = thrust::reduce(sumArrayPtr, sumArrayPtr + 6);
	
			dosPut<<<blocks,threads>>>(i,j,intN,dosMatrix,result);
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
                        A[index] = 0;


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

                doublek = (double) k;
                doublel = (double) l;

		kNew = i + k - N/2;
		lNew = j + l - N/2;

                kObs = C_mod(kNew,N);
                lObs = C_mod(lNew,N);
		
		diffXThere = diffX[kObs];
		diffXHere = diffX[i];
		if((kNew < 0) || (kNew > N)) {
			diffXHere = -diffX[i];
			diffXThere = -diffX[kObs];
		}
		diffYThere = diffY[lObs];
		diffYHere = diffY[j];
		if((lNew < 0) || (lNew > N)) {
			diffYHere = -diffY[j];
                        diffYThere = -diffY[lObs];
                }

		deltaX = diffXHere - (diffXThere + L*(doublek - N/2));
		deltaY = diffYHere - (diffYThere + L*(doublel - N/2));
		
		r = sqrt(deltaX*deltaX + deltaY*deltaY);

		A[idx] = r;


	}

return A;
	
}

//clumps all of the original electrons ( to show relaxation)
REAL *C_clump(double N,double nparticles,REAL *A) {
        int idx;

	for (idx = 0;idx < N*N; idx++) {
		A[idx] = 0;
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
                A[idx] = 0;
        }

        for (idx = 0; idx < N*N; idx++) {
        	i = idx/N;
        	j = idx%intN;
		
		if((i + j)%2) {        
			A[idx] = 1;
        	}
	}


return A;
}

__global__ void particleSwap(int i,int j,int k,int l,int intN,REAL *particles) {
	int temp;
	temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
	particles[k + l*intN] = temp;
}

__device__ void g_particleSwap(int i,int j,int k,int l,int intN,REAL *particles){
	
        int temp;
        temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
        particles[k + l*intN] = temp;	
}

__global__ void particlePick(int i,int j,int intN,REAL *particles,REAL *sumArray) {

	if ((-sumArray[0] < sumArray[1] ) ||(-sumArray[0] < sumArray[2] ) ||(-sumArray[0] < sumArray[3] ) ||(-sumArray[0] < sumArray[4] ) ) {

        int iPrev,jPrev,iPost,jPost;
        iPrev = G_mod(i - 1,intN);
        jPrev = G_mod(j - 1,intN);
        iPost = G_mod(i + 1,intN);
        jPost = G_mod(j + 1,intN);


		if ((sumArray[1] > sumArray[2] ) &&(sumArray[1] > sumArray[3] ) &&(sumArray[1] > sumArray[4] )  ) {
			g_particleSwap(i,j,iPrev,j,intN,particles);
		}

		else if ((sumArray[2] > sumArray[3] ) &&(sumArray[2] > sumArray[4] )) {
                        g_particleSwap(i,j,i,jPrev,intN,particles);		
		}

		else if (sumArray[3] > sumArray[4]) {
	                g_particleSwap(i,j,iPost,j,intN,particles);
 		}

		else {
                        g_particleSwap(i,j,i,jPost,intN,particles);
		}

	}
}

void testMove(double L,int i, int j,int intN,int blocks, int threads,REAL *particles,REAL *potentials,REAL *g_itemp, REAL *g_otemp,REAL *boxR,REAL *sumArray) {
	int iPrev,jPrev,iPost,jPost;
       	iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
	jPost = C_mod(j + 1,intN);
	REAL result;

        thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(potentials);

 	potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
        result = thrust::reduce(g_go, g_go + intN*intN);
	fillSum<<<blocks,threads>>>(0,intN,-1,sumArray,result);

	particleSwap<<<blocks,threads>>>(i,j,iPrev,j,intN,particles);
        potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
        result = thrust::reduce(g_go, g_go + intN*intN);
        fillSum<<<blocks,threads>>>(1,intN,-1,sumArray,result);
	particleSwap<<<blocks,threads>>>(i,j,iPrev,j,intN,particles); //A' = A

	particleSwap<<<blocks,threads>>>(i,j,i,jPrev,intN,particles);
	potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
        result = thrust::reduce(g_go, g_go + intN*intN);
        fillSum<<<blocks,threads>>>(2,intN,-1,sumArray,result);
	particleSwap<<<blocks,threads>>>(i,j,i,jPrev,intN,particles);

        particleSwap<<<blocks,threads>>>(i,j,iPost,j,intN,particles);
        potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
        result = thrust::reduce(g_go, g_go + intN*intN);
        fillSum<<<blocks,threads>>>(3,intN,-1,sumArray,result);
        particleSwap<<<blocks,threads>>>(i,j,iPost,j,intN,particles);
	
	particleSwap<<<blocks,threads>>>(i,j,i,jPost,intN,particles);
        potOnParticles<<<blocks,threads>>>(particles,potentials, intN,L,boxR);
        result = thrust::reduce(g_go, g_go + intN*intN);
        fillSum<<<blocks,threads>>>(4,intN,-1,sumArray,result);
        particleSwap<<<blocks,threads>>>(i,j,i,jPost,intN,particles);

			
	particlePick<<<blocks,threads>>>(i,j, intN,particles,sumArray);


}
__global__ void potOnParticles2(REAL *particles,REAL *potentials,REAL *rangeMatrix,int intN, double L,REAL *boxR,int k, int l) {//no other way 
        int i,j,intx,inty,checkx,checky,distancex,distancey;
        double N = (double) intN;
        int checkRange = N/2; //(*2)
	if (rangeMatrix[k + intN*l] == 1) {
  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

        double k,l,sum,distanceTerm;
//      double deltax,deltay;
         if(idx<intN*intN) {

                     i = idx/intN;
                j = idx%intN;
                sum = 0;
                if (particles[i + intN * j] > 0 ) {
                       for(l = 0 ; l < checkRange*2; l++) {
                                for(k = 0; k < checkRange*2; k++) {
                                        checkx = G_mod(i + k - checkRange,N);
                                        checky = G_mod(j + l - checkRange,N);

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
                potentials[i + intN*j] = sum*particles[i + intN*j];

        }

 	}
}

__global__ void fillSum2(int index,int intN,int addSub,REAL result,REAL *sumArray,REAL *rangeMatrix,int k, int l) {
	if (rangeMatrix[k + intN*l] == 1) {
        sumArray[index] = addSub*result;
	}
}
__global__ void particleSwap2(int i,int j,int k,int l,int intN,REAL *particles,REAL *rangeMatrix, int q, int w) {
	if (rangeMatrix[q + intN*w] == 1) {

        int temp;
        temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
        particles[k + l*intN] = temp;

	}
}

__global__ void particlePick2(int i,int j,int intN,REAL *particles,REAL *sumArray,REAL *rangeMatrix,int q, int w) {
	 if (rangeMatrix[q + intN*w] == 1) {
	
        if ((-sumArray[0] < sumArray[1] ) ||(-sumArray[0] < sumArray[2] ) ||(-sumArray[0] < sumArray[3] ) ||(-sumArray[0] < sumArray[4] ) ) {

        int iPrev,jPrev,iPost,jPost;
        iPrev = G_mod(i - 1,intN);
        jPrev = G_mod(j - 1,intN);
        iPost = G_mod(i + 1,intN);
        jPost = G_mod(j + 1,intN);


                if ((sumArray[1] > sumArray[2] ) &&(sumArray[1] > sumArray[3] ) &&(sumArray[1] > sumArray[4] )  ) {
                        g_particleSwap(i,j,iPrev,j,intN,particles);
                }

                else if ((sumArray[2] > sumArray[3] ) &&(sumArray[2] > sumArray[4] )) {
                        g_particleSwap(i,j,i,jPrev,intN,particles);
                }

                else if (sumArray[3] > sumArray[4]) {
                        g_particleSwap(i,j,iPost,j,intN,particles);
                }

                else {
                        g_particleSwap(i,j,i,jPost,intN,particles);
     }

        }
	}
}



void testMove2(double L,int i, int j,int intN,int blocks, int threads,REAL *particles,REAL *potentials,REAL *g_itemp, REAL *g_otemp,REAL *boxR,REAL *sumArray,REAL *rangeMatrix) {
        int iPrev,jPrev,iPost,jPost;
	REAL result;
        iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
        jPost = C_mod(j + 1,intN);

        thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(potentials);


        potOnParticles2<<<blocks,threads>>>(particles,potentials,rangeMatrix, intN,L,boxR,i,j);
	result = thrust::reduce(g_go,  g_go + intN*intN);
	fillSum2<<<blocks,threads>>>(0,intN,-1,result,sumArray,rangeMatrix, i,  j);

        particleSwap2<<<blocks,threads>>>(i,j,iPrev,j,intN,particles,rangeMatrix,i,j);
        potOnParticles2<<<blocks,threads>>>(particles,potentials,rangeMatrix, intN,L,boxR,i,j);
        result = thrust::reduce(g_go,  g_go + intN*intN);
        fillSum2<<<blocks,threads>>>(1,intN,-1,result,sumArray,rangeMatrix, i,  j);
	particleSwap2<<<blocks,threads>>>(i,j,iPrev,j,intN,particles,rangeMatrix,i,j); //A' = A

        particleSwap2<<<blocks,threads>>>(i,j,i,jPrev,intN,particles,rangeMatrix,i,j);
        potOnParticles2<<<blocks,threads>>>(particles,potentials,rangeMatrix, intN,L,boxR,i,j);
        result = thrust::reduce(g_go,  g_go + intN*intN);
        fillSum2<<<blocks,threads>>>(2,intN,-1,result,sumArray,rangeMatrix, i,  j);
	particleSwap2<<<blocks,threads>>>(i,j,i,jPrev,intN,particles,rangeMatrix,i,j);

        particleSwap2<<<blocks,threads>>>(i,j,iPost,j,intN,particles,rangeMatrix,i,j);
        potOnParticles2<<<blocks,threads>>>(particles,potentials,rangeMatrix, intN,L,boxR,i,j);
        result = thrust::reduce(g_go,  g_go + intN*intN);
        fillSum2<<<blocks,threads>>>(3,intN,-1,result,sumArray,rangeMatrix, i,  j);	
	particleSwap2<<<blocks,threads>>>(i,j,iPost,j,intN,particles,rangeMatrix,i,j);

        particleSwap2<<<blocks,threads>>>(i,j,i,jPost,intN,particles,rangeMatrix,i,j);
        potOnParticles2<<<blocks,threads>>>(particles,potentials,rangeMatrix, intN,L,boxR,i,j);
        result = thrust::reduce(g_go,  g_go + intN*intN);
        fillSum2<<<blocks,threads>>>(4,intN,-1,result,sumArray,rangeMatrix, i,  j);
	particleSwap2<<<blocks,threads>>>(i,j,i,jPost,intN,particles,rangeMatrix,i,j);


        particlePick2<<<blocks,threads>>>(i,j, intN,particles,sumArray,rangeMatrix,i,j);


}

void checkArea(double L,int intN,int blocks, int threads,REAL *particles,REAL *potentials,REAL *reducedSum,REAL *rangeMatrix,REAL *g_itemp,REAL *g_otemp,REAL *boxR,REAL *sumArray) {
	int i,j;
	for (int n = 0; n < intN*intN;n++) {
		        i = n/intN;
		        j = n%intN;
			testMove2( L,i,  j,intN, blocks, threads,particles,potentials,g_itemp,g_otemp,boxR,sumArray,rangeMatrix); 

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

__global__ void checkStable(REAL * particles,int *g_stable,REAL min_value,REAL max_value,int min_offset,int max_offset){
	int temp;
	
	if (min_value + max_value > 0) {
		g_stable[0] = 0;
		temp = particles[min_offset];
                particles[min_offset] = particles[max_offset];
                particles[max_offset] = temp;
		
	}
	else g_stable[0] = 1;

}

int *highsToLows(int max_offset,int min_offset,REAL max_value,REAL min_value,int *g_stable,int *c_stable,REAL * sumArray,REAL *boxR,REAL *g_itemp,REAL *g_otemp, REAL *particles,REAL *potentials,REAL *reducedSum,REAL *rangeMatrix, int N,double L,int blocks,int threads) {
	
	checkStable<<<blocks,threads>>>(particles,g_stable, min_value, max_value, min_offset, max_offset);	
	cudaMemcpy(c_stable,g_stable,sizeof(int),cudaMemcpyDeviceToHost);

	if (c_stable == 0) {
		
		checkRange<<<blocks,threads>>>(max_offset,rangeMatrix,N);
		checkArea(L,N,blocks,threads,particles,potentials,reducedSum,rangeMatrix,g_itemp,g_otemp,boxR,sumArray); //look at area around max

                checkRange<<<blocks,threads>>>(min_offset,rangeMatrix,N);
                checkArea(L,N,blocks,threads,particles,potentials,reducedSum,rangeMatrix,g_itemp,g_otemp,boxR,sumArray); //look at area around min
	
	}
		
	return c_stable;
}

void switcharoo(int *c_stable,int *g_stable,REAL *sumArray,REAL *rangeMatrix,REAL *g_temp,REAL *substrate,REAL *extraArray,REAL *g_itemp,REAL *g_otemp,REAL *boxR,REAL *dosMatrix, REAL *particles,REAL *potentials,REAL *reducedSum,int N, double L,int slices,int threads, int blocks) {

	int min_offset,max_offset;
	REAL min_value,max_value;
	  thrust::device_ptr<REAL> g_ptr =  thrust::device_pointer_cast(dosMatrix);
	
        while (c_stable[0] == 0) {
		G_dos(sumArray,extraArray,boxR,particles,substrate,reducedSum,dosMatrix,potentials,g_temp, slices,N, L, threads,blocks) ;
	     
		min_offset = thrust::min_element(g_ptr, g_ptr + N) - g_ptr;
		min_value = *(g_ptr + min_offset);

		max_offset = thrust::max_element(g_ptr, g_ptr + N) - g_ptr;
		max_value = *(g_ptr + max_offset);
	
		c_stable = highsToLows( max_offset,min_offset, max_value,min_value,g_stable,c_stable, sumArray,boxR,g_itemp,g_otemp,particles,potentials,reducedSum,rangeMatrix, N, L, blocks,threads);
	}
	
	


}

void pairExchange(REAL *sumArray,REAL *particles,REAL *potentials,REAL *boxR,REAL *g_itemp,REAL *g_otemp,REAL *reducedSum, double N, double L, int slices, int i, int j,int threads, int blocks) {
	int intN = (int) N;
	

	testMove( L,i, j, intN, blocks, threads,particles,potentials,g_itemp, g_otemp,boxR,sumArray);
	
}


void glatzRelax(int threads,int blocks,double L,double N,REAL* potentials,REAL *substrate, REAL *particles, REAL *reducedSum,REAL *g_itemp, REAL *g_otemp,REAL *boxR,REAL *g_temp,REAL *dosMatrix) {

//        int sizeShared = 512*sizeof(REAL)/blocks;
        REAL *rangeMatrix,*extraArray,*sumArray,*hereSum;
	int *g_stable,*c_stable;
	int i,j,intN,slices;
	intN = (int) N;
        slices = (intN*intN)/512 + 1;
	c_stable =  new int[0];		
	c_stable[0] = 0;
	hereSum = new REAL[10];
        hereSum = C_zeros(10,hereSum);

        cudaMalloc(&extraArray,N*N*sizeof(REAL));
	cudaMalloc(&rangeMatrix,N*N*sizeof(REAL));
	cudaMalloc(&sumArray,10*sizeof(REAL));
	cudaMalloc(&g_stable,sizeof(int));
	cudaMemcpy(sumArray,hereSum,10*sizeof(REAL),cudaMemcpyHostToDevice);
	
		
//original pair exchange
	for(j = 0; j < N; j++) {
	       	for(i = 0; i < N; i++) {
			pairExchange(sumArray,particles,potentials,boxR,g_itemp,g_otemp,reducedSum,  N,  L, slices,  i, j,threads,blocks);
//			cout<<i<<" "<<j<<endl;
		}
	}

//	i = 1;
//	j = 2;
//	pairExchange<<<blocks,threads,sizeShared>>>(particles,potentials,boxR,g_itemp,g_otemp,reducedSum, N, L, slices,  i,j);
	
	errorAsk("pair exchange");




//highs to lows
	switcharoo(c_stable,g_stable,sumArray,rangeMatrix,g_temp,substrate,extraArray,g_itemp,g_otemp,boxR,dosMatrix, particles,potentials,reducedSum, N,  L,slices,threads, blocks);
	errorAsk("switching highs to lows");


	cudaFree(extraArray);
	cudaFree(rangeMatrix);
	cudaFree(g_stable);


}



int main(int argc,char *argv[])
{
	int threads,blocks;
	int N,t,tSteps,nParticles,relax;
	double xi,muVar,xVar,yVar,eV,Ec,L,T,input,alphaOne,alphaTwo;


	srand48(time(0));

	input = atof(argv[1]);
//	N = 30;
	N = 100;
	muVar = 0;
//	muVar = 1e-5;
	
//	eV = .05;
	eV = 0;
	Ec = 16000;
//	Ec = 1.6e-5;
//	Ec = 1;
//	T = 1;
	alphaOne = 1; // technically combined with density of states
	alphaTwo = 1e7; // technically combined with e^2 and epsilon
	T = input;
//	nParticles = input;
	nParticles = .5*N*N;
//	nParticles = 1;
	L = 1e-8;	
//	tSteps = 1000000; //for statistically accurate runs
//	tSteps = 100000; //for potential runs
//	tSteps = 100; // for seeing the fields
	tSteps = 1;
//	relax = 1;
	relax = 0; 
	
	REAL *reducedProb,*particles,*probabilities,*potentials,*substrate,*hereP,*hereProb,*herePot,*hereS,*boxR,*hereBoxR,*hereXDiff,*hereYDiff,*dosMatrix,*reducedSum,*g_itemp,*g_otemp,*g_temp,*hereDos;
	xi = L/sqrt(sqrt(2));
	xVar = 0;
	yVar = 0;

//	xi = 1; // xi/a	
	clock_t begin = clock();
	

	threads=MAXT;
	blocks=N*N/threads+(N*N%threads==0?0:1);

        cudaMalloc(&reducedProb,N*N*sizeof(REAL));
	cudaMalloc(&particles,N*N*sizeof(REAL));
	cudaMalloc(&probabilities,N*N*sizeof(REAL));
	cudaMalloc(&potentials,N*N*sizeof(REAL));
	cudaMalloc(&substrate,N*N*sizeof(REAL));
	cudaMalloc(&dosMatrix,N*N*sizeof(REAL));
        cudaMalloc(&reducedSum,N*N*sizeof(REAL));
        cudaMalloc(&g_itemp,512*sizeof(REAL));
        cudaMalloc(&g_otemp,512*sizeof(REAL));
	cudaMalloc(&g_temp,512*sizeof(REAL));
	cudaMalloc(&boxR,N*N*N*N*sizeof(REAL));
	

	herePot =  new REAL[N*N];
        herePot = C_random(N,0,herePot); 
	hereProb = new REAL[N*N];	
	hereProb = C_random(N,0,hereProb);
	hereP = new REAL[N*N];
        hereDos = new REAL[N*N];
//	hereP = C_clump(N,nParticles,hereP);//test relaxation
	hereP = C_spread(N,nParticles,hereP); //test general potential
//	hereP = C_random(N,nParticles,hereP);	
//        hereP = C_more(N,nParticles,hereP);
	hereXDiff = new REAL[N*N];
	hereYDiff = new REAL[N*N];
	hereXDiff = createDiff(hereXDiff, xVar, N);
	hereYDiff = createDiff(hereYDiff, yVar, N);

	hereS = new REAL[N*N];
	hereS = createSub(hereS,muVar,N);
	hereBoxR = new REAL[N*N*N*N];
	hereBoxR = createR(hereBoxR,hereXDiff,hereYDiff,N,L,xi);

//	showMove(hereBoxR,N);
//	showMove(hereS,N);
	cudaMemcpy(potentials,herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
	cudaMemcpy(substrate,hereS,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
	cudaMemcpy(boxR,hereBoxR,N*N*N*N*sizeof(REAL),cudaMemcpyHostToDevice);
	cudaMemcpy(particles,hereP,N*N*sizeof(REAL),cudaMemcpyHostToDevice);

//	printGPU(particles,N);

//system is run but results arent output for the relaxation phase
        if (relax == 1) {
		glatzRelax(threads, blocks, L, N, potentials,substrate, particles, reducedSum,g_itemp, g_otemp,boxR,g_temp,dosMatrix);
	}

	
	
//find the DoS

//      dosMatrix = dosFind(hereP, hereS,herePot,dosMatrix, particles,potentials,boxR, N, L, threads, blocks);  
  //    showMove(dosMatrix,N);  



	for(t = 0; t < tSteps ; t++) {
		countThese = 1;
		findJump(hereP,hereProb,herePot,particles,probabilities,potentials,substrate,reducedProb, N, xi, threads, blocks,eV,Ec,L,T,boxR,alphaOne,alphaTwo);
	}

	
/*
        cudaMemcpy(hereP,particles,N*N*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        char    str1[256];
        sprintf(str1, "particles.txt");
        fp1 = fopen(str1, "w");
	for (int k = 0; k < N*N ; k++){
                fprintf(fp1, "%lf ",hereP[k]);
        }
//cleanup
	fclose(fp1);

*/

        cudaMemcpy(hereDos,dosMatrix,N*N*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp2;
        char    str2[256];
        sprintf(str2, "dosMatrix.txt");
        fp2 = fopen(str2, "w");
        for (int k = 0; k < N*N ; k++){
                fprintf(fp2, "%lf ",hereDos[k]);
        }
//cleanup
        fclose(fp2);

	
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
	cudaFree(reducedSum);
	cudaFree(g_itemp);
	cudaFree(g_otemp);
	cudaFree(g_temp);
	cudaFree(dosMatrix);

	
	clock_t end = clock();
//  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

//cout<<currentCount<<endl;
//cout<<"this took "<<elapsed_secs<<" seconds"<<endl;
}
