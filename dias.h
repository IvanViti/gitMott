#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DOUBLE
 #define REAL double
 #define MAXT 256
#else
 #define REAL float
 #define MAXT 512
#endif



using namespace std;

class parameters {
public:
        double xi,muVar,xVar,yVar,eV,Ec,L,T,alphaOne,alphaTwo;
        int N,tSteps,nParticles,relax,grabJ;


        char boxName[256];
        char lineName[256];
        char timeName[256];

};


class vectors {
public:
        REAL *reducedProb,*particles,*probabilities,*potentials,*substrate,*hereP,*hereProb,*herePot,*hereS,*boxR,*hereBoxR,*hereXDiff,*hereYDiff,*Ematrix,*jumpRecord,*tempDos,*tempPar,*tempPot,*invertedDos,*watcher,*aMatrix,*timeRun;
        REAL *rangeMatrix,*extraArray,*sumArray,*hereSum;
	REAL results[5];
};


