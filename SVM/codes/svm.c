#include <stdio.h>
#include <stdlib.h>
#define DIM 2

typedef struct data{
	double x[DIM];
	double y;
}data;

typedef struct svm{
	double *w;
	double b;
}svm;

double dot(double *x,double* y){
	double ret=0.0;
	for(int i=0;i<DIM;i++){
		ret+=x[i]*y[i];
	}
	return ret;
}
double* sub(double *x,double* y){
        double* ret=(double*)malloc(sizeof(double)*DIM);
        for(int i=0;i<DIM;i++){
                ret[i]=x[i]-y[i];
        }       
        return ret;
}       

double* smul(double x,double* y){
        double* ret=(double*)malloc(sizeof(double)*DIM);
        for(int i=0;i<DIM;i++){
                ret[i]=x*y[i];
        }       
        return ret;
}       


svm FindHyperplane(data* data,int size){
	svm ret;
	ret.w=(double*)malloc(sizeof(double)*DIM);
	for(int i=0;i<DIM;i++) ret.w[i]=0.0;
	ret.b=0.0;
	double h=0.001; //Learing Rate
	int iter=10000;
	for(int i=0;i<iter;i++){
		for(int j=0;j<size;j++){
			if(data[j].y*(dot(ret.w,data[j].x)+ret.b)<1.0){
			        double* w1=smul(1+h,ret.w),*w2=smul(-data[j].y*h,data[j].x),*w3=sub(w1,w2);
				free(w1);
				free(w2);
				free(ret.w);
				ret.w=w3;
				ret.b=ret.b+h*data[j].y;
			}
		}
	}
	return ret;
}
 
