#include <iostream>
#include <hdf5.h> // viene de Crossvalidation.CPP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <fstream>// VIENE DE READ INPUT.CPP
#include <iosfwd> // VIENE DE READ INPUT.CPP

#include <unistd.h>// VIENE DE TOOL.CPP

//#include <sys/time.h>
//#include "shared_var.h"
//#include <shared_var.h>
#include <limits.h> // VIENE DE READDIST.CPP
//#define MPI_SUCCESS          0

//#define DLEN_ 		     9
//typedef int MPI_Comm;


//#include "src/shared_var.h"
//#include <shared_var.h>

#define MPI_SUCCESS 0
#define MPI_CHAR ((MPI_Datatype)0x4c000101)

//#define DLEN_ 9

using namespace std;

typedef int MPI_Comm;
typedef int MPI_Datatype;

/* al punto h*/
//-------------------------------------------------
// VARAIBLES GLOBALES
//-------------------------------------------------
double d_one=1.0, d_zero=0.0, d_negone=-1.0;
int DLEN_=9, i_negone=-1, i_zero=0, i_one=1, i_two=2, i_four=4;
int m,n,t, b, m_plus, t_plus;
int lld_C, Cb, Cd;
long Cr, Cc;
int size, *ds, *pst, ICTXT2D, iam;
int nt, mi,dh5, Ccop;
char *Sd, *pd;
char *fDn, *fXn, *Ts;
double l, e;

//-------------------------------------------------
// Lo que estaba en el archivo shared_var.h
//-------------------------------------------------
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
void process_mem_usage(double& , double& , double& , double& );
void printdense ( int , int , double *, char * );
int Csu (int* , double* , int* , double* , double* ) ;
int Csu5 ( int* , double* , int*, double* , double*  ) ;
int Cu ( int * , double * , double ) ;
int Asu (double* , int* , int* , double* , int* , double* , double );
int Asu5 ( double* , int* , int* , double* , int* , double* , double  );
double CZt(double *, int * );
double Cld ( double *, int *  ) ;
int ri(char* ) ;
int cv(double* , int* );
int cv5(double * , int *) ;

/*
extern double d_one, d_zero, d_negone;
extern int DLEN_, i_negone, i_zero, i_one, i_two, i_four;
extern int m,n,t, b, m_plus;
extern int lld_C, Cb, Cd, t_plus;
extern long Cr, Cc;
extern int size, *ds, * pst, ICTXT2D, iam;
extern int nt, mi,dh5, Ccop;
extern char *Sd, *pd;
extern char *fDn, *fXn, *Ts;
*/
//---------------------------------------------------
//FUNCIONES DE CROSSVALIDATION PARA BLACS
//---------------------------------------------------
extern "C" {
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    void dgsum2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rdest, int *cdest);
    void dgebs2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda);
    void dgebr2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc);
    void dgesd2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgerv2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
}

// FUNCIONES BLACS VIENE DE READHDF5.CPP

extern "C" {
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_ ( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_ ( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_ ( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    H5_DLL hid_t H5Pcreate ( hid_t cls_id );
}

#define MPI_INFO_NULL ((MPI_Info)0x1c000000)


// FUNCIONES DE BLAS DESDE READDIST.CPP


extern "C" {
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
}




//Funciones de BLAS, LAPACK y SCALAPACK

extern "C" {
    void blacs_pinfo_ ( int *mypnum, int *nprocs );
    void blacs_setup_ ( int *mypnum, int *nprocs );
    void blacs_get_ ( int *ConTxt, int *what, int *val );
    void blacs_gridinit_ ( int *ConTxt, char *order, int *nprow, int *npcol );
    void blacs_gridexit_ ( int *ConTxt );
    void blacs_pcoord_ ( int *ConTxt, int *nodenum, int *prow, int *pcol );
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void igebs2d_(int *ConTxt, char *scope, char *top, int *m, int *n, int *A, int *lda);
    void igebr2d_(int *ConTxt, char *scope, char *top, int *m, int *n, int *A, int *lda, int *rsrc, int *csrc);
    void dgebs2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda);
    void dgebr2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc);
    void dgsum2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rdest, int *cdest);
    void dgesd2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgerv2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
    void pdcopy_ ( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pddot_ ( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    double pdlansy_ ( char *norm, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, double *work );
    void pdlacpy_ (char *uplo, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb);
    void pdpotrf_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pdpotri_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info);
    void pdnrm2_( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void dsyrk(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *beta, double *c, const int *ldc);
    void dgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);
    double  dnrm2(const int *n, const double *x, const int *incx);
    void    dcopy(const int *n, const double *x, const int *incx, double *y, const int *incy);
    double  ddot (const int *n, const double *x, const int *incx, const double *y, const int *incy);
    void dpotrf_( const char* uplo, const int* n, double* a, const int* lda, int* info );
    void dpotrs_( const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info );
    void dpotri_( const char* uplo, const int* n, double* a, const int* lda, int* info );
    int MPI_Init ( int *, char *** );
    int MPI_Dims_create ( int, int, int * );
    int MPI_Finalize ( void );
    int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm );
}

int main ( int argc, char **argv ) {
    int info;
    info = MPI_Init ( NULL, NULL);


    // Para iniciar MPI
    //info = MPI_Init ( &argc, &argv );
    if ( info != MPI_SUCCESS ) {
        printf ( "Error in MPI initialisation: %d",info );
        return info;
    }

    int i,j,pcol, co, bv;
    double *mc, *ytot, *RHS, *rn, *ran, *mai, *ets, *mcc;
    double si, dot, tp, ccrit, llh,pll,ull;
    int *DESCC, *DESCYTOT, *DESCRHS, *DESCAI, *DESCCCOPY;
    double c0, c1, c2, c3, c4;
    struct timeval tz0,tz1, tz2,tz3;
    double vm_usage, resident_set, cpu_sys, cpu_user;
    double *work, normC,norm1C, norminv, norm1inv, Cmax, colmax;

    //Descriptor para C
    DESCC= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCC==NULL ) {
        printf ( "unable to allocate memory for descriptor for C\n" );
        return -1;
    }

    //Descriptor para Y Total
    DESCYTOT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCYTOT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Ytot\n" );
        return -1;
    }

    //Descriptor para RHS
    DESCRHS= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCRHS==NULL ) {
        printf ( "unable to allocate memory for descriptor for RHS\n" );
        return -1;
    }


    //Descriptor para AI
    DESCAI= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCAI==NULL ) {
        printf ( "unable to allocate memory for descriptor for AI\n" );
        return -1;
    }

    //Memoria para coordinar la posicion del procesador
    pst= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( pst==NULL ) {
        printf ( "unable to allocate memory for processor position coordinate\n" );
        return EXIT_FAILURE;
    }

    //Memoria para coordinar el tamano de la grilla
    ds= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( ds==NULL ) {
        printf ( "unable to allocate memory for grid dimensions coordinate\n" );
        return EXIT_FAILURE;
    }


    //Para inicializar el BLAC
    blacs_pinfo_ ( &iam,&size );
    blacs_setup_ ( &iam,&size );

    if ( iam ==-1 ) {
        printf ( "Error in initialisation of proces grid" );
        return -1;
    }

    //Para crear la grilla MPI
    info=MPI_Dims_create ( size, 2, ds );
    if ( info != MPI_SUCCESS ) {
        printf ( "Error in MPI creation of dimensions: %d",info );
        return info;
    }


    blacs_get_ ( &i_negone,&i_zero,&ICTXT2D );

    blacs_gridinit_ ( &ICTXT2D,"R",ds, ds+1 );

    blacs_pcoord_ ( &ICTXT2D,&iam,pst, pst+1 );
    if ( *pst ==-1 ) {
        printf ( "Error in proces grid" );
        return -1;
    }

/*
    if ( argc !=2 ) {
        if ( * ( pst+1 ) ==0 && *pst==0 ) {

            printf ( "The correct use of DAIRRy-BLUP is:\n ./DAIRRy-BLUP <input_file>\n" );
            return -1;
        }
        else
            return -1;
    }

    info=ri(*++argv);
*/

    // Para leer el archivo de configuracion
    info=ri("defaultinput.txt");
    if(info!=0) {
        printf("Something went wrong when reading input file for processor %d\n",iam);
        return -1;
    }

    blacs_barrier_(&ICTXT2D,"ALL");
    if ( * ( pst+1 ) ==0 && *pst==0 )
        printf("Reading of input-file succesful\n");

    m_plus=m+1;
    t_plus=t+1;

    Cd=m+t;
    pcol= * ( pst+1 );
    Cb= Cd%b==0 ? Cd/b : Cd/b +1;
    Cr= ( Cb - *pst ) % *ds == 0 ? ( Cb- *pst ) / *ds : ( Cb- *pst ) / *ds +1;
    Cr= Cr<1? 1 : Cr;
    Cc= ( Cb - pcol ) % * ( ds+1 ) == 0 ? ( Cb- pcol ) / * ( ds+1 ) : ( Cb- pcol ) / * ( ds+1 ) +1;
    Cc=Cc<1? 1 : Cc;
    lld_C=Cr*b;

    if(Ccop)
    {
        DESCCCOPY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCCCOPY==NULL )
        {
            printf ( "unable to allocate memory for descriptor for copy of C\n" );
            return -1;
        }

        descinit_ ( DESCCCOPY, &Cd, &Cd, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
        if ( info!=0 )
        {
            printf ( "Descriptor of copy of matrix C returns info: %d\n",info );
            return info;
        }

        mcc= ( double* ) calloc ( Cr * (long) b * Cc * (long) b,sizeof ( double ) );

        if ( mcc==NULL )
        {
            printf ( "unable to allocate memory for copy of Matrix C (required: %dl bytes)\n", Cr * (long) b * Cc * (long) b );
            return EXIT_FAILURE;
        }
    }

    descinit_ ( DESCC, &Cd, &Cd, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 )
    {
        printf ( "Descriptor of matrix C returns info: %d\n",info );
        return info;
    }

    descinit_ ( DESCYTOT, &Cd, &i_one, &b, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 )
    {
        printf ( "Descriptor of response matrix returns info: %d\n",info );
        return info;
    }

    descinit_ ( DESCRHS, &Cd, &i_one, &b, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 )
    {
        printf ( "Descriptor of RHS matrix returns info: %d\n",info );
        return info;
    }

    descinit_ ( DESCAI, &i_two, &i_two, &i_two, &i_two, &i_zero, &i_zero, &ICTXT2D, &i_two, &info );
    if ( info!=0 )
    {
        printf ( "Descriptor of AI matrix returns info: %d\n",info );
        return info;
    }

    ccrit=0;
    co=0;

    if ( * ( pst+1 ) ==0 && *pst==0 )
    {
        printf("\nA linear mixed model with %d observations, %d random effects and %d fixed effects\n", n,m,t);
        printf("was analyzed using %d (%d x %d) processors\n",size,*ds,*(ds+1));
        gettimeofday ( &tz2,NULL );
        c2= tz2.tv_sec*1000000 + ( tz2.tv_usec );
        ets=(double *) calloc(Cb * b, sizeof(double));
        if(ets==NULL)
        {
            printf("unable to allocate memory for solution matrix\n");
            return EXIT_FAILURE;
        }
    }

    mc= ( double* ) calloc ( Cr * (long) b * Cc * (long) b,sizeof ( double ) );
    if ( mc==NULL )
    {
        printf ( "unable to allocate memory for Matrix C  (required: %dl bytes)\n", Cr * (long) b * Cc * (long) b * sizeof ( double ) );
        return EXIT_FAILURE;
    }

    ytot = ( double* ) calloc ( Cr * b,sizeof ( double ) );
    if ( ytot==NULL )
    {
        printf ( "unable to allocate memory for Matrix Y (required: %d bytes)\n", Cr * b*sizeof ( double )  );
        return EXIT_FAILURE;
    }

    RHS = ( double* ) calloc ( Cr * b,sizeof ( double ) );
    if ( RHS==NULL )
    {
        printf ( "unable to allocate memory for RHS (required: %d bytes)\n", Cr * b * sizeof ( double )  );
        return EXIT_FAILURE;
    }

    mai = ( double* ) calloc ( 2*2,sizeof ( double ) );
    if ( RHS==NULL )
    {
        printf ( "unable to allocate memory for AI matrix (required: %d bytes)\n",2*2*sizeof ( double ) );
        return EXIT_FAILURE;
    }

    rn= ( double * ) calloc ( 1,sizeof ( double ) );
    if ( rn==NULL )
    {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }

    ran= ( double * ) calloc ( 1,sizeof ( double ) );
    if ( ran==NULL )
    {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }

    while ( fabs ( ccrit ) >e || fabs(ull/llh) > e || co<2 )
    {
        ++co;

        if (co > mi)
        {
            if ( * ( pst+1 ) ==0 && *pst==0 )
            {
                printf("maximum number of iterations reached, AI-REML has not converged\n");
            }
            break;
        }

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            printf ( "\nParallel results: loop %d\n",co );
            printf ( "=========================\n" );
            gettimeofday ( &tz3,NULL );
            c3= tz3.tv_sec*1000000 + ( tz3.tv_usec );
        }

        //printf("esto es un demo...... aca estoy");

        cout<<"Valor de co="<<co<<endl;
        if (co > 1 )
        {
            free ( ytot );
            //free ( Cmat );
            free ( mai );

            mc= ( double* ) calloc ( Cr*b*Cc*b,sizeof ( double ) );
            if ( mc==NULL )
            {
                printf ( "unable to allocate memory for Matrix C (required: %d bytes)\n", Cr*b*Cc*b*sizeof ( double ) );
                return EXIT_FAILURE;
            }
            ytot = ( double* ) calloc ( Cr * b,sizeof ( double ) );

            if ( ytot==NULL )
            {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }

            mai = ( double* ) calloc ( 2*2,sizeof ( double ) );

            if ( RHS==NULL )
            {
                printf ( "unable to allocate memory for AI matrix\n" );
                return EXIT_FAILURE;
            }

            blacs_barrier_ ( &ICTXT2D,"A" );

            if ( * ( pst+1 ) ==0 && *pst==0 )
            {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time allocation of memory:		%10.3f s\n", ( c1 - c3 ) /1000000.0 );
            }

            if (Ccop)
            {

                Cu(DESCCCOPY,mcc,l/ ( 1+ccrit ) - l);
                pdlacpy_ ( "U", &Cd, &Cd, mcc, &i_one, &i_one, DESCCCOPY, mc, &i_one, &i_one, DESCC);
                pdcopy_ ( &Cd, RHS,&i_one,&i_one,DESCRHS,&i_one,ytot,&i_one,&i_one,DESCYTOT,&i_one );
                if ( * ( pst+1 ) ==0 && *pst==0 )
                {
                    gettimeofday ( &tz1,NULL );
                    c0= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                    printf ( "\t elapsed wall time copy of Y and C:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                }
                l=l/ ( 1+ccrit );
            }
            else
            {
                free ( rn );
                rn= ( double * ) calloc ( 1,sizeof ( double ) );
                if ( rn==NULL )
                {
                    printf ( "unable to allocate memory for norm\n" );
                    return EXIT_FAILURE;
                }
                cout<<"valor de l="<<l<<endl;
                l=l/ ( 1+ccrit );



                if(dh5)
                    info = Csu5 ( DESCC, mc, DESCYTOT, ytot, rn);
                else
                    info = Csu ( DESCC, mc, DESCYTOT, ytot, rn);

                if ( info!=0 )
                {
                    printf ( "Something went wrong with set-up of matrix C, error nr: %d\n",info );
                    return info;
                }

                if ( * ( pst+1 ) ==0 && *pst==0 )
                {
                    gettimeofday ( &tz0,NULL );
                    c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                    printf ( "\t elapsed wall time set-up of C and Y:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                }
            }
        }
        else
        {

            if ( * ( pst+1 ) ==0 && *pst==0 )
            {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time allocation of memory:		%10.3f s\n", ( c1 - c3 ) /1000000.0 );
            }


            cout<<"valor de dh5="<<dh5<<endl;
            cout<<"valor de l="<<l<<endl;

            cout<<DESCC<<" "<<mc<<" "<<DESCYTOT<<" "<<ytot<<" "<<rn<<endl;

            if(dh5)
                info = Csu5 ( DESCC, mc, DESCYTOT, ytot, rn);
            else
                info = Csu ( DESCC, mc, DESCYTOT, ytot, rn);

            printf("esto es un demo...... aca estoy");

            if ( info!=0 )
            {
                printf ( "Something went wrong with set-up of matrix C, error nr: %d\n",info );
                return info;
            }

            if ( * ( pst+1 ) ==0 && *pst==0 )
            {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time set-up of C and Y:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            }

            pdcopy_ ( &Cd, ytot,&i_one,&i_one,DESCYTOT,&i_one,RHS,&i_one,&i_one,DESCRHS,&i_one );
            if(Ccop)
                pdlacpy_ ( "U", &Cd, &Cd, mc, &i_one, &i_one, DESCC, mcc, &i_one, &i_one, DESCCCOPY);

            if ( * ( pst+1 ) ==0 && *pst==0 ) {
                gettimeofday ( &tz1,NULL );
                c4= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time copy of Y (and C):			%10.3f s\n", ( c4 - c0 ) /1000000.0 );
            }
        }

        work= ( double * ) calloc ( 2*b*(Cc+Cr),sizeof ( double ) );

        if ( work==NULL )
        {
            printf ( "unable to allocate memory for work (norm)\n" );
            return EXIT_FAILURE;
        }

        normC=pdlansy_ ( "F","U",&Cd,mc,&i_one,&i_one,DESCC,work );
        norm1C=pdlansy_ ( "1","U",&Cd,mc,&i_one,&i_one,DESCC,work );
        Cmax=pdlansy_ ( "M","U",&Cd,mc,&i_one, &i_one, DESCC,work );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c1= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time norm and max of C:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            printf ( "The new parallel lambda is: %15.10g\n",l );
            printf ( "norm of y-vector is: %g\n",*rn );
        }

        pdpotrf_ ( "U",&Cd,mc,&i_one, &i_one,DESCC,&info );
        if ( info!=0 )
        {
            printf ( "Parallel Cholesky decomposition of C was unsuccesful 600, error returned: %d\n",info );
            return -1;
        }

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time Cholesky decomposition of C:		%10.3f s\n", ( c1 - c0 ) /1000000.0 );
        }

        pdpotrs_ ( "U",&Cd,&i_one,mc,&i_one,&i_one,DESCC,ytot,&i_one,&i_one,DESCYTOT,&info );

        if ( info!=0 )
            printf ( "Parallel Cholesky solution was unsuccesful, error returned: %d\n",info );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time estimation of effects:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
        }

        pddot_ ( &Cd,&dot,RHS,&i_one,&i_one,DESCRHS,&i_one,ytot,&i_one,&i_one,DESCYTOT,&i_one );

        si= ( *rn - dot ) / ( n-t );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            dgebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&si,&i_one );
        }
        else
            dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &si,&i_one,&i_zero,&i_zero );

        llh=Cld(mc,DESCC);
        dgsum2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&llh,&i_one,&i_negone,&i_negone );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time calculation and sending of sigma and log(det(C)):	%10.3f s\n", ( c1 - c0 ) /1000000.0 );
        }

        if(dh5)
            info = Asu5 ( mai, DESCAI,DESCYTOT, ytot, DESCC, mc, si) ;
        else
            info = Asu ( mai, DESCAI,DESCYTOT, ytot, DESCC, mc, si) ;

        if ( info!=0 )
        {
            printf ( "Something went wrong with set-up of AI-matrix, error nr: %d\n",info );
            return EXIT_FAILURE;
        }

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set up of AI matrix:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
        }

        pdpotri_ ( "U",&Cd,mc,&i_one,&i_one,DESCC,&info );
        if ( info!=0 )
            printf ( "Parallel Cholesky inverse was unsuccesful, error returned: %d\n",info );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time inverse of C:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
        }

        norminv=pdlansy_ ( "F","U",&Cd,mc,&i_one,&i_one,DESCC,work );
        norm1inv=pdlansy_ ( "1","U",&Cd,mc,&i_one,&i_one,DESCC,work );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set norm of inverse of C:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
        }

        free ( work );
        tp=CZt ( mc,DESCC );
        free ( mc );

        dgsum2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&tp,&i_one,&i_negone,&i_negone );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time trace of inverse of C:		%10.3f s\n", ( c1 - c0 ) /1000000.0 );
        }

        pdnrm2_ ( &m,ran,ytot,&t_plus,&i_one,DESCYTOT,&i_one );

        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set norm of estimation of u:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            double *score;
            printf ( "dot product = %15.10g \n",dot );
            printf ( "parallel sigma = %15.10g\n",si );
            printf ( "The trace of CZZ is: %15.10g \n",tp );
            printf ( "The norm of the estimation of u is: %g \n",*ran );
            llh += (m * log(1/l) + (n-t) * log(si) + n-t)/2;
            llh *= -1.0;

            score= ( double * ) calloc ( 2,sizeof ( double ) );
            if ( score==NULL )
            {
                printf ( "unable to allocate memory for score function\n" );
                return EXIT_FAILURE;
            }
            * ( score+1 ) = - ( m-tp*l- *ran * *ran * l / si ) * l / 2;

            printf ( "The score function is: %g\n",* ( score+1 ) );
            printdense ( 2,2, mai, "AI_par.txt" );
            bv=0;
            if (fabs(*(score+1))< e * e)
            {
                printf("Score function too close to zero to go further, solution may not have converged\n ");
                bv=1;
                igebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&bv,&i_one );
                break;
            }

            igebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&bv,&i_one );

            if (co==1)
            {
                tp= *mai + *(mai+3);
                pll=llh;
                printf ( "The loglikelihood is: %g\n",llh );
            }

            else
            {
                ull=llh-pll;
                pll=llh;
                printf ( "The update for the loglikelihood is: %g\n",ull );
                printf ( "The new loglikelihood is: %g\n",llh );
            }

            dgebs2d_ (&ICTXT2D,"ALL","1-tree",&i_one,&i_one,&ull,&i_one);
            dpotrf_ ( "U", &i_two, mai, &i_two, &info );

            if ( info!=0 )
            {
                printf ( "Cholesky decomposition of AI matrix was unsuccesful, error returned: %d\n",info );
                return -1;
            }
            dpotrs_ ( "U",&i_two,&i_one,mai,&i_two,score,&i_two,&info );

            if ( info!=0 )
            {
                printf ( "Parallel solution for AI matrix was unsuccesful, error returned: %d\n",info );
                return -1;
            }

            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time update for lambda:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            printf ( "The update for gamma is: %g \n", *(score+1) );

            while (*(score+1)+1/l <0)
            {
                *(score+1)=*(score+1)/2;
                printf("Half a step is used to avoid negative gamma\n");
            }

            ccrit=l * * ( score+1 );
            dgebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&ccrit,&i_one );
            free ( score );
            printf ( "The eventual relative update for gamma is: %g \n", ccrit );

        }
        else
        {
            igebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &bv,&i_one,&i_zero,&i_zero );
            if (bv >0)
            {
                break;
            }
            dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &ull,&i_one,&i_zero,&i_zero );
            dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &ccrit,&i_one,&i_zero,&i_zero );
        }
        if ( * ( pst+1 ) ==0 && *pst==0 )
        {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time sending and receiving update lambda:	%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            printf ( "\t elapsed wall time iteration loop %d:			%10.3f s\n", co, ( c0 - c3 ) /1000000.0 );
        }

    }

    blacs_barrier_ ( &ICTXT2D,"A" );


    free ( mai );
    free ( RHS );
    free ( rn );
    free ( ran );
    free ( DESCC );
    free ( DESCAI );
    free ( DESCRHS );

    if (Ccop)
    {
      free ( mcc );
      free ( DESCCCOPY);
    }

    if (nt>0)
    {
        if(dh5)
            info=cv5(ytot,DESCYTOT);
        else
            info=cv( ytot, DESCYTOT);
        if ( info!=0 )
        {
            printf ( "Cross-validation was unsuccesful, error returned: %d\n",info );
            return -1;
        }
    }

    if ( * ( pst+1 ) ==0 )
    {
        for ( i=0,j=0; i<Cb; ++i,++j )
        {
            if ( j==*ds )
                j=0;

            if ( *pst==j )
            {
                dgesd2d_ ( &ICTXT2D,&b,&i_one,ytot+ i / *ds *b,&b,&i_zero,&i_zero );
            }
            if ( *pst==0 )
            {
                dgerv2d_ ( &ICTXT2D,&b,&i_one,ets+b*i,&b,&j,&i_zero );
            }
        }
    }

    blacs_barrier_ ( &ICTXT2D, "A" );

    if ( * ( pst+1 ) ==0 && *pst==0 )
    {
        gettimeofday ( &tz0,NULL );
        c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
        printf ( "\n\tOverall results:\n");
        printf ( "\t================\n");
        printf ( "\tThe maximum element in C is:          %10.5f\n", Cmax );
        printf ( "\tThe Frobenius norm of C is:           %15.10e\n", normC );
        printf ( "\tThe 1-norm of C is:                   %15.10e\n", norm1C );
        printf ( "\tThe Frobenius norm of Cinv is:        %15.10e\n", norminv );
        printf ( "\tThe 1-norm of Cinv is:                %15.10e\n", norm1inv );
        printf ( "\tThe Frobenius condition number is:    %15.10e\n", norminv*normC );
        printf ( "\tThe condition number (1-norm) is:     %15.10e\n", norm1inv*norm1C );
        printf ( "\tThe accuracy is:                      %15.10e\n", norminv*normC*Cmax/pow ( 2,53 ) );
        printf ( "\tThe ultimate lambda is:               %15.10g\n",l );
        printf ( "\tThe ultimate sigma is:                %15.10g\n", si );

        printf ( "\telapsed total wall time:              %10.3f s\n", ( c0 - c2 ) /1000000.0 );

        printf ( "\tProcessor: %d \n\t ========================\n", iam );
        printf ( "\tVirtual memory used:                  %10.0f kb\n", vm_usage );
        printf ( "\tResident set size:                    %10.0f kb\n", resident_set );
        printf ( "\tCPU time (user):                      %10.3f s\n", cpu_user );
        printf ( "\tCPU time (system):                    %10.3f s\n", cpu_sys );
        printdense(Cd,1,ets,"estimates.txt" );
        free(ets);
    }

    free ( DESCYTOT );
    free ( pst ),free ( ds );
    free ( fDn );
    free ( fXn );
    free ( Ts );
    free ( ytot );
    free (Sd);
    free (pd);

    blacs_barrier_ ( &ICTXT2D, "A" );
    blacs_gridexit_ ( &ICTXT2D );
//
    MPI_Finalize();
    return 0;
}


//--------------------------------------------------------
// CROSSVALIDATION.CPP
//--------------------------------------------------------



int cv(double * ets, int *dets) {
    FILE *fT;
    int ni, i,j, info;
    int *DT, *DE;
    double *Tb, v, *Ebr;
    int nTb, ns, pTb, sc, lld_T,pcol,ct, lld_E;

    DT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DE= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DE==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    ns= nt % ( b * * ( ds+1 ) ) ==0 ?  nt / ( b * * ( ds+1 ) ) : ( nt / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    nTb= m%b==0 ? m/b : m/b +1;
    pTb= ( nTb - *pst ) % *ds == 0 ? ( nTb- *pst ) / *ds : ( nTb- *pst ) / *ds +1;
    pTb= pTb <1? 1:pTb;
    lld_T=pTb*b;
    lld_E=ns*b* *(ds+1);

    descinit_ ( DT, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    descinit_ ( DE, &lld_E, &i_one, &lld_E, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_E, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of EBV returns info: %d\n",info );
        return info;
    }

    Tb= ( double* ) calloc ( pTb*b*b, sizeof ( double ) );
    if ( Tb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }

    Ebr = ( double* ) calloc ( lld_E,sizeof ( double ) );
    if ( Ebr==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    fT=fopen ( Ts,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file 955\n" );
        return -1;
    }

    for ( ni=0; ni<ns; ++ni ) {
        if ( ni==ns-1 ) {

            free ( Tb );
            Tb= ( double* ) calloc ( pTb*b*b, sizeof ( double ) );
            if ( Tb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
                return -1;
            }
        }
        if ( ( nTb-1 ) % *ds == *pst && m%b !=0 ) {
            for ( i=0; i<b; ++i ) {
                info=fseek ( fT, ( long ) ( ( ( ni * * ( ds+1 ) * b + pcol * b + i ) * ( m+1 ) + b * *pst ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading file" );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( &v,sizeof ( double ),1,fT );
                else
                    info=fseek ( fT,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pTb-1; ++j ) {
                    fread ( Tb + i*pTb*b + j*b,sizeof ( double ),b,fT );
                    info=fseek ( fT, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading file" );
                        return -1;
                    }
                }
                fread ( Tb + i*pTb*b + j*b,sizeof ( double ),m%b,fT );
            }
        } else {
            for ( i=0; i<b; ++i ) {
                info=fseek ( fT, ( long ) ( ( ( ni * * ( ds+1 ) * b + pcol * b + i ) * ( m+1 ) + b * *pst ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading file" );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( &v,sizeof ( double ),1,fT );
                else
                    info=fseek ( fT,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pTb; ++j ) {
                    fread ( Tb + i*pTb*b + j*b,sizeof ( double ),b,fT );
                    info=fseek ( fT, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading file" );
                        return -1;
                    }
                }
            }
        }
        blacs_barrier_ ( &ICTXT2D,"A" );

        ct=1 + ni * sc;

        pdgemm_ ( "T","N",&sc,&i_one,&m,&d_one,Tb,&i_one,&i_one,DT,ets,&t_plus,&i_one,dets,&d_one,Ebr,&ct,&i_one,DE);

    }

    fclose(fT);
    if(*pst==0 && *(pst+1)==0) {
        printdense(nt,1,Ebr,"EBV.txt" );
    }
    blacs_barrier_(&ICTXT2D, "A" );
    free(Ebr);
    free(Tb);
    free(DE);
    free(DT);

    return info;

}

int cv5(double * ets, int *dets) {
    int ni, i,j, info;
    int *DT, *DE;
    double *tb, v, *ebr;
    int ntb, ns, pTb, sc, lld_T,pcol,curtest, lld_E;

    hid_t       fid, dgi, sgi;
    hid_t	pid, mg;
    herr_t	status;
    hsize_t	dm[2], os[2],co[2], st[2],bl[2];

    int mpinfo  = MPI_INFO_NULL;

    pid = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( pid, MPI_COMM_WORLD, mpinfo );

    fid = H5Fopen ( fDn, H5F_ACC_RDWR, pid );
    dgi = H5Dopen ( fid, Ts, H5P_DEFAULT );
    sgi=H5Dget_space(dgi);

    DT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }

    DE= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DE==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    ns= nt % ( b * * ( ds+1 ) ) ==0 ?  nt / ( b * * ( ds+1 ) ) : ( nt / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    ntb= m%b==0 ? m/b : m/b +1;
    pTb= ( ntb - *pst ) % *ds == 0 ? ( ntb- *pst ) / *ds : ( ntb- *pst ) / *ds +1;
    pTb= pTb <1? 1:pTb;
    lld_T=pTb*b;
    lld_E=ns*b* *(ds+1);

    descinit_ ( DT, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    descinit_ ( DE, &lld_E, &i_one, &lld_E, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_E, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of EBV returns info: %d\n",info );
        return info;
    }

    tb= ( double* ) calloc ( pTb*b*b, sizeof ( double ) );
    if ( tb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    dm[0]=b;
    dm[1]=pTb*b;
    mg = H5Screate_simple(2,dm,NULL);

    ebr = ( double* ) calloc ( lld_E,sizeof ( double ) );
    if ( ebr==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    pid = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( pid, H5FD_MPIO_INDEPENDENT );

    for ( ni=0; ni<ns; ++ni ) {
        if(*pst >= ntb)
            goto CALC1;
        if ( ni==ns-1 ) {

            free ( tb );
            tb= ( double* ) calloc ( pTb*b*b, sizeof ( double ) );
            if ( tb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
                return -1;
            }
            if((pcol + 1 + (ns-1) * *(ds+1))*b <= nt)
                bl[0]=b;
            else if ((pcol + (ns-1) * *(ds+1))*b >= nt)
                bl[0]=0;
            else
                bl[0]=nt%b;
        }
        else {
            bl[0]=b;
        }
        if ( ( ntb-1 ) % *ds == *pst && m%b !=0 ) {
            os[0] = ni * *(ds+1) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pTb-1;
            st[0] = b * *(ds+1);
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if (status<0) {
                printf("selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni);
                return status;
            }
            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( mg, H5S_SELECT_SET, os, st, co, bl );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful, strip: %d\n",ni);
                return status;
            }

            os[0] = ni * *(ds+1) * b + pcol * b;
            os[1] = (ntb-1) * b;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * *(ds+1);
            st[1] = b * *ds;
            bl[1] = m%b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_OR, os, st, co, bl );
            if (status<0) {
                printf("selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni);
                return status;
            }

            os[0] = 0;
            os[1] = (pTb-1) * b;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( mg, H5S_SELECT_OR, os, st, co, bl );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful, strip: %d\n",ni);
                return status;
            }
        }
        else {
            os[0] = ni * *(ds+1) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pTb;
            st[0] = b * *(ds+1);
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if (status<0) {
                printf("selection of geno hyperslab in file was unsuccesful\n");
                return status;
            }

            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( mg, H5S_SELECT_SET, os, st, co, bl );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful\n");
                return status;
            }
        }
        status= H5Dread ( dgi,H5T_NATIVE_DOUBLE_g,mg,sgi,pid,tb );
        if (status<0) {
            printf("reading of geno hyperslab was unsuccesful\n");
            return status;
        }
CALC1:
        blacs_barrier_ ( &ICTXT2D,"A" );

        curtest=1 + ni * sc;

        pdgemm_ ( "T","N",&sc,&i_one,&m,&d_one,tb,&i_one,&i_one,DT,ets,&t_plus,&i_one,dets,&d_one,ebr,&curtest,&i_one,DE);

    }

    if(*pst==0 && *(pst+1)==0) {
        printdense(nt,1,ebr,"EBV.txt" );
    }
    blacs_barrier_(&ICTXT2D, "A" );
    free(ebr);
    free(tb);
    free(DE);
    free(DT);

    H5Dclose ( dgi );
    H5Sclose ( mg );
    H5Sclose ( sgi );
    H5Pclose ( pid );
    H5Fclose ( fid );

    return info;

}

//--------------------------------------------------
// READINPUT.CPP
//--------------------------------------------------

int ri(char * filename) {

    std::ifstream inputfile(filename);
    string line;

    bool obs_bool=false, SNP_bool=false, fixed_bool=false, fixedfile_bool=false, datafile_bool=false, lam_bool=false, phenopath_bool=false, testpath_bool=false;
    bool eps_bool=false, maxit_bool=false, blocksize_bool=false, testfile_bool=false, ntest_bool=false, h5_bool=false, genopath_bool=false;

    fDn=( char* ) calloc ( 100,sizeof ( char ) );
    fXn=( char* ) calloc ( 100,sizeof ( char ) );
    Ts=( char* ) calloc ( 100,sizeof ( char ) );
    Sd = ( char* ) calloc ( 100,sizeof ( char ) );
    pd = ( char* ) calloc ( 100,sizeof ( char ) );

    int l=100;
    b=64;
    int e=0.01;
    mi=20;
    nt=0;
    Ccop=0;


    while( std::getline (inputfile,line)) {
        if(line=="#Observations") {
            std::getline (inputfile,line);
            n=atoi(line.c_str());
            obs_bool=true;
        }
        else if (line=="#SNPs") {
            std::getline (inputfile,line);
            m=atoi(line.c_str());
            SNP_bool=true;
        }
        else if (line=="#FixedEffects") {
            std::getline (inputfile,line);
            t=atoi(line.c_str());
            fixed_bool=true;
        }
        else if (line=="#FileFixedEffects") {
            std::getline (inputfile,line);
            line.copy(fXn,100);
            fixedfile_bool=true;
        }
        else if (line=="#PathGeno") {
            std::getline (inputfile,line);
            line.copy(Sd,100);
            genopath_bool=true;
        }
        else if (line=="#PathPheno") {
            std::getline (inputfile,line);
            line.copy(pd,100);
            phenopath_bool=true;
        }
        else if (line=="#TestPath") {
            std::getline (inputfile,line);
            line.copy(Ts,100);
            testpath_bool=true;
        }
        else if (line=="#DataFile") {
            std::getline (inputfile,line);
            line.copy(fDn,100);
            datafile_bool=true;
        }
        else if (line=="#DataFileHDF5") {
            std::getline (inputfile,line);
            dh5=atoi(line.c_str());
            h5_bool=true;
        }
        else if (line=="#KeepCopyOfCMatrix") {
            std::getline (inputfile,line);
            Ccop=atoi(line.c_str());
        }
        else if (line=="#TestFile") {
            std::getline (inputfile,line);
            line.copy(Ts,100);
            testfile_bool=true;
        }
        else if(line=="#MaximumIterations") {
            std::getline (inputfile,line);
            mi=atoi(line.c_str());
            maxit_bool=true;
        }
        else if (line=="#Lambda") {
            std::getline (inputfile,line);
            l=atof(line.c_str());
            lam_bool=true;
        }
        else if (line=="#Epsilon") {
            std::getline (inputfile,line);
            e=atof(line.c_str());
            eps_bool=true;
        }
        else if (line=="#BlockSize") {
            std::getline (inputfile,line);
            b=atoi(line.c_str());
            blocksize_bool=true;
        }
        else if (line=="#TestSamples") {
            std::getline (inputfile,line);
            nt=atoi(line.c_str());
            ntest_bool=true;
        }
        else if (line[0]=='/' || line.size()==0) {}
        else {
            printf("Unknown parameter in inputfile, the following line was ignored: \n");
            printf("%s\n",line.c_str());
        }
    }
    if(obs_bool) {
        if(SNP_bool) {
            if(fixed_bool) {
                if(datafile_bool) {
                    if(fixedfile_bool) {
                        if(h5_bool) {
                            if(*pst==0 && *(pst+1)==0) {
                                printf("number of observations:   \t %d\n", n);
                                printf("number of SNP effects:    \t %d\n", m);
                                printf("number of fixed effects:  \t %d\n", t);
                                printf("filename of dataset:      \t %s\n", fDn);
                                printf("filename of fixed effects:\t %s\n", fXn);
                            }
                        }
                        else {
                            printf("ERROR: filetype of dataset was not in input file or not read correctly\n");
                            return -1;
                        }
                    }
                    else {
                        printf("ERROR: filename of fixed effects was not in input file or not read correctly\n");
                        return -1;
                    }
                }
                else {
                    printf("ERROR: filename of SNP effects was not in input file or not read correctly\n");
                    return -1;
                }
            }
            else {
                printf("ERROR: number of fixed effects was not in input file or not read correctly\n");
                return -1;
            }
        }
        else {
            printf("ERROR: number of SNP effects was not in input file or not read correctly\n");
            return -1;
        }
    }
    else {
        printf("ERROR: number of observations was not in input file or not read correctly\n");
        return -1;
    }
    if(*pst==0 && *(pst+1)==0) {
        if (dh5) {
            if (genopath_bool) {
                if(phenopath_bool) {
		    printf("Dataset file is an HDF5-file \n");
                    printf("path for genotypes in dataset:      \t %s\n", Sd);
                    printf("path for phenotypes in dataset:     \t %s\n", pd);
                }
                else {
                    printf("ERROR: path for phenotypes of dataset was not in input file or not read correctly\n");
                    return -1;
                }
            }
            else {
                printf("ERROR: path for genotypes of dataset was not in input file or not read correctly\n");
                return -1;
            }
        }
        else
	  printf("Dataset file is a binary file, with as first column the phenotypical score\n");
	if(Ccop)
	  printf("A copy of the coefficient matrix will be stored throughout the computations\n");
	else
	  printf("The coefficient matrix will be read in at the beginning of every iteration to save memory\n");
        if(blocksize_bool) {
            printf("Blocksize of %d was used to distribute matrices across processes\n", b);
        }
        else {
            printf("Default blocksize of %d was used to distribute matrices across processes\n", b);
        }
        if(lam_bool)
            printf("Start value of %g was used to estimate variance component lambda\n", l);
        else
            printf("Default start value of %g was used to estimate variance component lambda\n", l);
        if(eps_bool)
            printf("Convergence criterium of %g was used to estimate variance component lambda\n", e);
        else
            printf("Default convergence criterium of %g was used to estimate variance component lambda\n", e);
        if(maxit_bool)
            printf("Maximum number of REML iterations : %d\n", mi);
        else
            printf("Default maximum number of REML iterations : %d\n", mi);
        if(dh5==0 && testfile_bool) {
            printf("Cross-validation will be performed with test set in file: \t%s\n", Ts);
            if(ntest_bool)
                printf("Cross-validation will be performed on sample with size: \t%d\n", nt);
            else {
                printf("ERROR: Number of test samples is required when cross-validation is performed\n");
                return -1;
            }
        }
        else if (dh5 && testpath_bool) {
            printf("Cross-validation will be performed with test set in path: \t%s\n", Ts);
            if(ntest_bool)
                printf("Cross-validation will be performed on sample with size: \t%d\n", nt);
            else {
                printf("ERROR: Number of test samples is required when cross-validation is performed\n");
                return -1;
            }
        }

        else
            printf("No cross-validation is performed\n");
    }
    else {
        if(testfile_bool && !ntest_bool)
            return -1;
        if(dh5 && (!genopath_bool || !phenopath_bool))
            return -1;
    }
    return 0;
}

//----------------------------------------------
// TOOL.CPP
//-----------------------------------------

void printdense ( int m, int n, double *mat, char *filename ) {
    FILE *fd;
    fd = fopen ( filename,"w" );
    if ( fd==NULL )
        printf ( "error creating file" );
    int i,j;
    for ( i=0; i<m; ++i ) {
        for ( j=0; j<n; ++j ) {
            fprintf ( fd,"%12.8g\t",*(mat+i*n +j));
        }
    }
    fclose ( fd );
}


//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

void process_mem_usage(double& vm_usage, double& resident_set, double& cpu_user, double& cpu_sys)
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage     = 0.0;
    resident_set = 0.0;
    cpu_user     = 0.0;
    cpu_sys      = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat",ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize, utime, stime;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = vsize / 1024.0;
    resident_set = rss * page_size_kb;
    cpu_user     = utime / (float) sysconf(_SC_CLK_TCK);
    cpu_sys      = stime / (float) sysconf(_SC_CLK_TCK);
}

//-------------------------------------------------
// READDIST.CPP
//-------------------------------------------------


int Csu ( int * DC, double * mc, int * DYT, double * yt, double *rn ) {

    // [ara ver que pasa/.......................
    //double l;

    FILE *fZ, *fX;
    int ni, i,j, info;
    int *DZ, *DY, *DX;
    double *zb, *Xb, *yb, *nb, *temp;
    int nzb, nxb, ns, pzb, pxb, sc, lld_Z, lld_X, pcol, ccu,rcu;

    DZ= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DZ==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DY==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    DX= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DX==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    ns= n % ( b * * ( ds+1 ) ) ==0 ?  n / ( b * * ( ds+1 ) ) : ( n / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    nzb= m%b==0 ? m/b : m/b +1;
    pzb= ( nzb - *pst ) % *ds == 0 ? ( nzb- *pst ) / *ds : ( nzb- *pst ) / *ds +1;
    pzb= pzb <1? 1:pzb;
    lld_Z=pzb*b;
    nxb= t%b==0 ? t/b : t/b +1;
    pxb= ( nxb - *pst ) % *ds == 0 ? ( nxb- *pst ) / *ds : ( nxb- *pst ) / *ds +1;
    pxb= pxb <1? 1:pxb;
    lld_X=pxb*b;

    descinit_ ( DZ, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_Z, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    descinit_ ( DY, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DX, &t, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_X, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix X returns info: %d\n",info );
        return info;
    }

    zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
    if ( zb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }

    yb = ( double* ) calloc ( b,sizeof ( double ) );
    if ( yb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    Xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
    if ( Xb==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    nb = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    temp=mc;
    for ( i=0,rcu=0,ccu=0; i<Cb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds ) {
            rcu=0;
            temp += b;
        }
        if ( ccu==* ( ds+1 ) ) {
            ccu=0;
            temp += b*lld_C;
        }
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            for ( j=0; j<b; ++j ) {
                * ( temp + j  * lld_C +j ) =l;
            }
            if ( i==Cb-1 && Cd % b != 0 ) {
                for ( j=b-1; j>= Cd % b; --j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            }
        }

    }
    temp=mc;
    for ( i=0,rcu=0,ccu=0; i<nxb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds ) {
            rcu=0;
            temp += b;
        }
        if ( ccu==* ( ds+1 ) ) {
            ccu=0;
            temp += b*lld_C;
        }
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i<nxb-1 ) {
                for ( j=0; j<b; ++j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            } else {
                for ( j=0; j<= ( t-1 ) %b; ++j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            }
        }
    }


    fZ=fopen ( fDn,"rb" );
    if ( fZ==NULL ) {
        printf ( "Error opening file 1661\n" );
        return -1;
    }

    fX=fopen ( fXn,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file 1667\n" );
        return -1;
    }
    *rn=0.0;
    *nb=0.0;

    for ( ni=0; ni<ns; ++ni ) {
        if ( ni==ns-1 ) {

            free ( zb );
            free ( yb );
            free ( Xb );
            zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
            if ( zb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }
            yb = ( double* ) calloc ( b,sizeof ( double ) );
            if ( yb==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
            Xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
            if ( Xb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }
        }

        if ( ( nzb-1 ) % *ds == *pst && m%b !=0 ) {
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * b * ( m+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( b * (*( ds+1 )-1) * ( m+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fZ, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( yb + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pzb-1; ++j ) {
                    fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                    info=fseek ( fZ, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( zb + i*pzb*b + j*b,sizeof ( double ),m%b,fZ );
            }
        }
        else {
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * b * ( m+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( b * (*( ds+1 )-1) * ( m+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fZ, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( yb + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pzb-1; ++j ) {
                    fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                    info=fseek ( fZ, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                info=fseek ( fZ, ( long ) ( (m - b * ((pzb-1) * *ds + *pst +1 )) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
        }

        if ( ( nxb-1 ) % *ds == *pst && t%b !=0 ) {
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( b * (*( ds+1 )-1) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( Xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( Xb + i*pxb*b + j*b,sizeof ( double ),t%b,fX );
            }
        } else {
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( b * (*( ds+1 )-1) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( Xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( Xb + i*pxb*b + j*b,sizeof ( double ),b,fX  );
                info=fseek ( fX, ( long ) ( (t - b * ((pxb-1) * *ds + *pst +1 )) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );

        pdsyrk_ ( "U","N",&m,&sc,&d_one, zb,&i_one, &i_one,DZ, &d_one, mc, &t_plus, &t_plus, DC );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&d_one,zb,&i_one, &i_one, DZ,yb,&i_one,&i_one,DY,&d_one,yt,&t_plus,&i_one,DYT );

        pdsyrk_ ( "U","N",&t,&sc,&d_one, Xb,&i_one, &i_one,DX, &d_one, mc, &i_one, &i_one, DC );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&d_one,Xb,&i_one, &i_one, DX,yb,&i_one,&i_one,DY,&d_one,yt,&i_one,&i_one,DYT );

        pdgemm_ ( "N","T",&t,&m,&sc,&d_one,Xb,&i_one, &i_one, DX,zb,&i_one,&i_one,DZ,&d_one,mc,&i_one,&t_plus,DC );

        pdnrm2_ ( &sc,nb,yb,&i_one,&i_one,DY,&i_one );
        *rn += *nb * *nb;

        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    info=fclose ( fZ );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    free ( DX );
    free ( DY );
    free ( DZ );
    free ( zb );
    free ( Xb );
    free ( yb );
    free ( nb );
    return 0;
}

int Cu ( int * DC, double * mc, double du) {

    int i,j, rcu,ccu,nxb;

    nxb= t%b==0 ? t/b : t/b +1;

    for ( i=0,rcu=0,ccu=0; i<Cb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds )
            rcu=0;
        if ( ccu==* ( ds+1 ) )
            ccu=0;
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i< ( Cb -1 ) ) {
                for ( j=0; j<b; ++j ) {
                    * ( mc+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ) +=du;
                }
            } else {
                for ( j=0; j< Cd % b; ++j ) {
                    * ( mc+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ) +=du;
                }
            }
        }
    }

    for ( i=0,rcu=0,ccu=0; i<nxb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds )
            rcu=0;
        if ( ccu==* ( ds+1 ) )
            ccu=0;
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i<nxb-1 ) {
                for ( j=0; j<b; ++j ) {
                    * ( mc+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ) -=du;
                }
            } else {
                for ( j=0; j< t%b; ++j ) {
                    * ( mc+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ) -=du;
                }
            }
        }
    }
}


int Asu ( double * mai, int * dai,int * dyt, double * yt, int * dc, double * mc, double sa ) {

    FILE *fZ, *fX;
    int ni, i,j, info;
    int *dz, *dy, *dx, *dzu, *dqr, *dqs;
    double *zb, *xb, *yb, *zub, *qr, *qs,*nb, sr;
    int nzb, nxb, nst, pzb, pxb, sc, lld_Z, lld_X, pcol, ccu,rcu;

    dz= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dz==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    dy= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dy==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    dx= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dx==NULL ) {
        printf ( "unable to allocate memory for descriptor for X\n" );
        return -1;
    }
    dzu= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dzu==NULL ) {
        printf ( "unable to allocate memory for descriptor for Zu\n" );
        return -1;
    }
    dqr= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dqr==NULL ) {
        printf ( "unable to allocate memory for descriptor for QRHS\n" );
        return -1;
    }
    dqs= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dqs==NULL ) {
        printf ( "unable to allocate memory for descriptor for QSOL\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    nst= n % ( b * * ( ds+1 ) ) ==0 ?  n / ( b * * ( ds+1 ) ) : ( n / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    nzb= m%b==0 ? m/b : m/b +1;
    pzb= ( nzb - *pst ) % *ds == 0 ? ( nzb- *pst ) / *ds : ( nzb- *pst ) / *ds +1;
    pzb= pzb <1? 1:pzb;
    lld_Z=pzb*b;
    nxb= t%b==0 ? t/b : t/b +1;
    pxb= ( nxb - *pst ) % *ds == 0 ? ( nxb- *pst ) / *ds : ( nxb- *pst ) / *ds +1;
    pxb= pxb <1? 1:pxb;
    lld_X=pxb*b;
    sr=1/sa;

    descinit_ ( dz, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_Z, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dy, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( dx, &t, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_X, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix X returns info: %d\n",info );
        return info;
    }
    descinit_ ( dzu, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dqr, &Cd, &i_two, &b, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dqs, &Cd, &i_two, &b, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
    if ( zb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    yb = ( double* ) calloc ( b,sizeof ( double ) );
    if ( yb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    zub = ( double* ) calloc ( b,sizeof ( double ) );
    if ( zub==NULL ) {
        printf ( "unable to allocate memory for Matrix Zu\n" );
        return EXIT_FAILURE;
    }
    xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
    if ( xb==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    qr= ( double * ) calloc ( Cr * b * 2,sizeof ( double ) );
    if ( qr==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    qs= ( double * ) calloc ( Cr * b * 2,sizeof ( double ) );
    if ( qs==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    nb = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nb==NULL ) {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }

    fZ=fopen ( fDn,"rb" );
    if ( fZ==NULL ) {
        printf ( "Error opening file 2047\n" );
        return -1;
    }

    fX=fopen ( fXn,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file 2053\n" );
        return -1;
    }
    *nb=0.0;

    for ( ni=0; ni<nst; ++ni ) {
        if ( ni==nst-1 ) {
            free ( zb );
            free ( zub );
            free ( yb );
            free ( xb );
            zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
            if ( zb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }
            yb = ( double* ) calloc ( b,sizeof ( double ) );
            if ( yb==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
            xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
            if ( xb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
                return -1;
            }
            zub = ( double* ) calloc ( b,sizeof ( double ) );
            if ( zub==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
        }


        if ( ( nzb-1 ) % *ds == *pst && m%b !=0 ) {
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * b * ( m+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( b * (*( ds+1 )-1) * ( m+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fZ, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( yb + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pzb-1; ++j ) {
                    fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                    info=fseek ( fZ, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( zb + i*pzb*b + j*b,sizeof ( double ),m%b,fZ );
            }
        }
        else {
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * b * ( m+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( b * (*( ds+1 )-1) * ( m+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fZ, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                if ( *pst==0 )
                    fread ( yb + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pzb-1; ++j ) {
                    fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                    info=fseek ( fZ, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( zb + i*pzb*b + j*b,sizeof ( double ),b,fZ );
                if (m>*pst * b) {
                    info=fseek ( fZ, ( long ) ( (m - b * ((pzb-1) * *ds + *pst +1 )) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
            }
        }

        if ( ( nxb-1 ) % *ds == *pst && t%b !=0 ) {
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( b * (*( ds+1 )-1) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),t%b,fX );
            }
        } else {
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( b * (*( ds+1 )-1) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX  );
                if (t>*pst * b) {
                    info=fseek ( fX, ( long ) ( (t - b * ((pxb-1) * *ds + *pst +1 )) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
            }
        }
        blacs_barrier_ ( &ICTXT2D,"A" );

        pdgemm_ ( "T","N", &i_one, &sc,&m,&l, yt, &t_plus,&i_one,dyt,zb,&i_one,&i_one,dz,&d_zero,zub,&i_one,&i_one,dzu );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&sr,zb,&i_one, &i_one, dz,yb,&i_one,&i_one,dy,&d_one,qr,&t_plus,&i_one,dqr );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&sr,xb,&i_one, &i_one, dx,yb,&i_one,&i_one,dy,&d_one,qr,&i_one,&i_one,dqr );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&d_one,zb,&i_one, &i_one, dz,zub,&i_one,&i_one,dzu,&d_one,qr,&t_plus,&i_two,dqr );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&d_one,xb,&i_one, &i_one, dx,zub,&i_one,&i_one,dzu,&d_one,qr,&i_one,&i_two,dqr );

        pdnrm2_ ( &sc,nb,yb,&i_one,&i_one,dy,&i_one );
        *mai += *nb * *nb/sa/sa;
        pdnrm2_ ( &sc,nb,zub,&i_one,&i_one,dzu,&i_one );
        * ( mai + 3 ) += *nb * *nb;
        pddot_ ( &sc,nb,zub,&i_one,&i_one,dzu,&i_one,yb,&i_one,&i_one,dy,&i_one );
        * ( mai + 1 ) += *nb /sa;
        * ( mai + 2 ) += *nb /sa;
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    pdcopy_ ( &Cd,qr,&i_one,&i_two,dqr,&i_one,qs,&i_one,&i_two,dqs,&i_one );
    pdcopy_ ( &Cd,yt,&i_one,&i_one,dyt,&i_one,qs,&i_one,&i_one,dqs,&i_one );
    pdscal_ ( &Cd,&sr,qs,&i_one,&i_one,dqs,&i_one );
    pdpotrs_ ( "U",&Cd,&i_one,mc,&i_one,&i_one,dc,qs,&i_one,&i_two,dqs,&info );
    if ( info!=0 )
        printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );

    pdgemm_ ( "T","N",&i_two,&i_two,&Cd,&d_negone,qr,&i_one,&i_one,dqr,qs,&i_one,&i_one,dqs,&d_one, mai,&i_one,&i_one,dai );

    for ( i=0; i<4; ++i )
        * ( mai + i ) = * ( mai + i ) / 2 / sa;

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    info=fclose ( fZ );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    free ( dqr );
    free ( dqs );
    free ( dx );
    free ( dy );
    free ( dz );
    free ( dzu );
    free ( zb );
    free ( xb );
    free ( yb );
    free ( nb );
    free ( qr );
    free ( qs );
    free ( zub );

    return 0;
}

double CZt ( double *cm, int * dcm ) {

    double tp;
    int i, j, rcu,ccu, nxb;

    tp=0.0;

    nxb= t%b==0 ? t/b : t/b +1;

    for ( i=0,rcu=0,ccu=0; i<Cb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds )
            rcu=0;
        if ( ccu==* ( ds+1 ) )
            ccu=0;
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i< ( Cb -1 ) ) {
                for ( j=0; j<b; ++j ) {
                    tp += * ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j );
                }
            } else {
                for ( j=0; j< Cd % b; ++j ) {
                    tp += * ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j );
                }
            }
        }
    }

    for ( i=0,rcu=0,ccu=0; i<nxb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds )
            rcu=0;
        if ( ccu==* ( ds+1 ) )
            ccu=0;
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i<nxb-1 ) {
                for ( j=0; j<b; ++j ) {
                    tp -= * ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j );
                }
            } else {
                for ( j=0; j<= (t-1)%b; ++j ) {
                    tp -= * ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j );
                }
            }
        }
    }
    return tp;
}

double Cld ( double *cm, int * dcm ) {

    double ldp;
    int i, j, rcu,ccu, nxb;

    ldp=0.0;

    for ( i=0,rcu=0,ccu=0; i<Cb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds )
            rcu=0;
        if ( ccu==* ( ds+1 ) )
            ccu=0;
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i< ( Cb -1 ) ) {
                for ( j=0; j<b; ++j ) {
                    ldp += log( * ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ));
                }
            } else {
                for ( j=0; j< Cd % b; ++j ) {
                    ldp += log(* ( cm+ ( j + i / * ( ds+1 ) * b ) * lld_C + i / *ds *b +j ));
                }
            }
        }
    }
    return ldp;
}

//--------------------------------------------------------------
// READHDF5.CPP
//--------------------------------------------------------------

int Csu5 ( int * dco, double * mco, int * dy, double * ty, double *rn) {
    FILE *fX;
    int ni, i,j, info;
    int *dz, *dey, *dx;
    double *zb, *xb, *yb, *nb, *temp;
    int nzb, nxb, nst, pzb, pxb, sc, lld_Z, lld_X, pcol, ccu,rcu;

    // para vber qe sucede..................
    //double l;


    hid_t       fid, dgi, dpi, sgi;
    hid_t	pid, msg, spi, msp;
    herr_t	status;
    hsize_t	dm[2], os[2],co[2], st[2],bl[2];

    int mpinfo  = MPI_INFO_NULL;

    pid = H5Pcreate ( H5P_FILE_ACCESS );
cout<<"----estoy en Csu5 pid es:"<< pid<<" mpi info:"<<MPI_INFO_NULL<<endl;


//    H5Pset_fapl_mpio ( pid, MPI_COMM_WORLD, mpinfo );

    if (mpinfo<0) {
        printf("Something went wrong with setting IO options for HDF5-file, error: %d \n",mpinfo);
        return mpinfo;
    }

    fid = H5Fopen ( fDn, H5F_ACC_RDONLY, pid );
    if (fid <0) {
        printf("Something went wrong with opening HDF5-file, error: %d \n",fid);
        return fid;
    }


    dgi = H5Dopen ( fid, Sd, H5P_DEFAULT );
    if (dgi <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dgi);
        return dgi;
    }
    dpi = H5Dopen ( fid, pd, H5P_DEFAULT );
    if (dpi <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dpi);
        return dpi;
    }
    sgi=H5Dget_space ( dgi );
    if (sgi <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",sgi);
        return sgi;
    }
    spi=H5Dget_space ( dpi );
    if (spi <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",spi);
        return spi;
    }

    dz= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dz==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    dey= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dey==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    dx= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dx==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    nst= n % ( b * * ( ds+1 ) ) ==0 ?  n / ( b * * ( ds+1 ) ) : ( n / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    nzb= m%b==0 ? m/b : m/b +1;
    pzb= ( nzb - *pst ) % *ds == 0 ? ( nzb- *pst ) / *ds : ( nzb- *pst ) / *ds +1;
    pzb= pzb <1? 1:pzb;
    lld_Z=pzb*b;
    nxb= t%b==0 ? t/b : t/b +1;
    pxb= ( nxb - *pst ) % *ds == 0 ? ( nxb- *pst ) / *ds : ( nxb- *pst ) / *ds +1;
    pxb= pxb <1? 1:pxb;
    lld_X=pxb*b;

    descinit_ ( dz, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_Z, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    descinit_ ( dey, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( dx, &t, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_X, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix X returns info: %d\n",info );
        return info;
    }

    zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
    if ( zb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    dm[0]=b;
    dm[1]=pzb*b;
    msg = H5Screate_simple ( 2,dm,NULL );

    yb = ( double* ) calloc ( b,sizeof ( double ) );
    if ( yb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    dm[0]=b;
    dm[1]=1;
    msp = H5Screate_simple ( 1,dm,NULL );

    xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
    if ( xb==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    nb = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    temp=mco;
    for ( i=0,rcu=0,ccu=0; i<Cb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds ) {
            rcu=0;
            temp += b;
        }
        if ( ccu==* ( ds+1 ) ) {
            ccu=0;
            temp += b*lld_C;
        }
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            for ( j=0; j<b; ++j ) {
                * ( temp + j  * lld_C +j ) =l;
            }
            if ( i==Cb-1 && Cd % b != 0 ) {
                for ( j=b-1; j>= Cd % b; --j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            }
        }

    }
    temp=mco;
    for ( i=0,rcu=0,ccu=0; i<nxb; ++i, ++ccu, ++rcu ) {
        if ( rcu==*ds ) {
            rcu=0;
            temp += b;
        }
        if ( ccu==* ( ds+1 ) ) {
            ccu=0;
            temp += b*lld_C;
        }
        if ( *pst==rcu && * ( pst+1 ) == ccu ) {
            if ( i<nxb-1 ) {
                for ( j=0; j<b; ++j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            } else {
                for ( j=0; j<= ( t-1 ) %b; ++j ) {
                    * ( temp + j * lld_C + j ) =0.0;
                }
            }
        }
    }

    //cout<<"archivo que no se deja abrir"<<fXn<<endl;
    fX=fopen ( fXn,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file 2555\n" );
        return -1;
    }
    *rn=0.0;
    *nb=0.0;

    pid = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( pid, H5FD_MPIO_INDEPENDENT );

    for ( ni=0; ni<nst; ++ni ) {
        if ( *pst >= nzb )
            goto CALC2;
        if ( ni==nst-1 ) {

            free ( zb );
            free ( yb );
            free ( xb );
            zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
            if ( zb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }
            yb = ( double* ) calloc ( b,sizeof ( double ) );
            if ( yb==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
            xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
            if ( xb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }

            if ( ( pcol + 1 + ( nst-1 ) * * ( ds+1 ) ) *b <= n )
                bl[0]=b;
            else if ( ( pcol + ( nst-1 ) * * ( ds+1 ) ) *b >= n )
                bl[0]=0;
            else
                bl[0]=n%b;
        } else {
            bl[0]=b;
        }
        if ( ( nzb-1 ) % *ds == *pst && m%b !=0 ) {
            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pzb-1;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }
            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }

            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = ( nzb-1 ) * b;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = m%b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_OR, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            os[0] = 0;
            os[1] = ( pzb-1 ) * b;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_OR, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }
        } else {
            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pzb;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful\n" );
                return status;
            }
        }
        status= H5Dread ( dgi,H5T_NATIVE_DOUBLE_g,msg,sgi,pid,zb );
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }
        if ( *pst==0 ) {

            os[0] = ni * b * * ( ds+1 ) + pcol * b;
            os[1] = 0;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * *ds;
            st[1] = 1;
            bl[1] = 1;

            status = H5Sselect_hyperslab ( spi, H5S_SELECT_SET, os, st, co,bl );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }
            os[0] = 0;
            os[1] = 0;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * *ds;
            st[1] = 1;
            bl[1] = 1;

            status = H5Sselect_hyperslab ( msp, H5S_SELECT_SET, os, st, co,bl );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }

            status=H5Dread ( dpi,H5T_NATIVE_DOUBLE_g,msp,spi,pid,yb );
            if ( status<0 ) {
                printf ( "reading of pheno hyperslab was unsuccesful\n" );
                return -1;
            }

        }

        if ( ( nxb-1 ) % *ds == *pst && t%b !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( b * ( * ( ds+1 )-1 ) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),t%b,fX );
            }
        } else {
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( b * ( * ( ds+1 )-1 ) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                if ( t>*pst * b ) {
                    info=fseek ( fX, ( long ) ( ( t - b * ( ( pxb-1 ) * *ds + *pst +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
            }
        }

CALC2:
        blacs_barrier_ ( &ICTXT2D,"A" );

        pdsyrk_ ( "U","N",&m,&sc,&d_one, zb,&i_one, &i_one,dz, &d_one, mco, &t_plus, &t_plus, dco );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&d_one,zb,&i_one, &i_one, dz,yb,&i_one,&i_one,dey,&d_one,ty,&t_plus,&i_one,dy );

        pdsyrk_ ( "U","N",&t,&sc,&d_one, xb,&i_one, &i_one,dx, &d_one, mco, &i_one, &i_one, dco );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&d_one,xb,&i_one, &i_one, dx,yb,&i_one,&i_one,dey,&d_one,ty,&i_one,&i_one,dy );

        pdgemm_ ( "N","T",&t,&m,&sc,&d_one,xb,&i_one, &i_one, dx,zb,&i_one,&i_one,dz,&d_one,mco,&i_one,&t_plus,dco );

        pdnrm2_ ( &sc,nb,yb,&i_one,&i_one,dey,&i_one );
        *rn += *nb * *nb;

        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    H5Dclose ( dgi );
    H5Dclose ( dpi );
    H5Sclose ( msg );
    H5Sclose ( msp );
    H5Sclose ( sgi );
    H5Sclose ( spi );

    H5Pclose ( pid );

    H5Fclose ( fid );
    free ( dx );
    free ( dey );
    free ( dz );
    free ( zb );
    free ( xb );
    free ( yb );
    free ( nb );
    return 0;

}

int Asu5 ( double * mai, int * dai,int * dyt, double * yt, int * dc, double * mc, double sa) {

    FILE *fX;
    int ni, i,j, info;
    int *dz, *dy, *dx, *dzu, *dqr, *dqs;
    double *zb, *xb, *yb, *zub, *qr, *qs,*nb, sr;
    int nzb, nxb, nst, pzb, pxb, sc, lld_Z, lld_X, pcol, ccu,rcu;

    hid_t       fid, dgi, dpi, sgi;
    hid_t	pid, msg, spi, msp;
    herr_t	status;
    hsize_t	dm[2], os[2],co[2], st[2],bl[2];

    MPI_Info mpinfo  = MPI_INFO_NULL;

    pid = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( pid, MPI_COMM_WORLD, mpinfo );

    fid = H5Fopen ( fDn, H5F_ACC_RDWR, pid );
    dgi = H5Dopen ( fid, Sd, H5P_DEFAULT );
    dpi = H5Dopen ( fid, pd, H5P_DEFAULT );
    sgi=H5Dget_space ( dgi );
    spi=H5Dget_space ( dpi );

    dz= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dz==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    dy= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dy==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    dx= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dx==NULL ) {
        printf ( "unable to allocate memory for descriptor for X\n" );
        return -1;
    }
    dzu= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dzu==NULL ) {
        printf ( "unable to allocate memory for descriptor for Zu\n" );
        return -1;
    }
    dqr= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dqr==NULL ) {
        printf ( "unable to allocate memory for descriptor for QRHS\n" );
        return -1;
    }
    dqs= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( dqs==NULL ) {
        printf ( "unable to allocate memory for descriptor for QSOL\n" );
        return -1;
    }

    pcol= * ( pst+1 );
    nst= n % ( b * * ( ds+1 ) ) ==0 ?  n / ( b * * ( ds+1 ) ) : ( n / ( b * * ( ds+1 ) ) ) +1;
    sc= b * * ( ds+1 );
    nzb= m%b==0 ? m/b : m/b +1;
    pzb= ( nzb - *pst ) % *ds == 0 ? ( nzb- *pst ) / *ds : ( nzb- *pst ) / *ds +1;
    pzb= pzb <1? 1:pzb;
    lld_Z=pzb*b;
    nxb= t%b==0 ? t/b : t/b +1;
    pxb= ( nxb - *pst ) % *ds == 0 ? ( nxb- *pst ) / *ds : ( nxb- *pst ) / *ds +1;
    pxb= pxb <1? 1:pxb;
    lld_X=pxb*b;
    sr=1/sa;

    descinit_ ( dz, &m, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_Z, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dy, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( dx, &t, &sc, &b, &b, &i_zero, &i_zero, &ICTXT2D, &lld_X, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix X returns info: %d\n",info );
        return info;
    }
    descinit_ ( dzu, &i_one, &sc, &i_one, &b, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dqr, &Cd, &i_two, &b, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( dqs, &Cd, &i_two, &b, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_C, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
    if ( zb==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    dm[0]=b;
    dm[1]=pzb*b;
    msg = H5Screate_simple ( 2,dm,NULL );

    yb = ( double* ) calloc ( b,sizeof ( double ) );
    if ( yb==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    dm[0]=b;
    dm[1]=1;
    msp = H5Screate_simple ( 1,dm,NULL );

    zub = ( double* ) calloc ( b,sizeof ( double ) );
    if ( zub==NULL ) {
        printf ( "unable to allocate memory for Matrix Zu\n" );
        return EXIT_FAILURE;
    }
    xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
    if ( xb==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    qr= ( double * ) calloc ( Cr * b * 2,sizeof ( double ) );
    if ( qr==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    qs= ( double * ) calloc ( Cr * b * 2,sizeof ( double ) );
    if ( qs==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*pst,* ( pst+1 ) );
        return -1;
    }
    nb = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nb==NULL ) {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }

    fX=fopen ( fXn,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file 2974\n" );
        return -1;
    }
    *nb=0.0;

    pid = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( pid, H5FD_MPIO_INDEPENDENT );

    for ( ni=0; ni<nst; ++ni ) {

        if ( *pst >= nzb )
            goto CALC;
        if ( ni==nst-1 ) {

            free ( zb );
            free ( yb );
            free ( xb );
            zb= ( double* ) calloc ( pzb*b*b, sizeof ( double ) );
            if ( zb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }
            yb = ( double* ) calloc ( b,sizeof ( double ) );
            if ( yb==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
            xb= ( double* ) calloc ( pxb*b*b, sizeof ( double ) );
            if ( xb==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*pst,* ( pst+1 ) );
                return -1;
            }

            if ( ( pcol + 1 + ( nst-1 ) * * ( ds+1 ) ) *b <= n )
                bl[0]=b;
            else if ( ( pcol + ( nst-1 ) * * ( ds+1 ) ) *b >= n )
                bl[0]=0;
            else
                bl[0]=n%b;
        } else {
            bl[0]=b;
        }
        if ( ( nzb-1 ) % *ds == *pst && m%b !=0 ) {
            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pzb-1;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }
            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }

            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = ( nzb-1 ) * b;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = m%b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_OR, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            os[0] = 0;
            os[1] = ( pzb-1 ) * b;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_OR, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }
        } else {
            os[0] = ni * * ( ds+1 ) * b + pcol * b;
            os[1] = *pst * b;
            co[0] = 1;
            co[1] = pzb;
            st[0] = b * * ( ds+1 );
            st[1] = b * *ds;
            bl[1] = b;

            status = H5Sselect_hyperslab ( sgi, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            os[0] = 0;
            os[1] = 0;
            st[0] = b;
            st[1] = b;

            status = H5Sselect_hyperslab ( msg, H5S_SELECT_SET, os, st, co, bl );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful\n" );
                return status;
            }
        }
        status= H5Dread ( dgi,H5T_NATIVE_DOUBLE_g,msg,sgi,pid,zb );
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }
        if ( *pst==0 ) {

            os[0] = ni * b * * ( ds+1 ) + pcol * b;
            os[1] = 0;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * *ds;
            st[1] = 1;
            bl[1] = 1;

            status = H5Sselect_hyperslab ( spi, H5S_SELECT_SET, os, st, co,bl );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }
            os[0] = 0;
            os[1] = 0;
            co[0] = 1;
            co[1] = 1;
            st[0] = b * *ds;
            st[1] = 1;
            bl[1] = 1;

            status = H5Sselect_hyperslab ( msp, H5S_SELECT_SET, os, st, co,bl );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }

            status=H5Dread ( dpi,H5T_NATIVE_DOUBLE_g,msp,spi,pid,yb );
            if ( status<0 ) {
                printf ( "reading of pheno hyperslab was unsuccesful\n" );
                return -1;
            }

        }

        if ( ( nxb-1 ) % *ds == *pst && t%b !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( b * ( * ( ds+1 )-1 ) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( ( *ds ) -1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),t%b,fX );
            }
        } else {
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * b *  t * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( b * ( * ( ds+1 )-1 ) * t * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<b; ++i ) {
                info=fseek ( fX, ( long ) ( b * *pst * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                    return -1;
                }
                for ( j=0; j < pxb-1; ++j ) {
                    fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                    info=fseek ( fX, ( long ) ( ( * ( ds )-1 ) * b * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
                fread ( xb + i*pxb*b + j*b,sizeof ( double ),b,fX );
                if ( t>*pst * b ) {
                    info=fseek ( fX, ( long ) ( ( t - b * ( ( pxb-1 ) * *ds + *pst +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *pst,pcol,info );
                        return -1;
                    }
                }
            }
        }
CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );

        //[ara ver que pasa
        //double l;


        pdgemm_ ( "T","N", &i_one, &sc,&m,&l, yt, &t_plus,&i_one,dyt,zb,&i_one,&i_one,dz,&d_zero,zub,&i_one,&i_one,dzu );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&sr,zb,&i_one, &i_one, dz,yb,&i_one,&i_one,dy,&d_one,qr,&t_plus,&i_one,dqr );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&sr,xb,&i_one, &i_one, dx,yb,&i_one,&i_one,dy,&d_one,qr,&i_one,&i_one,dqr );

        pdgemm_ ( "N","T",&m,&i_one,&sc,&d_one,zb,&i_one, &i_one, dz,zub,&i_one,&i_one,dzu,&d_one,qr,&t_plus,&i_two,dqr );

        pdgemm_ ( "N","T",&t,&i_one,&sc,&d_one,xb,&i_one, &i_one, dx,zub,&i_one,&i_one,dzu,&d_one,qr,&i_one,&i_two,dqr );

        pdnrm2_ ( &sc,nb,yb,&i_one,&i_one,dy,&i_one );
        *mai += *nb * *nb/sa/sa;
        pdnrm2_ ( &sc,nb,zub,&i_one,&i_one,dzu,&i_one );
        * ( mai + 3 ) += *nb * *nb;
        pddot_ ( &sc,nb,zub,&i_one,&i_one,dzu,&i_one,yb,&i_one,&i_one,dy,&i_one );
        * ( mai + 1 ) += *nb /sa;
        * ( mai + 2 ) += *nb /sa;
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    pdcopy_ ( &Cd,qr,&i_one,&i_two,dqr,&i_one,qs,&i_one,&i_two,dqs,&i_one );
    pdcopy_ ( &Cd,yt,&i_one,&i_one,dyt,&i_one,qs,&i_one,&i_one,dqs,&i_one );
    pdscal_ ( &Cd,&sr,qs,&i_one,&i_one,dqs,&i_one );
    pdpotrs_ ( "U",&Cd,&i_one,mc,&i_one,&i_one,dc,qs,&i_one,&i_two,dqs,&info );
    if ( info!=0 )
        printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );

    pdgemm_ ( "T","N",&i_two,&i_two,&Cd,&d_negone,qr,&i_one,&i_one,dqr,qs,&i_one,&i_one,dqs,&d_one, mai,&i_one,&i_one,dai );

    for ( i=0; i<4; ++i )
        * ( mai + i ) = * ( mai + i ) / 2 / sa;

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    H5Dclose ( dgi );
    H5Dclose ( dpi );
    H5Sclose ( msg );
    H5Sclose ( msp );
    H5Sclose ( sgi );
    H5Sclose ( spi );
    H5Pclose ( pid );
    H5Fclose ( fid );

    free ( dqr );
    free ( dqs );
    free ( dx );
    free ( dy );
    free ( dz );
    free ( dzu );
    free ( zb );
    free ( xb );
    free ( yb );
    free ( nb );
    free ( qr );
    free ( qs );
    free ( zub );

    return 0;
}

