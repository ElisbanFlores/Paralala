DAIRRy-BLUP
============

DAIRRy-BLUP: A high performance computing approach to genomic prediction

A. De Coninck*1, J. Fostier†, S. Maenhout‡, B. De Baets*
* KERMIT, Department of Mathematical Modelling, Statistics and Bioinformatics, Ghent University, B-9000
Ghent, Belgium; † IBCN, Department of Information Technology, Ghent University - iMinds, B-9000 Ghent,
Belgium; ‡ Progeno, B-9052 Zwijnaarde, Belgium

A Distributed AI-REML Ridge Regression Best Linear Unbiased Prediction framework for genomic prediction.

This software was developed by Arne De Coninck and can only be used for research purposes.

En la predicción genómica, los métodos comunes de análisis se basan en un marco de modelo mixto lineal para estimar los efectos de marcadores SNP y los valores de cría tanto de animales como de plantas. Ridge Regression – Best Linear Unbiased Prediction (RR-BLUP) se basa en la hipótesis de que los efectos de marcadores SNP están distribuidos normalmente, no están correlacionados y tienen varianzas iguales. Se ha propuesto DAIRRy-BLUP, una implementación en paralelo de RR-BLUP en  memoria distribuida que utiliza el algoritmo de promedio información para la estimación restringida de máxima verosimilitud de los componentes de la varianza.

#Instalación

## Dependencias

DAIRRy-BLUP requiere lo siguiente:

1. MPI ([OpenMPI](http://www.open-mpi.org/), [MPICH](http://www.mpich.org/), [IntelMPI](http://software.intel.com/en-us/intel-mpi-library))
2. [ScaLAPACK](http://www.netlib.org/scalapack/) and all its dependencies BLAS, BLACS, LAPACK, PBLAS (Recomiendan instalar [vendor optimized implementation](http://www.netlib.org/scalapack/faq.html#1.3) )
3. HDF5 (http://www.hdfgroup.org/HDF5/)
4. CMake (http://www.cmake.org/)

La version original se compila con la libreria Intel MKL. para esta implementacion de ha utilizado MPICH2. 

# Uso

DAIRRy-BLUP requiere un archivo de entrada: `defaultinput.txt`, 
una vex generado el ejecurtable se debe ejecutar con el comando
`mpirun -np 4 ./DAIRRy-BLUP defaultinput.txt`

# Salida

DAIRRy-BLUP crea 1 or 2 archivos segun se haya configurado en el archivo de entrada.
* `estimates.txt`: Lista las estimaciones de los efectos aleatorios y fijos.  
* `EBV.txt`: Retorna los valores de cria estimados para cada individuo basado en los valores estimados fijos y los efectos de SNP y los genotipos de prueba de los individuos .

# Versiones

* Version 0.1 (12/2013):
  1. First public release of DAIRRy-BLUP

# Contact

 Correo del Autor arne.deconinck[at]ugent.be. 
