set(CMAKE_Fortran_COMPILER "/apps/gcc/14.1.0/bin/gfortran")
set(CMAKE_Fortran_COMPILER_ARG1 "")
set(CMAKE_Fortran_COMPILER_ID "GNU")
set(CMAKE_Fortran_COMPILER_VERSION "14.1.0")
set(CMAKE_Fortran_COMPILER_WRAPPER "")
set(CMAKE_Fortran_PLATFORM_ID "")
set(CMAKE_Fortran_SIMULATE_ID "")
set(CMAKE_Fortran_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_Fortran_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_Fortran_COMPILER_AR "/apps/gcc/14.1.0/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_Fortran_COMPILER_RANLIB "/apps/gcc/14.1.0/bin/gcc-ranlib")
set(CMAKE_TAPI "CMAKE_TAPI-NOTFOUND")
set(CMAKE_COMPILER_IS_GNUG77 1)
set(CMAKE_Fortran_COMPILER_LOADED 1)
set(CMAKE_Fortran_COMPILER_WORKS TRUE)
set(CMAKE_Fortran_ABI_COMPILED TRUE)

set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")

set(CMAKE_Fortran_COMPILER_SUPPORTS_F90 1)

set(CMAKE_Fortran_COMPILER_ID_RUN 1)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;fpp;FPP;f77;F77;f90;F90;for;For;FOR;f95;F95;f03;F03;f08;F08)
set(CMAKE_Fortran_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_Fortran_LINKER_PREFERENCE 20)
set(CMAKE_Fortran_LINKER_DEPFILE_SUPPORTED TRUE)
if(UNIX)
  set(CMAKE_Fortran_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Fortran_OUTPUT_EXTENSION .obj)
endif()

# Save compiler ABI information.
set(CMAKE_Fortran_SIZEOF_DATA_PTR "8")
set(CMAKE_Fortran_COMPILER_ABI "")
set(CMAKE_Fortran_LIBRARY_ARCHITECTURE "")

if(CMAKE_Fortran_SIZEOF_DATA_PTR AND NOT CMAKE_SIZEOF_VOID_P)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_Fortran_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_Fortran_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_Fortran_COMPILER_ABI}")
endif()

if(CMAKE_Fortran_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()





set(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES "/vapps/rhel9/x86_64/gcc/14.1.0/lib/gcc/x86_64-pc-linux-gnu/14.1.0/finclude;/grphome/grp_GR/apps/intel/oneapi/mkl/2024.2/include;/grphome/grp_GR/apps/gsl-2.8/include;/grphome/grp_GR/apps/openmpi-5.0.3/include;/apps/spack/root/opt/spack/linux-rhel9-haswell/gcc-13.2.0/eigen-3.4.0-6kqviac4ysj3yrvvolnitcc5zx7qgcml/include/eigen3;/apps/python/3.12.2/gcc-11.4.1/include/python3.12;/vapps/rhel9/x86_64/gcc/14.1.0/lib/gcc/x86_64-pc-linux-gnu/14.1.0/include;/vapps/rhel9/x86_64/gcc/14.1.0/lib/gcc/x86_64-pc-linux-gnu/14.1.0/include-fixed;/usr/local/include;/vapps/rhel9/x86_64/gcc/14.1.0/include;/usr/include")
set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "gfortran;m;gcc_s;gcc;quadmath;m;c;gcc_s;gcc")
set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "/vapps/rhel9/x86_64/gcc/14.1.0/lib/gcc/x86_64-pc-linux-gnu/14.1.0;/vapps/rhel9/x86_64/gcc/14.1.0/lib/gcc;/vapps/rhel9/x86_64/gcc/14.1.0/lib64;/lib64;/usr/lib64;/grphome/grp_GR/apps/intel/oneapi/mkl/2024.2/lib;/grphome/grp_GR/apps/gsl-2.8/lib;/grphome/grp_GR/apps/openmpi-5.0.3/lib/openmpi;/grphome/grp_GR/apps/openmpi-5.0.3/lib;/apps/python/3.12.2/gcc-11.4.1/lib;/vapps/rhel9/x86_64/gcc/14.1.0/lib")
set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
