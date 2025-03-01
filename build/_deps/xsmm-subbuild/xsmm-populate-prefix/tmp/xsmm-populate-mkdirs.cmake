# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-src"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-build"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/tmp"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/src"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
