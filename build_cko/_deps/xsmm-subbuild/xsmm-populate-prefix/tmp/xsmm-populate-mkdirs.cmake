# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-src"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-build"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/tmp"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/src"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/xsmm-subbuild/xsmm-populate-prefix/src/xsmm-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
