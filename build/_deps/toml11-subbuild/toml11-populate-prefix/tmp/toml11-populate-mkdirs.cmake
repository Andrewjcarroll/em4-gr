# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-src"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-build"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/tmp"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/src"
  "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/andrew/Programs/CFD/EM4/em4-gr/build/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
