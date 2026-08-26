# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-src"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-build"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/tmp"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/src"
  "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ngarey/Dendro/em4-CKO/build_cko/_deps/toml11-subbuild/toml11-populate-prefix/src/toml11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
