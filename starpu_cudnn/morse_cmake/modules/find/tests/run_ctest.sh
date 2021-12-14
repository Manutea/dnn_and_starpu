#!/bin/bash

# env. var. CI_PROJECT_DIR required
[ -z "$CI_PROJECT_DIR" ] && echo "Variable CI_PROJECT_DIR must be set to execute this script." && exit 1

mkdir -p build
cd build

cmake $CI_PROJECT_DIR/modules/find/tests -DENABLE_CTEST=ON -DLAPACKE_COMPONENTS="TMG" -DQUARK_COMPONENTS="HWLOC" -DCMAKE_PREFIX_PATH="$PARSEC_DIR"

ctest -D Experimental
