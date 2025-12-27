set -euo pipefail

CXX=${CXX:-g++}
LTO=yes

for arg in "$@"; do [[ "$arg" == "--no-lto" ]] && LTO=no; done

PYINCLUDE=$(python3.14 -m pybind11 --includes)
PYEXT=$(python3.14-config --extension-suffix)
PYLDFLAGS=$(python3.14-config --ldflags)

CXXFLAGS="-Ofast -DNDEBUG -std=c++20 -fPIC -march=native -funroll-loops -mfma"
# CXXFLAGS="-DNDEBUG -std=c++20 -fPIC -march=native -funroll-loops -mfma"
[[ "$LTO" == "yes" ]] && CXXFLAGS+=" -flto" && PYLDFLAGS+=" -flto"

# add -fvisibility=hidden if supported
if echo 'int main(){}' | "$CXX" -x c++ - -std=c++20 -fvisibility=hidden -o /dev/null &>/dev/null; then
  CXXFLAGS+=" -fvisibility=hidden"
fi

if [[ "$(uname)" == "Darwin" ]]; then OMPFLAGS="-Xpreprocessor -fopenmp -lomp"; else OMPFLAGS="-fopenmp"; fi

echo "[build] Compiler : $CXX"
echo "[build] CXXFLAGS : $CXXFLAGS"
echo "[build] OpenMP   : $OMPFLAGS"

"$CXX" $CXXFLAGS $OMPFLAGS $PYINCLUDE brinicle/src/bindings.cpp $PYLDFLAGS -shared -o brinicle/brinicle"$PYEXT"
echo "[build] -> ./brinicle/brinicle$PYEXT"
