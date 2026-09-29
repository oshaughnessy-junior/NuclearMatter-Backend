#!/usr/bin/env bash
# Fetch, verify, patch and build NSCool (D. Page) outside the repository.
# NSCool ships without a licence, so nmbackend does not vendor it.
#   NSCOOL_HOME   install prefix (default ~/.local/share/nmbackend/nscool)
#   NSCOOL_TARBALL  use a local copy of NSCool.tar.gz instead of downloading
#   FC            Fortran compiler (default gfortran)
set -euo pipefail
URL=https://www.astroscu.unam.mx/neutrones/NSCool/NSCool.tar.gz
SHA256=c671fb959c732af595834152416cb60336009aa0e218b65ead55cae32f6af35a
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFIX=${NSCOOL_HOME:-$HOME/.local/share/nmbackend/nscool}
FC=${FC:-gfortran}
FFLAGS="-O3 -fno-automatic -std=legacy -w"

mkdir -p "$PREFIX"
cd "$PREFIX"
if [[ -n "${NSCOOL_TARBALL:-}" ]]; then
  cp "$NSCOOL_TARBALL" NSCool.tar.gz
elif [[ ! -f NSCool.tar.gz ]]; then
  curl -fsSL --max-time 300 -o NSCool.tar.gz.part "$URL"
  mv NSCool.tar.gz.part NSCool.tar.gz
fi
echo "$SHA256  NSCool.tar.gz" | sha256sum -c -

rm -rf NSCool
tar xzf NSCool.tar.gz
find NSCool \( -name '.DS_Store' -o -name '.#*' \) -delete
rm -f NSCool/NSCool.out NSCool/Code/NSCool.out NSCool/Plot/Plot_Teff NSCool/Model_1/Plot_Teff NSCool/TOV/TOV
(cd NSCool && patch -p1 -s < "$HERE/nscool_nmbackend.patch")

(cd NSCool/Code && make -s COMPILER="$FC" OPTIONS="$FFLAGS" NSCool.out >/dev/null)
(cd NSCool/TOV && "$FC" $FFLAGS -o TOV TOV.f)

{
  echo "source: $URL"
  echo "sha256: $SHA256"
  echo "patch:  nscool_nmbackend.patch sha256 $(sha256sum "$HERE/nscool_nmbackend.patch" | cut -d' ' -f1)"
  echo "fc:     $("$FC" --version | head -1)"
  echo "flags:  $FFLAGS"
  echo "host:   $(hostname)"
  echo "date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > NSCool/BUILD_INFO.txt
echo "NSCool built in $PREFIX/NSCool"
