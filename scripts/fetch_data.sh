#!/bin/bash
# Downloads every corpus the experiments read into data/raw/ and checks SHA-256 hashes.
# Provenance, licences and the full inventory are in data/SOURCES.md. The Session and IrishMAN are not
# fetched: The Session's licence forbids processing its tunes with language-model tools.
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }

get() {  # get <dir> <file> <url> <sha256>
  mkdir -p "$1"
  if [ ! -f "$1/$2" ]; then curl -fL --retry 3 -o "$1/$2" "$3"; fi
  echo "$4  $1/$2" | shasum -a 256 -c -
}
unzip_once() { [ -d "${1%.zip}" ] || [ -n "$(ls -d "$(dirname "$1")"/*/ 2>/dev/null)" ] || unzip -q "$1" -d "$(dirname "$1")"; }

TCN=https://raw.githubusercontent.com/locuslab/TCN/2f8c2b817050206397458dfd1f5a25ce8a32fe65/TCN/poly_music/mdata
get boulanger2012 tcn_JSB_Chorales.mat $TCN/JSB_Chorales.mat ac0e608527cc411c7e21ef7d38be47f0de7fb6b13b3ae168b8eccf1c2a9b18d4
get boulanger2012 tcn_MuseData.mat     $TCN/MuseData.mat     e8d9eb422ec3833c25f43a9f854d4ee0b9118a20b45f2c183395cdc30ffa14ff
get boulanger2012 tcn_Nottingham.mat   $TCN/Nottingham.mat   d154f7ca2e90799dc1dc011d8fc048f2fc6ef3f7c69d974b9cf27a471634fbea
get boulanger2012 tcn_Piano_midi.mat   $TCN/Piano_midi.mat   990a8a71d5764a7eaa3e78d9e5023d1470380fe2db3a1448265a97a24c8b1c34

get nottingham_abc nottingham-dataset-0992bb6.zip https://codeload.github.com/jukedeck/nottingham-dataset/zip/0992bb6cd864f6d4b90d6663e08c0ef53ffaa08f 9c9c87d745779af617ad4cd0d715210bfc17ff89bc427928593462872542f9a8
get essen essen-folksong-collection-2d0ca75.zip https://codeload.github.com/ccarh/essen-folksong-collection/zip/2d0ca75e87dc7a725556c8090e3681c1fa3a0452 3971a8be56e9f8f50903c1b3e17ed1c9e4218d979e32c395b700fba0aed086e0
get pop909 POP909-Dataset-d83e6ed.zip https://codeload.github.com/music-x-lab/POP909-Dataset/zip/d83e6edba6872a704f5d3b8b32f5cb540088dae6 fe3d861f6eeaee6b40987e64ae26552b4c7b824f6fa40d29b0c5e5ebd62b03ea
for z in nottingham_abc/*.zip essen/*.zip pop909/*.zip; do unzip_once "$z"; done

get hooktheory Hooktheory.json.gz https://github.com/chrisdonahue/sheetsage-data/raw/06113c04b109a2f27517b0399ff47550099f2466/hooktheory/Hooktheory.json.gz 917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c

Z=https://zenodo.org/api/records/15571083/files
get pdmx PDMX.csv          $Z/PDMX.csv/content          fc2187e7e09185f4b28be57b6478a96a1243a037f8fd13624f17de2d8bfd44bd
get pdmx mid.tar.gz        $Z/mid.tar.gz/content        e444f9b466f02c9a054d31478c9886847f39c575a65ec45a0aaa1a5ee088c1d1
get pdmx subset_paths.tar.gz $Z/subset_paths.tar.gz/content 17529103f66af71bd313369029c0901a143049bf6fcaecdfd11f5ca588df4377
[ -d pdmx/mid ] || tar -xzf pdmx/mid.tar.gz -C pdmx
[ -d pdmx/subset_paths ] || tar -xzf pdmx/subset_paths.tar.gz -C pdmx
echo "all inputs present and verified"
