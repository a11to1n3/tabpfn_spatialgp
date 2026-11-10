#!/bin/bash
# Raw files of every dataset of the study (docs/PROTOCOL.md, Annex A), downloaded from the original sources into <root>/<group>/ and
# checked against SHA-256 checksums before anything reads them. Archives are unpacked next to them.
# Usage: bash experiments/study/fetch.sh <root>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$1; mkdir -p "$root"; cd "$root"
# get <file> <url>...: the first URL that answers; a partial download never stays under the final name
get() {
  local f=$1; shift; [ -s "$f" ] && return 0; mkdir -p "$(dirname "$f")"
  for u in "$@"; do curl -fsSL --retry 3 --max-time 1800 -o "$f.part" "$u" && { mv "$f.part" "$f"; return 0; }; done
  rm -f "$f.part"; echo "download failed: $f" >&2; return 1
}
Z=https://zenodo.org/api/records

# uniformity trials (geoR)
get geor/geoR_1.9-6.tar.gz https://cran.r-project.org/src/contrib/geoR_1.9-6.tar.gz \
  https://cran.r-project.org/src/contrib/Archive/geoR/geoR_1.9-6.tar.gz
# SensRes (Denmark, Northern Ireland, Turkey)
for c in Denmark Northern_Ireland Turkey; do get "sensres/SensRes_$c.zip" "https://zenodo.org/records/14161809/files/SensRes_$c.zip?download=1"; done
# Midwest (Potash et al.)
get midwest/locations.csv https://ndownloader.figshare.com/files/41539671
get midwest/measurements.csv https://ndownloader.figshare.com/files/42223749
# new fields
get "newfields/ontario/2. SOC_Database.csv" "$Z/6611475/files/2.%20SOC_Database.csv/content"
get newfields/czech/STEROPES_CZU_Collected_samples.xlsx "$Z/14134737/files/STEROPES_CZU_Collected_samples.xlsx/content"
for i in 01 02 03; do for x in shp shx dbf prj; do get "newfields/italy/Farm_${i}_SOC.$x" "$Z/14135622/files/Farm_${i}_SOC.$x/content"; done; done
get newfields/cookfarm/GSIF_0.5-5.1.tar.gz https://cran.r-project.org/src/contrib/Archive/GSIF/GSIF_0.5-5.1.tar.gz
for f in soil_properties.csv sample_origin.csv; do get "newfields/swiss/$f" "$Z/14076347/files/$f/content"; done
# further fields
DV=https://entrepot.recherche.data.gouv.fr/api/access/datafile
while read -r site id name; do get "furtherfields/carex_$site/$name.raw" "$DV/$id"; done <<'EOF'
auzeville 134694 positions
auzeville 134690 analyses_chimiques
auzeville 134693 granulometrie_5_fractions
auzeville 134692 horizons
auzeville 134689 resistivite
epoisses 172953 positions
epoisses 172958 analyses_chimiques
epoisses 172956 granulometrie_5_fractions
epoisses 172952 horizons
epoisses 186457 resistivite
ploudaniel 163604 positions
ploudaniel 163608 analyses_chimiques
ploudaniel 163607 granulometrie_5_fractions
ploudaniel 163603 horizons
ploudaniel 172686 resistivite
crouel 163484 positions
crouel 163483 analyses_chimiques
crouel 163489 granulometrie_5_fractions
crouel 163491 horizons
crouel 172690 resistivite
EOF
GH=https://raw.githubusercontent.com/USDA-REE-ARS/cottondata/856b5707434f2a566cbc7278995e185630c146b8
for f in F013B4 F033 F111; do
  for e in shp shx dbf prj; do get "furtherfields/maricopa/${f}_SoilPhysicalAnalysis_DJH.$e" "$GH/Data/Soil/${f}_SoilPhysicalAnalysis_DJH.$e"; done
  get "furtherfields/maricopa/${f}_SoilPhysicalAnalysis_DJH.geojson" "$GH/geojson/Soil/${f}_SoilPhysicalAnalysis_DJH.geojson"
done
for o in H V; do get "furtherfields/maricopa/F033_EM38${o}_2009d079.csv" "$GH/Data/Soil/EM38/F033_EM38${o}_2009d079.csv"; done
get furtherfields/walnutcreek/Soil_properties_0.csv https://ndownloader.figshare.com/files/44356958
get furtherfields/walnutcreek/Topograhic_metrics_0.csv https://ndownloader.figshare.com/files/44356961
get furtherfields/eidc_sheep/data.zip https://data-package.ceh.ac.uk/data/f3118fa8-6bec-488b-9713-2415912b8b9e.zip
get furtherfields/caroline/LCB_Data.zip https://ndownloader.figshare.com/files/60815242

# unpack
mkdir -p geor/x && tar -xzf geor/geoR_1.9-6.tar.gz -C geor/x
for c in Denmark Northern_Ireland Turkey; do mkdir -p "sensres/$c" && (cd "sensres/$c" && unzip -q -o "../SensRes_$c.zip"); done
mkdir -p newfields/cookfarm/x && tar -xzf newfields/cookfarm/GSIF_0.5-5.1.tar.gz -C newfields/cookfarm/x GSIF/data/cookfarm.rda
mkdir -p furtherfields/carex_epoisses/x_res furtherfields/carex_ploudaniel/x_res furtherfields/eidc_sheep/x furtherfields/caroline/x
unzip -q -o furtherfields/carex_epoisses/resistivite.raw -d furtherfields/carex_epoisses/x_res
if command -v bsdtar >/dev/null; then bsdtar -xf furtherfields/carex_ploudaniel/resistivite.raw -C furtherfields/carex_ploudaniel/x_res
elif command -v 7z >/dev/null; then 7z x -y -ofurtherfields/carex_ploudaniel/x_res furtherfields/carex_ploudaniel/resistivite.raw >/dev/null
else "${PY:-python}" -c "import py7zr, sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" furtherfields/carex_ploudaniel/resistivite.raw furtherfields/carex_ploudaniel/x_res; fi
unzip -q -o furtherfields/eidc_sheep/data.zip -d furtherfields/eidc_sheep/x
unzip -q -o furtherfields/caroline/LCB_Data.zip -d furtherfields/caroline/x

# checksums (Annex A)
sha256sum -c --quiet - <<'SUMS'
9baaf9180b5d7fabc25ad09c7b2bd0d3e629b6f07f144a1c3198b24e6b2f66b2  geor/geoR_1.9-6.tar.gz
9ea8c0843b511843443d180c195a2d59a1b40e15f728681a48fd2d59e393a8a0  sensres/SensRes_Denmark.zip
91cc482d72f94312a19a2e5629e4f6fb93659c6c7863a6e3d2e46438c345c26f  sensres/SensRes_Northern_Ireland.zip
54ba305ba33cbacfcfe90803df425bda8971803fcf89c0f528c3757663c6499a  sensres/SensRes_Turkey.zip
60947f4e2593d8a3688c9e05c0647ba0f5420c0284b7308f83df1256fff25f4c  midwest/locations.csv
18c64af084fae94c0e5430dfa7701d12de072016bfa09e9f2b2ad862d19e0bdd  midwest/measurements.csv
171f2ae8ad11a8454ca46d08c8586d49d18c350092bf6ee665c2e904a1280c66  newfields/ontario/2. SOC_Database.csv
ee7bdbebe8d4153737c5f7ad71f1bed16b97072f739c47a650a4c12c745daba2  newfields/czech/STEROPES_CZU_Collected_samples.xlsx
2ab33e330c0b6ae58cc57688497d792e69593132f15905f7ceb09e23097d26a3  newfields/italy/Farm_01_SOC.dbf
9f4cccdb69c1b061c2a197aeba65c31fc6a333ac79d29f13048b13e58f9eed5e  newfields/italy/Farm_01_SOC.shp
43eb20924733d916ed65a2090c7962974f83c77bf2744418e76a0d3154b965d8  newfields/italy/Farm_02_SOC.dbf
34f460142a31e9a632c24d5114acab00f956230d9e72986b4443c333a4029569  newfields/italy/Farm_02_SOC.shp
171279a0f17d352f9ddd4bf28bd9b7b31c436196a1775194f46782540cd66fd0  newfields/italy/Farm_03_SOC.dbf
1f191d77cb732261815f6be21b2f25837694c06d0f013b57ede30791f60c6580  newfields/italy/Farm_03_SOC.shp
0d45844ccfacf0dcfe094ffc0ad841fd5af18c6309b67aafe3a30220dc3981d9  newfields/cookfarm/GSIF_0.5-5.1.tar.gz
f1fa8e427a09497ac8faf18d31c940aa00b6919929966686d939977b48346b83  newfields/swiss/soil_properties.csv
f7abd4240f47f70807d180ff355d034a49ea4badc237b9a8a61f1dd4eb6eedf7  newfields/swiss/sample_origin.csv
SUMS
(cd furtherfields && sha256sum -c --quiet "$here/sources_further.sha256")
echo "all sources present and checked"
