# GiBUU_viz
### Before building
wget --content-disposition https://gibuu.hepforge.org/downloads?f=buuinput2025.tar.gz
copy all contents to buuinput/
### Building
cd GiBUU
make buildRootTuple
make withROOT=1
### Running
find a quiet place
mkdir test; cd test
/path-to-repo/GiBUU/testRun/GiBUU.x < /path-to-repo/GiBUU/testRun/test_nu.job
