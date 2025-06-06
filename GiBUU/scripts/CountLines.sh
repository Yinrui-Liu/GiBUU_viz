#!/bin/bash

# should be called in workingcode
path=`pwd`

cloc="$path/scripts/cloc-2.02.pl"
OPT=" --exclude-lang=Text"

rm -f exclude.files

##### Count all #####

$cloc --report-file=ZERO.rep --categorized=ZERO.cat --counted=ZERO.list "$path/code/"

##### Pythia 6.4 #####

# careful: no '/' at end of path for exclude file !
echo "$path/code/collisions/twoBodyReactions/Pythia64" > PYTHIA64.files

$cloc $OPT --report-file=PYTHIA64.rep --categorized=PYTHIA64.cat --list-file=PYTHIA64.files
cat PYTHIA64.files >> exclude.files

##### Fritiof, old Pythia etc. #####

echo "$path/code/collisions/twoBodyReactions/HiEnergy/ariadne_402r.f" > FRITIOF.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/fritiof_702.KG.f" >> FRITIOF.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/jetset_73.f" >> FRITIOF.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/pythia_55.f" >> FRITIOF.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/LUSTRF.orig.73.f" >> FRITIOF.files

$cloc $OPT --report-file=FRITIOF.rep --categorized=FRITIOF.cat --list-file=FRITIOF.files
cat FRITIOF.files >> exclude.files

##### Formation times ######

echo "$path/code/collisions/twoBodyReactions/HiEnergy/HandleSFREP.f" > FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/GetJetsetVec.f" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/LUSTRF.Modif.73.f" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/GetLeading.f90" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/HandleSFREP_S.f" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/MomPart.f" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/LUSTRF.switch.f" >> FORMATION.files
echo "$path/code/collisions/twoBodyReactions/HiEnergy/PYSTRF.switch.f" >> FORMATION.files

$cloc $OPT --report-file=FORMATION.rep --categorized=FORMATION.cat --list-file=FORMATION.files
cat FORMATION.files >> exclude.files

##### testruns #####

rm -f TESTRUN.files
for i in `grep -v "#" Makefile.testRun.List`;
do
    echo `realpath $i` >> TESTRUN.files
done

$cloc $OPT --report-file=TESTRUN.rep --categorized=TESTRUN.cat --list-file=TESTRUN.files
cat TESTRUN.files >> exclude.files

##### Externals #####

echo "$path/code/numerics/quadpack.f90" > EXTERNALS.files
echo "$path/code/density/ertf1_21.f" >> EXTERNALS.files
echo "$path/code/analysis/readHAFT2.f90" >> EXTERNALS.files
echo "$path/code/analysis/dls" >> EXTERNALS.files
echo "$path/code/collisions/twoBodyReactions/annihilation/um_tn.f" >> EXTERNALS.files
echo "$path/code/numerics/cernlib" >> EXTERNALS.files
echo "$path/code/collisions/twoBodyReactions/annihilation/um2.f90" >> EXTERNALS.files

$cloc $OPT --report-file=EXTERNALS.rep --exclude-list-file=exclude.files --categorized=EXTERNALS.cat --list-file=EXTERNALS.files
cat EXTERNALS.files >> exclude.files

##### Inits #####

echo "$path/code/init" > INITS.files
$cloc $OPT --report-file=INITS.rep --exclude-list-file=exclude.files --categorized=INITS.cat --list-file=INITS.files
cat INITS.files >> exclude.files

##### Analysis #####

echo "$path/code/analysis" > ANALYSIS.files
$cloc $OPT --report-file=ANALYSIS.rep --exclude-list-file=exclude.files --categorized=ANALYSIS.cat --list-file=ANALYSIS.files
cat ANALYSIS.files >> exclude.files

##### Everything left over #####

$cloc -v $OPT --list-file=ZERO.list --report-file=OTHER.rep --exclude-list-file=exclude.files --categorized=OTHER.cat  --ignored=OTHER.ignored
#$cloc $OPT --report-file=OTHER.rep --exclude-list-file=exclude.files --categorized=OTHER.cat "$path/code/"
#$cloc $OPT --by-file-by-lang --report-file=OTHER.rep --exclude-list-file=exclude.files "$path/code/"

##### Combine single report files to one overall result #####

$cloc -sum-reports --report_file=all.rep  PYTHIA64.rep FRITIOF.rep TESTRUN.rep OTHER.rep EXTERNALS.rep INITS.rep ANALYSIS.rep FORMATION.rep



#--by-file-by-lang
