#!/bin/bash

	#circa AmberTools22 this is the new env setup
source /Users/kehuang/amber24/amber.sh

read -e -p "Input PDB: " pdb
#read -p "Mol. shorthand: " mabv

	#output files
prmtop=${pdb}.prmtop
inpcrd=${pdb}.inpcrd

	#salt concentration in M
m2=0.15	#Na+
m3=0.15	#Cl-

	#make box with neutral salt conditions- note the Zn ions, which are counted in the overall charge of the system
cat>tleap.scrpt<<EOF
source leaprc.DNA.OL15
source leaprc.water.tip3p
source leaprc.protein.ff19SB
loadamberparams frcmod.ionsjc_tip3p
loadoff ../../../../../marina_ZnFF.lib
loadamberparams ../../../../../marina_ZnFF.frcmod
m1 = loadpdb ${pdb}
solvateOct m1 TIP3PBOX 12.5 iso 1.0
saveamberparm m1 test.prmtop test.inpcrd
quit
EOF

wait

tleap -f tleap.scrpt

	#clean up extra stuff
rm *.scrpt *.dat *.*~ leap.log

	#calc volume of no ion box
cat >ptraj.vol.in<<EOF
parm test.prmtop
trajin test.inpcrd
volume test out tvol.dat
EOF

cpptraj -i ptraj.vol.in

tvol=`awk 'NR>1{print $2}' tvol.dat` 

fvol=`python -c "print (round($tvol))"`

echo $fvol

nV=`echo "scale=50; $fvol*10^-27" | bc`

# Get Cl [m3]
fcon1=$(echo "scale=50; $nV*$m3*6.022*10^23" | bc)
clcon=$(python3 -c "print(int(round(float('$fcon1'))))")

# Get Na [m2]
fcon2=$(echo "scale=50; $nV*$m2*6.022*10^23" | bc)
nacon=$(python3 -c "print(int(round(float('$fcon2'))))")

echo '[Cl]=$clcon [Na]=$nacon'

	#box 2 with salt now
cat >PU1.scrpt<<EOF
source leaprc.DNA.OL15
source leaprc.water.tip3p
source leaprc.protein.ff19SB
loadamberparams frcmod.ionsjc_tip3p
loadoff ../../../../../marina_ZnFF.lib
loadamberparams ../../../../../marina_ZnFF.frcmod
m1 = loadpdb ${pdb}
solvateOct m1 TIP3PBOX 12.5 iso 1.0
addIons m1 Na+ 0
addIons m1 Cl- 0
addIonsRand m1 Na+ ${nacon} 
addIonsRand m1 Cl- ${clcon} 
saveamberparm m1 ${prmtop} ${inpcrd}
savepdb m1 check_$prmtop.pdb
quit
EOF

tleap -f PU1.scrpt

wait
	
	#Clean up files

rm *.dat ptraj.vol.in *.*~ leap.log *.scrpt test.prmtop test.inpcrd

