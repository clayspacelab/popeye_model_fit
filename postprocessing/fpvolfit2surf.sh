#given prf fits in fmriprep volume space, project back to surface


while getopts "d:s:" opt
do
   case "$opt" in
      d ) DATAVOL="$OPTARG" ;;
      s ) SURF_PREFIX="$OPTARG" ;;
   esac
done


if [ -z "$DATAVOL" ] || [ -z "$SURF_PREFIX" ]
then
   echo "Some or all of the parameters are empty";
   exit 1;
fi

DATADIR="$(dirname -- "$DATAVOL")" ; 
DATAFILE="$(basename -- "$DATAVOL")"	

surf_base=${SURF_PREFIX}_hemi-%s_midthickness.surf.gii

echo "Projecting $DATAFILE to surface..."

hemis=('L' 'R')

for h in ${hemis[@]};do

	#convert prf map to gifti

	DATAOUT="$DATADIR/${DATAFILE%%.*}_hemi-${h}.func.gii"
	surf_file=$(printf "$surf_base" "$h")	

	#project RF map to surface
	wb_command -volume-to-surface-mapping $DATAVOL $surf_file $DATAOUT -enclosing

	#run fix pol maps
	#MICHAEL TO ADD

	#run wb_command -set-map-names
	#MICHAEL TO ADD	

done
