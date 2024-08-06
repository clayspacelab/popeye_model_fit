#!/bin/bash

args=''
for i in "$@"; do
    i="${i//\\/\\\\}"
    args="$args \"${i//\"/\\\"}\""
done

if [ "${args}" == "" ]; then
    args="/bin/bash"; 
fi

if [[ -e /dev/nvidia0 ]]; then
    nv="--nv";
fi

singularity \
    exec \
    --nv --overlay /scratch/mdd9787/popeye_pRF_greene/overlay_img/popeye_overlay.ext3:ro \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
unset -f which
source /opt/apps/lmod/lmod/init/sh
source /ext3/env.sh
conda activate prf_fitter
${args}
"
