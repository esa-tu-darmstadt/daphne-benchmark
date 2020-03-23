#!/bin/bash

# kernel files
KERNEL1_SOURCE=findMinMax.cl.c
KERNEL2_SOURCE=initTargetCells.cl.c
KERNEL3_SOURCE=firstPass.cl.c
KERNEL4_SOURCE=secondPass.cl.c
KERNEL5_SOURCE=voxelRadiusSearch.cl.c

echo " "
echo "Stringified input kernel-files: "
echo $KERNEL1_SOURCE
echo $KERNEL2_SOURCE
echo $KERNEL3_SOURCE
echo $KERNEL4_SOURCE
echo $KERNEL5_SOURCE

# output file
KERNEL_TARGET=kernel.h

echo " "
echo "Stringified output file: "
echo $KERNEL_TARGET
echo " "

echo "" > $KERNEL_TARGET
echo "#ifndef EPHOS_KERNEL_H" >> $KERNEL_TARGET
echo "#define EPHOS_KERNEL_H" >> $KERNEL_TARGET

echo "const char* voxel_grid_ocl_kernel_source =" >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL1_SOURCE >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL2_SOURCE >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL3_SOURCE >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL4_SOURCE >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL5_SOURCE >> $KERNEL_TARGET
echo ";" >> $KERNEL_TARGET

echo "#endif // EPHOS_KERNEL_H" >> $KERNEL_TARGET
