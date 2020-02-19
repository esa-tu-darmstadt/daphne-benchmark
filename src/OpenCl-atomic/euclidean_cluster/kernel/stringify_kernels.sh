#!/bin/bash

# kernel-header files
# none

# device source-code folder

# kernel files


KERNEL1_SOURCE=distanceMatrix.cl.c
KERNEL2_SOURCE=radiusSearch.cl.c

echo " "
echo "Stringified input kernel-files: "
echo $KERNEL1_SOURCE
echo $KERNEL2_SOURCE

# output file
KERNEL_TARGET=kernel.h

echo "Stringified output file: "
echo $KERNEL_TARGET

echo "" > $KERNEL_TARGET
echo "#ifndef EPHOS_KERNEL_H" >> $KERNEL_TARGET
echo "#define EPHOS_KERNEL_H" >> $KERNEL_TARGET

echo "const char* radius_search_ocl_kernel_source=" >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL1_SOURCE >> $KERNEL_TARGET
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL2_SOURCE >> $KERNEL_TARGET
echo ";" >> $KERNEL_TARGET

echo "#endif // EPHOS_KERNEL_H" >>$KERNEL_TARGET


