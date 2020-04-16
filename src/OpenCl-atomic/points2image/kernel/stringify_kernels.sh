#!/bin/bash

# kernel files
KERNEL_IN=points2image.cl.c

echo " "
echo "Stringified input kernel-files: "
echo $KERNEL_IN

# output file
KERNEL_OUT=kernel.h

echo " "
echo "Stringified output file: $KERNEL_OUT"
echo " "

echo "#ifndef EPHOS_KERNEL_H" > $KERNEL_OUT
echo "#define EPHOS_KERNEL_H" >> $KERNEL_OUT
echo "const char* points2image_kernel_source_code =" >> $KERNEL_OUT
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $KERNEL_IN >> $KERNEL_OUT
echo ";" >> $KERNEL_OUT
echo "#endif" >> $KERNEL_OUT
