#!/bin/bash

# kernel-header files
# none

# device source-code folder
KERNEL_DIR="./ocl/device"

# kernel files
IN_KERNEL1=$KERNEL_DIR/ocl_initRadiusSearch.cl.c
IN_KERNEL2=$KERNEL_DIR/ocl_parallelRadiusSearch.cl.c

echo " "
echo "Stringified input kernel-files: "
echo $IN_KERNEL1
echo $IN_KERNEL2

# output file
OUT=./stringify.h

echo " "
echo "Stringified output file: "
echo $OUT
echo " "

# temporal file
TMP=$KERNEL_DIR/stringify_tmp

echo "#ifndef STRINGIFY_H" >>$TMP
echo "#define STRINGIFY_H" >>$TMP

echo "const char *initRadiusSearch_ocl_krnl =" >>$TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL1 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL2 >> $TMP
echo ";" >>$TMP

echo "#endif // End of STRINGIFY_H" >>$TMP

# remove "#include" lines
grep -v '#include' $TMP > $OUT

