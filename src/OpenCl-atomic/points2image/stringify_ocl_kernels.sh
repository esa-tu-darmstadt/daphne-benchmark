#!/bin/bash

# kernel-header files
# none

# device source-code folder
KERNEL_DIR="./ocl/device"

# kernel files
IN_KERNEL=$KERNEL_DIR/ocl_kernel.cl.c

echo " "
echo "Stringified input kernel-files: "
echo $IN_KERNEL

# output file
OUT=$KERNEL_DIR/ocl_kernel.h

echo " "
echo "Stringified output file: "
echo $OUT
echo " "

# temporal file
TMP=$KERNEL_DIR/ocl_kernel_tmp
echo "" > $TMP
echo "#ifndef OCL_KERNEL_H" >>$TMP
echo "#define OCL_KERNEL_H" >>$TMP

echo "const char *points2image_ocl_krnl =" >>$TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL >> $TMP
echo ";" >>$TMP

echo "#endif // End of OCL_KERNEL_H" >>$TMP

# remove "#include" lines
grep -v '#include' $TMP > $OUT

