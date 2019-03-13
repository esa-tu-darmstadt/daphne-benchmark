#include "benchmark.h"
#include <iostream>

class kernel_name : public kernel {
public:
    virtual void read_input();
    virtual void run();
    virtual bool check_output();
    uint32_t testcases = 1;
};

void kernel_name::read_input() {
    std::cout << "reading  input \n";
}

void kernel_name::run() {
    std::cout << "running kernel" << "\n";
}

bool kernel_output::check_output() {
    std::cout << "checking output \n";
    return false;
}

kernel_name a = kernel_name();
kernel& myKernel = a;
