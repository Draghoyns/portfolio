#include <iostream>
#include "add.h"
#include "io.h"



int main(){

    double x {readDouble()};
    double y {readDouble()};

    readOperation(x,y);

    return 0;
}
