#include "add.h"
#include "io.h"



int main(){

    int x = readNumber();
    int y = readNumber();

    writeAnswer(add(x,y));

    return 0;
}
