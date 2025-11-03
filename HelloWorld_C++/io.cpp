#include <iostream>
#include "io.h"

int readNumber(){
    int x{};
    std::cout << "Insert a number: ";
    std::cin >> x;
    return x;
}

void writeAnswer(int x){
    std::cout << "The answer is: " << x << "\n";

}
