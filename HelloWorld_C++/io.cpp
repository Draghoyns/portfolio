#include <iostream>
#include "io.h"

int readInt(){
    int x{};
    std::cout << "Insert a number: ";
    std::cin >> x;
    return x;
}

double readDouble(){
    double x{};
    std::cout << "Enter a double value: ";
    std::cin >> x;
    return x;
}


char readChar(){
    char x{};
    std::cout << "Enter a single character: ";
    std::cin >> x;
    return x;
}

void readOperation(double x,double y){
    char op{};
    std::cout << "Enter +, -, *, or /: ";
    std::cin >> op;
    if (op == '+'){
        std::cout << x << " " << op << " "<< y << " is " << x+y <<"\n";
    }
    else if (op == '-'){
        std::cout << x << " " << op << " "<< y << " is " << x-y <<"\n";
    }
    else if (op == '*'){
        std::cout << x << " " << op << " "<< y << " is " << x*y <<"\n";
    }
    else if (op == '/'){
        std::cout << x << " " << op << " "<< y << " is " << x/y <<"\n";
    }
    

}

void writeAnswer(int x){
    std::cout << "Which has ASCII code "<< x << "\n";

}
