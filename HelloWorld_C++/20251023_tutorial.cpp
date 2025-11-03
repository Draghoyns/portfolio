#include <iostream>

int doubleNumber(int n){
    return n * 2;
}

int tuto(){

    std::cout << "Enter an integer: ";
    int x{};
    std::cin >> x;

    int doubled = doubleNumber(x);

    std::cout << "Doubled number: " << doubled << '\n';

    return 0;
}



