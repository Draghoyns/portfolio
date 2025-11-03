#include <iostream>


int helloWorld() {
    std::cout << "Hello, World!\n";

    std::cout << "Let's try the different initilizations:\n";
    
    [[maybe_unused]] int a;
    int b = 5;
    int c(7);
    int d{8};
    int e{};

    std::cout << "We know how these behave:" << "b: " << b << ", c: " << c << ", d: " << d << '\n';

    std::cout << "But what about a and e?" << '\n';
    // std::cout << "a: " << a <<std::endl;
    std::cout << "e: " << e << '\n';
    std::cout << "The behavior of 'a' is undefined since it is uninitialized." << '\n';
    std::cout << "Let's initialize 'a' then!" << '\n' << "Provide a value for a: \n";
    std::cin >> a;
    std::cout << "Now a: " << a << '\n';
    


    return 0;
    // toggle comment : cmd + :
}
