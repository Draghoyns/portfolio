#include <iostream>
#include "add.h"
#include "io.h"

double heightAtTime(int time){

    double g{9.8};
    double distance = g * time*time / 2.;
    return distance;
}

int readHeight(){
    int x{};
    std::cout << "Enter the height of the tower in meters: " << '\n';
    std::cin >> x;
    return x;
}


int main(){


    int height{readHeight()};

    for (int i=0;i < 6;i++){
        double h{heightAtTime(i) };
        if (height - h  > 0){
            std::cout << "At " << i << " seconds, the ball is at height: " << height - h << " meters\n";
        }
        else{
            std::cout << "At " << i << " seconds, the ball is on the ground.\n";
        }
    }

    return 0;
}
