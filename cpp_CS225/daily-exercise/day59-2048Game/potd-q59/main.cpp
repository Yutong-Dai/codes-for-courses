#include <iostream>

using namespace std;

#include "2048.cpp"

int main() {

    int arr[4][4];
    int dir;

    for(int i=0; i<4; i++) {
        cin >> arr[i][0] >> arr[i][1] >> arr[i][2] >> arr[i][3];
    }

    cin >> dir;

    run2048(arr,dir);

    for(int i=0; i<4; i++) {
      cout << arr[i][0] << " " << arr[i][1] << " " << arr[i][2] << " " << arr[i][3] << endl;
    }
}


