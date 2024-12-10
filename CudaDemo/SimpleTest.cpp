#include "SimpleTest.h"
#include <iostream>

using namespace std;

int SimpleTest::SimplePointerTest() {
	int* arr = new int[4]{ 2, 3, 4, 5 };
	cout << *(arr + 1) << endl;
	return 0;
}