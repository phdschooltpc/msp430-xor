#ifndef __XOR_TEST__
#define __XOR_TEST__

#include <stdint.h>


uint8_t num_data = 4;
uint8_t num_input = 2;
uint8_t num_output = 1;

// Fixed-point precision: 13 fractional bits
// 8192 is 1 in fixed-point (19.13)
// -8192 is -1 in fixed-point (19.13)

#pragma PERSISTENT(input) // Place data in FRAM
fann_type input[4][2] = {
    {-8192, -8192},
    {-8192, 8192},
    {8192, -8192},
    {8192, 8192}
};

#pragma PERSISTENT(output) // Place data in FRAM
fann_type output[4][1] = {
    {-8192},
    {8192},
    {8192},
    {-8192}
};


#endif // __XOR_TEST__
