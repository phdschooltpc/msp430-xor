/*
 *******************************************************************************
 * fann_train_data.c
 *
 * FANN testing data reading functions ported for embedded devices.
 *
 * Created on: Oct 23, 2017
 *    Authors: Dimitris Patoukas, Carlo Delle Donne
 *******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>

#include "config.h"
#include "fann.h"


/**
 * INTERNAL FUNCTION
 *
 * Test data is created from constant values contained in 
 * database/<example>_test.h
 * where <example> is the subject example (e.g. xor, thyroid, etc.)
 *
 * WARNING: dynamically allocated test database! Do not use.
 */
// struct fann_train_data *fann_read_test_msp430()
// {
//     uint8_t i, j;
//     struct fann_train_data *data;

//     data = fann_create_train(num_data, num_input, num_output);
//     if(data == NULL) {
//         return NULL;
//     }
    
//     for(i = 0; i != num_data; i++) {
//         for(j = 0; j != num_input; j++) {
//             data->input[i][j] = input[i][j];
//         }

//         for(j = 0; j != num_output; j++) {
//             data->output[i][j] = output[i][j];
//         }
//     }

//     return data;
// }

/*
 * Deallocates the train data structure.
 */
FANN_EXTERNAL void FANN_API fann_destroy_train(struct fann_train_data *data)
{
    if(data == NULL)
        return;
    if(data->input != NULL)
        fann_safe_free(data->input[0]);
    if(data->output != NULL)
        fann_safe_free(data->output[0]);
    fann_safe_free(data->input);
    fann_safe_free(data->output);
    fann_safe_free(data);
}

/*
 * Creates an empty set of training data
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train(unsigned int num_data, unsigned int num_input, unsigned int num_output)
{
    fann_type *data_input, *data_output;
    unsigned int i;

    // WARNING: dynamic allocation!
    struct fann_train_data* data = (struct fann_train_data*) malloc(sizeof(struct fann_train_data));

    if(data == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

    // fann_init_error_data((struct fann_error *) data);

    data->num_data = num_data;
    data->num_input = num_input;
    data->num_output = num_output;

    data->input = (fann_type **) calloc(num_data, sizeof(fann_type *));
    if(data->input == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data->output = (fann_type **) calloc(num_data, sizeof(fann_type *));
    if(data->output == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data_input = (fann_type *) calloc(num_input * num_data, sizeof(fann_type));
    if(data_input == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    data_output = (fann_type *) calloc(num_output * num_data, sizeof(fann_type));
    if(data_output == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_train(data);
        return NULL;
    }

    for(i = 0; i != num_data; i++) {
        data->input[i] = data_input;
        data_input += num_input;
        data->output[i] = data_output;
        data_output += num_output;
    }
    
    return data;
}


/**
 * Create test data from header file.
 */
FANN_EXTERNAL struct fann_train_data *FANN_API fann_create_tests_from_header()
{
    struct fann_train_data *data;

    // data = fann_read_test_msp430();

    return data;
}


FANN_EXTERNAL unsigned int FANN_API fann_length_train_data(struct fann_train_data *data)
{
    return data->num_data;
}
