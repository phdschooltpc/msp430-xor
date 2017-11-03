/*
 *******************************************************************************
 * fann_train.c
 *
 * FANN testing functions ported for embedded devices.
 *
 * NOTE: the name 'fann_train' sounds deceiving since this file contains 
 *       functions for testing, but we wanted to keep original FANN naming 
 *       conventions, so please blame FANN authors :)
 *
 * Created on: Oct 23, 2017
 *    Authors: Dimitris Patoukas, Carlo Delle Donne
 *******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "fann.h"


/* INTERNAL FUNCTION
   Helper function to update the MSE value and return a diff which takes symmetric functions into account
*/
fann_type fann_update_MSE(struct fann *ann, struct fann_neuron* neuron, fann_type neuron_diff)
{
    float neuron_diff2;

    switch (neuron->activation_function)
    {
        case FANN_LINEAR_PIECE_SYMMETRIC:
        case FANN_THRESHOLD_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
        case FANN_ELLIOT_SYMMETRIC:
        case FANN_GAUSSIAN_SYMMETRIC:
        case FANN_SIN_SYMMETRIC:
        case FANN_COS_SYMMETRIC:
            neuron_diff /= (fann_type)2.0;
            break;
        case FANN_THRESHOLD:
        case FANN_LINEAR:
        case FANN_SIGMOID:
        case FANN_SIGMOID_STEPWISE:
        case FANN_GAUSSIAN:
        case FANN_GAUSSIAN_STEPWISE:
        case FANN_ELLIOT:
        case FANN_LINEAR_PIECE:
        case FANN_SIN:
        case FANN_COS:
            break;
    }

#ifdef FIXEDFANN
    neuron_diff2 = (neuron_diff / (float) ann->multiplier) * (neuron_diff / (float) ann->multiplier);
#else
    neuron_diff2 = (float) (neuron_diff * neuron_diff);
#endif

    ann->MSE_value += neuron_diff2;

    /*printf("neuron_diff %f = (%f - %f)[/2], neuron_diff2=%f, sum=%f, MSE_value=%f, num_MSE=%d\n", neuron_diff, *desired_output, neuron_value, neuron_diff2, last_layer_begin->sum, ann->MSE_value, ann->num_MSE); */
    if(fann_abs(neuron_diff) >= ann->bit_fail_limit)
    {
        ann->num_bit_fail++;
    }

    return neuron_diff;
}


FANN_EXTERNAL fann_type *FANN_API fann_test(struct fann *ann, fann_type * input,
                                            fann_type * desired_output)
{
    fann_type neuron_value;
    fann_type *output_begin = fann_run(ann, input);
    fann_type *output_it;
    const fann_type *output_end = output_begin + ann->num_output;
    fann_type neuron_diff;
    struct fann_neuron *output_neuron = (ann->last_layer - 1)->first_neuron;

    /* calculate the error */
    for(output_it = output_begin; output_it != output_end; output_it++)
    {
        neuron_value = *output_it;

        neuron_diff = (*desired_output - neuron_value);

        neuron_diff = fann_update_MSE(ann, output_neuron, neuron_diff);

        desired_output++;
        output_neuron++;

        ann->num_MSE++;
    }

    return output_begin;
}

/* get the mean square error.
 */
FANN_EXTERNAL float FANN_API fann_get_MSE(struct fann *ann)
{
    if(ann->num_MSE)
    {
        return ann->MSE_value / (float) ann->num_MSE;
    }
    else
    {
        return 0;
    }
}

/* reset the mean square error.
 */
FANN_EXTERNAL void FANN_API fann_reset_MSE(struct fann *ann)
{
/*printf("resetMSE %d %f\n", ann->num_MSE, ann->MSE_value);*/
    ann->num_MSE = 0;
    ann->MSE_value = 0;
    ann->num_bit_fail = 0;
}
