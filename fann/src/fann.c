/*
 *******************************************************************************
 * fann_io.c
 *
 * FANN utility functions ported for embedded devices.
 *
 * Created on: Oct 23, 2017
 *    Authors: Dimitris Patoukas, Carlo Delle Donne
 *******************************************************************************
 */

#include <msp430.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "fann.h"


/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
 */
struct fann *fann_allocate_structure(unsigned int num_layers)
{
    struct fann *ann;

    if (num_layers < 2) {
        return NULL;
    }

    /* allocate and initialize the main network structure */
    ann = (struct fann *) malloc(sizeof(struct fann));
    if (ann == NULL) {
        // fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
#ifdef DEBUG_MALLOC
    printf("Allocated %u bytes for ann.\n", sizeof(struct fann));
#endif // DEBUG_MALLOC

    ann->errno_f = FANN_E_NO_ERROR;
    ann->error_log = fann_default_error_log;
    ann->errstr = NULL;
    ann->learning_rate = 0.7f;
    ann->learning_momentum = 0.0;
    ann->total_neurons = 0;
    ann->total_connections = 0;
    ann->num_input = 0;
    ann->num_output = 0;
    ann->train_errors = NULL;
    ann->train_slopes = NULL;
    ann->prev_steps = NULL;
    ann->prev_train_slopes = NULL;
    ann->prev_weights_deltas = NULL;
    ann->training_algorithm = FANN_TRAIN_RPROP;
    ann->num_MSE = 0;
    ann->MSE_value = 0;
    ann->num_bit_fail = 0;
    ann->bit_fail_limit = (fann_type)0.35;
    ann->network_type = FANN_NETTYPE_LAYER;
    ann->train_error_function = FANN_ERRORFUNC_TANH;
    ann->train_stop_function = FANN_STOPFUNC_MSE;
    ann->callback = NULL;
    ann->user_data = NULL; /* User is responsible for deallocation */
    ann->weights = NULL;
    ann->connections = NULL;
    ann->output = NULL;
#ifndef FIXEDFANN
    ann->scale_mean_in = NULL;
    ann->scale_deviation_in = NULL;
    ann->scale_new_min_in = NULL;
    ann->scale_factor_in = NULL;
    ann->scale_mean_out = NULL;
    ann->scale_deviation_out = NULL;
    ann->scale_new_min_out = NULL;
    ann->scale_factor_out = NULL;
#endif

    /* variables used for cascade correlation (reasonable defaults) */
    ann->cascade_output_change_fraction = 0.01f;
    ann->cascade_candidate_change_fraction = 0.01f;
    ann->cascade_output_stagnation_epochs = 12;
    ann->cascade_candidate_stagnation_epochs = 12;
    ann->cascade_num_candidate_groups = 2;
    ann->cascade_weight_multiplier = (fann_type)0.4;
    ann->cascade_candidate_limit = (fann_type)1000.0;
    ann->cascade_max_out_epochs = 150;
    ann->cascade_max_cand_epochs = 150;
    ann->cascade_min_out_epochs = 50;
    ann->cascade_min_cand_epochs = 50;
    ann->cascade_candidate_scores = NULL;
    ann->cascade_activation_functions_count = 10;
    ann->cascade_activation_functions = (enum fann_activationfunc_enum *) calloc(
        ann->cascade_activation_functions_count,
        sizeof(enum fann_activationfunc_enum)
    );
    if (ann->cascade_activation_functions == NULL) {
        //fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        free(ann);
        return NULL;
    }

    ann->cascade_activation_functions[0] = FANN_SIGMOID;
    ann->cascade_activation_functions[1] = FANN_SIGMOID_SYMMETRIC;
    ann->cascade_activation_functions[2] = FANN_GAUSSIAN;
    ann->cascade_activation_functions[3] = FANN_GAUSSIAN_SYMMETRIC;
    ann->cascade_activation_functions[4] = FANN_ELLIOT;
    ann->cascade_activation_functions[5] = FANN_ELLIOT_SYMMETRIC;
    ann->cascade_activation_functions[6] = FANN_SIN_SYMMETRIC;
    ann->cascade_activation_functions[7] = FANN_COS_SYMMETRIC;
    ann->cascade_activation_functions[8] = FANN_SIN;
    ann->cascade_activation_functions[9] = FANN_COS;

    ann->cascade_activation_steepnesses_count = 4;
    ann->cascade_activation_steepnesses = (fann_type *) calloc(
        ann->cascade_activation_steepnesses_count,
        sizeof(fann_type)
    );
    if (ann->cascade_activation_steepnesses == NULL) {
        fann_safe_free(ann->cascade_activation_functions);
        //fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        free(ann);
        return NULL;
    }

    ann->cascade_activation_steepnesses[0] = (fann_type) 0.25;
    ann->cascade_activation_steepnesses[1] = (fann_type) 0.5;
    ann->cascade_activation_steepnesses[2] = (fann_type) 0.75;
    ann->cascade_activation_steepnesses[3] = (fann_type) 1.0;

    /* Variables for use with with Quickprop training (reasonable defaults) */
    ann->quickprop_decay = -0.0001f;
    ann->quickprop_mu = 1.75;

    /* Variables for use with with RPROP training (reasonable defaults) */
    ann->rprop_increase_factor = 1.2f;
    ann->rprop_decrease_factor = 0.5;
    ann->rprop_delta_min = 0.0;
    ann->rprop_delta_max = 50.0;
    ann->rprop_delta_zero = 0.1f;

    /* Variables for use with SARPROP training (reasonable defaults) */
    ann->sarprop_weight_decay_shift = -6.644f;
    ann->sarprop_step_error_threshold_factor = 0.1f;
    ann->sarprop_step_error_shift = 1.385f;
    ann->sarprop_temperature = 0.015f;
    ann->sarprop_epoch = 0;

    //fann_init_error_data((struct fann_error *) ann);

    /* these values are only boring defaults, and should really
     * never be used, since the real values are always loaded from a file. */
    fann_type decimal_point = 13;
    fann_type multiplier = 1 << decimal_point;

    /* allocate room for the layers */
    ann->first_layer = (struct fann_layer *) calloc(num_layers, sizeof(struct fann_layer));
    if(ann->first_layer == NULL) {
        //fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
        free(ann);
        return NULL;
    }
#ifdef DEBUG_MALLOC
    printf("Allocated %u bytes for the layers.\n", num_layers * sizeof(struct fann_layer));
#endif // DEBUG_MALLOC

    ann->last_layer = ann->first_layer + num_layers;

    return ann;
}


#ifdef FIXEDFANN

FANN_GET(unsigned int, decimal_point)
FANN_GET(unsigned int, multiplier)

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise(struct fann *ann)
{
    unsigned int i = 0;

    /* Calculate the parameters for the stepwise linear
     * sigmoid function fixed point.
     * Using a rewritten sigmoid function.
     * results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
     */
    ann->sigmoid_results[0] = fann_max((fann_type) (ann->multiplier / 200.0 + 0.5), 1);
    ann->sigmoid_results[1] = fann_max((fann_type) (ann->multiplier / 20.0 + 0.5), 1);
    ann->sigmoid_results[2] = fann_max((fann_type) (ann->multiplier / 4.0 + 0.5), 1);
    ann->sigmoid_results[3] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 4.0 + 0.5), ann->multiplier - 1);
    ann->sigmoid_results[4] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 20.0 + 0.5), ann->multiplier - 1);
    ann->sigmoid_results[5] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 200.0 + 0.5), ann->multiplier - 1);

    ann->sigmoid_symmetric_results[0] = fann_max((fann_type) ((ann->multiplier / 100.0) - ann->multiplier - 0.5),
                                                 (fann_type) (1 - (fann_type) ann->multiplier));
    ann->sigmoid_symmetric_results[1] = fann_max((fann_type) ((ann->multiplier / 10.0) - ann->multiplier - 0.5),
                                                 (fann_type) (1 - (fann_type) ann->multiplier));
    ann->sigmoid_symmetric_results[2] = fann_max((fann_type) ((ann->multiplier / 2.0) - ann->multiplier - 0.5),
                                                 (fann_type) (1 - (fann_type) ann->multiplier));
    ann->sigmoid_symmetric_results[3] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 2.0 + 0.5),
                                                 ann->multiplier - 1);
    ann->sigmoid_symmetric_results[4] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 10.0 + 0.5),
                                                 ann->multiplier - 1);
    ann->sigmoid_symmetric_results[5] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 100.0 + 1.0),
                                                 ann->multiplier - 1);

    for(i = 0; i < 6; i++)
    {
        ann->sigmoid_values[i] =
            (fann_type) (((log(ann->multiplier / (float) ann->sigmoid_results[i] - 1) *
                           (float) ann->multiplier) / -2.0) * (float) ann->multiplier);
        ann->sigmoid_symmetric_values[i] =
            (fann_type) (((log((ann->multiplier -
                             (float) ann->sigmoid_symmetric_results[i]) /
                            ((float) ann->sigmoid_symmetric_results[i] +
                             ann->multiplier)) * (float) ann->multiplier) / -2.0) *
                         (float) ann->multiplier);
    }
}
#endif


FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann)
{
    if(ann == NULL)
        return;
    fann_safe_free(ann->weights);
    fann_safe_free(ann->connections);
    fann_safe_free(ann->first_layer->first_neuron);
    fann_safe_free(ann->first_layer);
    fann_safe_free(ann->output);
    fann_safe_free(ann->train_errors);
    fann_safe_free(ann->train_slopes);
    fann_safe_free(ann->prev_train_slopes);
    fann_safe_free(ann->prev_steps);
    fann_safe_free(ann->prev_weights_deltas);
    fann_safe_free(ann->errstr);
    fann_safe_free(ann->cascade_activation_functions);
    fann_safe_free(ann->cascade_activation_steepnesses);
    fann_safe_free(ann->cascade_candidate_scores);

#ifndef FIXEDFANN
    fann_safe_free( ann->scale_mean_in );
    fann_safe_free( ann->scale_deviation_in );
    fann_safe_free( ann->scale_new_min_in );
    fann_safe_free( ann->scale_factor_in );

    fann_safe_free( ann->scale_mean_out );
    fann_safe_free( ann->scale_deviation_out );
    fann_safe_free( ann->scale_new_min_out );
    fann_safe_free( ann->scale_factor_out );
#endif

    fann_safe_free(ann);
}

/* INTERNAL FUNCTION
   Allocates room for the neurons.
 */
void fann_allocate_neurons(struct fann *ann)
{
    struct fann_layer *layer_it;
    struct fann_neuron *neurons;
    unsigned int num_neurons_so_far = 0;
    unsigned int num_neurons = 0;
    unsigned int i;

    /* all the neurons is allocated in one long array (calloc clears mem) */
    neurons = (struct fann_neuron *) calloc(ann->total_neurons, sizeof(struct fann_neuron));
    ann->total_neurons_allocated = ann->total_neurons;
    if (neurons == NULL) {
        //fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
        return;
    }
#ifdef DEBUG_MALLOC
    printf("Allocated %u bytes for neurons.\n", ann->total_neurons * sizeof(struct fann_neuron));
#endif // DEBUG_MALLOC

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        num_neurons = (unsigned int) (layer_it->last_neuron - layer_it->first_neuron);
        layer_it->first_neuron = neurons + num_neurons_so_far;
        layer_it->last_neuron = layer_it->first_neuron + num_neurons;
        num_neurons_so_far += num_neurons;
    }

    ann->output = (fann_type *) calloc(num_neurons, sizeof(fann_type));
    if (ann->output == NULL) {
        // fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
        return;
    }
}

/* INTERNAL FUNCTION
   Allocate room for the connections.
 */
void fann_allocate_connections(struct fann *ann)
{
    ann->weights = (fann_type *) calloc(ann->total_connections, sizeof(fann_type));
    if (ann->weights == NULL) {
        // fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
        return;
    }
#ifdef DEBUG_MALLOC
    printf("Allocated %u bytes for weights.\n", ann->total_connections * sizeof(fann_type));
#endif // DEBUG_MALLOC

    ann->total_connections_allocated = ann->total_connections;

    /* TODO make special cases for all places where the connections
     * is used, so that it is not needed for fully connected networks.
     */
    ann->connections = (struct fann_neuron **) calloc(
        ann->total_connections_allocated,
        sizeof(struct fann_neuron *)
    );
    if (ann->connections == NULL) {
        // fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
        return;
    }
#ifdef DEBUG_MALLOC
    printf("Allocated %u bytes for connections.\n", 
            ann->total_connections_allocated * sizeof(struct fann_neuron *));
#endif // DEBUG_MALLOC
}

FANN_EXTERNAL fann_type *FANN_API fann_run(struct fann * ann, fann_type * input)
{
    struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
    unsigned int i, num_connections, num_input, num_output;
    fann_type neuron_sum, *output;
    fann_type *weights;
    struct fann_layer *layer_it, *last_layer;
    unsigned int activation_function;
    fann_type steepness;

    /* store some variabels local for fast access */
    struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

#ifdef FIXEDFANN
    unsigned long multiplier = ann->multiplier;
    unsigned int decimal_point = ann->decimal_point; // useless?

    /* values used for the stepwise linear sigmoid function */
    fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
    fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

    fann_type last_steepness = 0;
    unsigned int last_activation_function = 0;
#else
    fann_type max_sum = 0;
#endif

    /* first set the input */
    num_input = ann->num_input;
    for (i = 0; i != num_input; i++) {
#ifdef FIXEDFANN
        if (fann_abs(input[i]) > multiplier) {
            printf
                ("Warning input number %d is out of range -%d - %d with value %d, integer overflow may occur.\n",
                 i, multiplier, multiplier, input[i]);
        }
#endif
        first_neuron[i].value = input[i];
    }
    /* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
    (ann->first_layer->last_neuron - 1)->value = multiplier;
#else
    (ann->first_layer->last_neuron - 1)->value = 1;
#endif

    last_layer = ann->last_layer;
    for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
        last_neuron = layer_it->last_neuron;
        for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
            if (neuron_it->first_con == neuron_it->last_con) {
                /* bias neurons */
#ifdef FIXEDFANN
                neuron_it->value = multiplier;
#else
                neuron_it->value = 1;
#endif
                continue;
            }

            activation_function = neuron_it->activation_function;
            steepness = neuron_it->activation_steepness;

            neuron_sum = 0;
            num_connections = neuron_it->last_con - neuron_it->first_con;
            weights = ann->weights + neuron_it->first_con;

            if (ann->connection_rate >= 1) {
                if (ann->network_type == FANN_NETTYPE_SHORTCUT) {
                    neurons = ann->first_layer->first_neuron;
                }
                else {
                    neurons = (layer_it - 1)->first_neuron;
                }

                /* unrolled loop start */
                i = num_connections & 3;    /* same as modulo 4 */
                switch (i) {
                case 3:
                    neuron_sum += fann_mult(weights[2], neurons[2].value);
                case 2:
                    neuron_sum += fann_mult(weights[1], neurons[1].value);
                case 1:
                    neuron_sum += fann_mult(weights[0], neurons[0].value);
                case 0:
                    break;
                }

                for (; i != num_connections; i += 4) {
                    neuron_sum +=
                        fann_mult(weights[i], neurons[i].value) +
                        fann_mult(weights[i + 1], neurons[i + 1].value) +
                        fann_mult(weights[i + 2], neurons[i + 2].value) +
                        fann_mult(weights[i + 3], neurons[i + 3].value);
                }
                /* unrolled loop end */

                /*
                 * for(i = 0;i != num_connections; i++){
                 * printf("%f += %f*%f, ", neuron_sum, weights[i], neurons[i].value);
                 * neuron_sum += fann_mult(weights[i], neurons[i].value);
                 * }
                 */
            }
            else {
                neuron_pointers = ann->connections + neuron_it->first_con;

                i = num_connections & 3;    /* same as modulo 4 */
                switch (i) {
                case 3:
                    neuron_sum += fann_mult(weights[2], neuron_pointers[2]->value);
                case 2:
                    neuron_sum += fann_mult(weights[1], neuron_pointers[1]->value);
                case 1:
                    neuron_sum += fann_mult(weights[0], neuron_pointers[0]->value);
                case 0:
                    break;
                }

                for (; i != num_connections; i += 4) {
                    neuron_sum +=
                        fann_mult(weights[i], neuron_pointers[i]->value) +
                        fann_mult(weights[i + 1], neuron_pointers[i + 1]->value) +
                        fann_mult(weights[i + 2], neuron_pointers[i + 2]->value) +
                        fann_mult(weights[i + 3], neuron_pointers[i + 3]->value);
                }
            }

#ifdef FIXEDFANN
            neuron_it->sum = fann_mult(steepness, neuron_sum);

            if (activation_function != last_activation_function || steepness != last_steepness) {
                switch (activation_function) {
                case FANN_SIGMOID:
                case FANN_SIGMOID_STEPWISE:
                    r1 = ann->sigmoid_results[0];
                    r2 = ann->sigmoid_results[1];
                    r3 = ann->sigmoid_results[2];
                    r4 = ann->sigmoid_results[3];
                    r5 = ann->sigmoid_results[4];
                    r6 = ann->sigmoid_results[5];
                    v1 = ann->sigmoid_values[0] / steepness;
                    v2 = ann->sigmoid_values[1] / steepness;
                    v3 = ann->sigmoid_values[2] / steepness;
                    v4 = ann->sigmoid_values[3] / steepness;
                    v5 = ann->sigmoid_values[4] / steepness;
                    v6 = ann->sigmoid_values[5] / steepness;
                    break;
                case FANN_SIGMOID_SYMMETRIC:
                case FANN_SIGMOID_SYMMETRIC_STEPWISE:
                    r1 = ann->sigmoid_symmetric_results[0];
                    r2 = ann->sigmoid_symmetric_results[1];
                    r3 = ann->sigmoid_symmetric_results[2];
                    r4 = ann->sigmoid_symmetric_results[3];
                    r5 = ann->sigmoid_symmetric_results[4];
                    r6 = ann->sigmoid_symmetric_results[5];
                    v1 = ann->sigmoid_symmetric_values[0] / steepness;
                    v2 = ann->sigmoid_symmetric_values[1] / steepness;
                    v3 = ann->sigmoid_symmetric_values[2] / steepness;
                    v4 = ann->sigmoid_symmetric_values[3] / steepness;
                    v5 = ann->sigmoid_symmetric_values[4] / steepness;
                    v6 = ann->sigmoid_symmetric_values[5] / steepness;
                    break;
                case FANN_THRESHOLD:
                    break;
                }
            }

            switch (activation_function) {
            case FANN_SIGMOID:
            case FANN_SIGMOID_STEPWISE:
                neuron_it->value =
                    (fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, 0,
                                              multiplier, neuron_sum);
                break;
            case FANN_SIGMOID_SYMMETRIC:
            case FANN_SIGMOID_SYMMETRIC_STEPWISE:
                neuron_it->value =
                    (fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6,
                                              -multiplier, multiplier, neuron_sum);
                break;
            case FANN_THRESHOLD:
                neuron_it->value = (fann_type) ((neuron_sum < 0) ? 0 : multiplier);
                break;
            case FANN_THRESHOLD_SYMMETRIC:
                neuron_it->value = (fann_type) ((neuron_sum < 0) ? -multiplier : multiplier);
                break;
            case FANN_LINEAR:
                neuron_it->value = neuron_sum;
                break;
            case FANN_LINEAR_PIECE:
                neuron_it->value = (fann_type)((neuron_sum < 0) ? 0 : (neuron_sum > multiplier) ? multiplier : neuron_sum);
                break;
            case FANN_LINEAR_PIECE_SYMMETRIC:
                neuron_it->value = (fann_type)((neuron_sum < -multiplier) ? -multiplier : (neuron_sum > multiplier) ? multiplier : neuron_sum);
                break;
            case FANN_ELLIOT:
            case FANN_ELLIOT_SYMMETRIC:
            case FANN_GAUSSIAN:
            case FANN_GAUSSIAN_SYMMETRIC:
            case FANN_GAUSSIAN_STEPWISE:
            case FANN_SIN_SYMMETRIC:
            case FANN_COS_SYMMETRIC:
                // fann_error((struct fann_error *) ann, FANN_E_CANT_USE_ACTIVATION);
                break;
            }
            last_steepness = steepness;
            last_activation_function = activation_function;
#else
            neuron_sum = fann_mult(steepness, neuron_sum);

            max_sum = 150/steepness;
            if (neuron_sum > max_sum)
                neuron_sum = max_sum;
            else if (neuron_sum < -max_sum)
                neuron_sum = -max_sum;

            neuron_it->sum = neuron_sum;

            fann_activation_switch(activation_function, neuron_sum, neuron_it->value);
#endif
        }
    }

    /* set the output */
    output = ann->output;
    num_output = ann->num_output;
    neurons = (ann->last_layer - 1)->first_neuron;
    for (i = 0; i != num_output; i++) {
        output[i] = neurons[i].value;
    }
    return ann->output;
}
