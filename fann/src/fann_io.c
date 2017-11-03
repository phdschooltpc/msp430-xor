/*
 *******************************************************************************
 * fann_io.c
 *
 * FANN training data reading function ported for embedded devices.
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
#include <stdint.h>

#include "config.h"
#include "fann.h"
#include "fann_data.h"

#include "xor_trained.h"


/**
 * INTERNAL FUNCTION
 *
 * The ANN is created from constant values contained in 
 * database/<example>_trained.h
 * where <example> is the subject example (e.g. xor, thyroid, etc.)
 */
struct fann *fann_create_msp430()
{
    uint8_t input_neuron;
    uint8_t i;
    uint8_t num_connections;
    uint8_t tmp_val;

    struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;
    fann_type *weights;
    struct fann_layer *layer_it;
    struct fann *ann = NULL;

    /* Layer Sizes. */
    uint8_t layer_size_store[] = {
        LAYER_SIZE_1, 
        LAYER_SIZE_2, 
        LAYER_SIZE_3
    };
    uint8_t layer_size;
    
    /* Allocate network. */
    // WARNING: dynamic allocation!
    ann = fann_allocate_structure((uint8_t) NUM_LAYERS);
    if(ann == NULL) {
        return NULL;
    }

    /* Assign parameters. */

    ann->decimal_point = DECIMAL_POINT;
    ann->multiplier = 1 << DECIMAL_POINT;
    ann->learning_rate = LEARNING_RATE;
    ann->connection_rate = CONNECTION_RATE;

    ann->network_type = (enum fann_nettype_enum) NETWORK_TYPE;
    ann->learning_momentum = LEARNING_MOMENTUM;
    ann->training_algorithm = (enum fann_train_enum) TRAINING_ALGORITHM;
    ann->train_error_function = (enum fann_errorfunc_enum) TRAIN_ERROR_FUNCTION;
    ann->train_stop_function = (enum fann_stopfunc_enum) TRAIN_STOP_FUNCTION;

    ann->cascade_output_change_fraction = CASCADE_OUTPUT_CHANGE_FRACTION;
    ann->quickprop_decay = QUICKPROP_DECAY;
    ann->quickprop_mu = QUICKPROP_MU;
    ann->rprop_increase_factor = RPROP_INCREASE_FACTOR;
    ann->rprop_decrease_factor = RPROP_DECREASE_FACTOR;
    ann->rprop_delta_min = RPROP_DELTA_MIN;
    ann->rprop_delta_max = RPROP_DELTA_MAX;
    ann->rprop_delta_zero = RPROP_DELTA_ZERO;
    ann->cascade_output_stagnation_epochs = CASCADE_OUTPUT_STAGNATION_EPOCHS;
    ann->cascade_candidate_change_fraction = CASCADE_CANDIDATE_CHANGE_FRACTION;
    ann->cascade_candidate_stagnation_epochs = CASCADE_CANDIDATE_STAGNATION_EPOCHS;
    ann->cascade_max_out_epochs = CASCADE_MAX_OUT_EPOCHS;
    ann->cascade_min_out_epochs = CASCADE_MIN_OUT_EPOCHS;
    ann->cascade_max_cand_epochs = CASCADE_MAX_CAND_EPOCHS;
    ann->cascade_min_cand_epochs = CASCADE_MIN_CAND_EPOCHS;
    ann->cascade_num_candidate_groups = CASCADE_NUM_CANDIDATE_GROUPS;
    ann->bit_fail_limit = BIT_FAIL_LIMIT;
    ann->cascade_candidate_limit = CASCADE_CANDIDATE_LIMIT;
    ann->cascade_weight_multiplier = CASCADE_WEIGHT_MULTIPLIER;

    ann->cascade_activation_functions_count = CASCADE_ACTIVATION_FUNCTIONS_COUNT;
    // WARNING: dynamic allocation!
    ann->cascade_activation_functions = (enum fann_activationfunc_enum *) realloc(
        ann->cascade_activation_functions,
        ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum)
    );
    if (ann->cascade_activation_functions == NULL) {
        // fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(ann);
        return NULL;
    }
#ifdef DEBUG_MALLOC
    printf("Re-allocated %u bytes for activation functions.\n",
            ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
#endif // DEBUG_MALLOC

    // TODO: [not urgent] dynamically parse activation functions (macro)
    uint8_t cascade_activation_functions[]= { 
        CASCADE_ACTIVATION_FUNCTION_1,
        CASCADE_ACTIVATION_FUNCTION_2,
        CASCADE_ACTIVATION_FUNCTION_3,
        CASCADE_ACTIVATION_FUNCTION_4,
        CASCADE_ACTIVATION_FUNCTION_5, 
        CASCADE_ACTIVATION_FUNCTION_6,
        CASCADE_ACTIVATION_FUNCTION_7,
        CASCADE_ACTIVATION_FUNCTION_8, 
        CASCADE_ACTIVATION_FUNCTION_9, 
        CASCADE_ACTIVATION_FUNCTION_10
    };

    for (i = 0; i < ann->cascade_activation_functions_count; i++) {
        ann->cascade_activation_functions[i] = (enum fann_activationfunc_enum) cascade_activation_functions[i];
    }

    ann->cascade_activation_steepnesses_count = CASCADE_ACTIVATION_STEEPNESSES_COUNT;
    // WARNING: dynamic allocation!
    ann->cascade_activation_steepnesses = (fann_type *) realloc(
        ann->cascade_activation_steepnesses,
        ann->cascade_activation_steepnesses_count * sizeof(fann_type)
    );
    if (ann->cascade_activation_steepnesses == NULL) {
        //fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(ann);
        return NULL;
    }
#ifdef DEBUG_MALLOC
    printf("Re-allocated %u bytes for activation steepnesses.\n",
            ann->cascade_activation_steepnesses_count * sizeof(fann_type));
#endif // DEBUG_MALLOC

    // TODO: [not urgent] dynamically parse activation steepnesses (macro)
    uint16_t cascade_activation_steepnesses[] = {
        CASCADE_ACTIVATION_STEEPNESS_1,
        CASCADE_ACTIVATION_STEEPNESS_2,
        CASCADE_ACTIVATION_STEEPNESS_3,
        CASCADE_ACTIVATION_STEEPNESS_4
    };

    for (i = 0; i < ann->cascade_activation_steepnesses_count; i++) {
        ann->cascade_activation_steepnesses[i] = cascade_activation_steepnesses[i];
    }

#ifdef FIXEDFANN
    fann_update_stepwise(ann);
#endif // FIXEDFANN

#ifdef DEBUG_INIT
    printf("Creating network with %d layers\n", NUM_LAYERS);
    printf("Input\n");
#endif // DEBUG_INIT

    i = 0;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {

        layer_size = layer_size_store[i++];
        if (layer_size == 0) {
            fann_destroy(ann);
            return NULL;
        }

        /* we do not allocate room here, but we make sure that
         * last_neuron - first_neuron is the number of neurons */
        layer_it->first_neuron = NULL;
        layer_it->last_neuron = layer_it->first_neuron + layer_size;
        ann->total_neurons += layer_size;
#ifdef DEBUG_INIT
        if (ann->network_type == FANN_NETTYPE_SHORTCUT && layer_it != ann->first_layer) {
            printf("  layer       : %d neurons, 0 bias\n", layer_size);
        } else {
            printf("  layer       : %d neurons, 1 bias\n", layer_size - 1);
        }
#endif // DEBUG_INIT
    }

    ann->num_input = (unsigned int) (ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
    ann->num_output = (unsigned int) ((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
    
    if (ann->network_type == FANN_NETTYPE_LAYER) {
        // One too many (bias) in the output layer
        ann->num_output--;
    }

    // WARNING: dynamic allocation!
    fann_allocate_neurons(ann);
    if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
        fann_destroy(ann);
        return NULL;
    }
    
    i = 0;

    last_neuron = (ann->last_layer - 1)->last_neuron;
    for (neuron_it = ann->first_layer->first_neuron; neuron_it != last_neuron; neuron_it++) {
        num_connections = neurons[i][0];
        tmp_val = neurons[i][1];
        neuron_it->activation_steepness = neurons[i][2];
        i++;

        neuron_it->activation_function = (enum fann_activationfunc_enum) tmp_val;
        neuron_it->first_con = ann->total_connections;
        ann->total_connections += num_connections;
        neuron_it->last_con = ann->total_connections;
    }

    // WARNING: dynamic allocation!
    fann_allocate_connections(ann);
    if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
        fann_destroy(ann);
        return NULL;
    }

    connected_neurons = ann->connections;
    weights = ann->weights;
    first_neuron = ann->first_layer->first_neuron;

    for (i = 0; i < ann->total_connections; i++) {
        input_neuron = connections[i][0];
        weights[i] = connections[i][1];
        connected_neurons[i] = first_neuron + input_neuron;
    }

    return ann;
}


/**
 * Create network from header file.
 */
FANN_EXTERNAL struct fann *FANN_API fann_create_from_header()
{
    struct fann *ann;

    ann = fann_create_msp430();

    return ann;
}
