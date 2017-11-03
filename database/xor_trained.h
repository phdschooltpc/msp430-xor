#ifndef __XOR_TRAINED__
#define __XOR_TRAINED__


// FANN_FIX_2.0

#define DECIMAL_POINT                        13
#define NUM_LAYERS                           3
#define LEARNING_RATE                        0.700000
#define CONNECTION_RATE                      1.000000
#define NETWORK_TYPE                         0
#define LEARNING_MOMENTUM                    0.000000
#define TRAINING_ALGORITHM                   2
#define TRAIN_ERROR_FUNCTION                 1
#define TRAIN_STOP_FUNCTION                  1
#define CASCADE_OUTPUT_CHANGE_FRACTION       0.010000
#define QUICKPROP_DECAY                      -0.000100
#define QUICKPROP_MU                         1.750000
#define RPROP_INCREASE_FACTOR                1.200000
#define RPROP_DECREASE_FACTOR                0.500000
#define RPROP_DELTA_MIN                      0.000000
#define RPROP_DELTA_MAX                      50.000000
#define RPROP_DELTA_ZERO                     0.100000
#define CASCADE_OUTPUT_STAGNATION_EPOCHS     12
#define CASCADE_CANDIDATE_CHANGE_FRACTION    0.010000
#define CASCADE_CANDIDATE_STAGNATION_EPOCHS  12
#define CASCADE_MAX_OUT_EPOCHS               150
#define CASCADE_MIN_OUT_EPOCHS               50
#define CASCADE_MAX_CAND_EPOCHS              150
#define CASCADE_MIN_CAND_EPOCHS              50
#define CASCADE_NUM_CANDIDATE_GROUPS         2
#define BIT_FAIL_LIMIT                       82
#define CASCADE_CANDIDATE_LIMIT              8192000
#define CASCADE_WEIGHT_MULTIPLIER            3277
#define CASCADE_ACTIVATION_FUNCTIONS_COUNT   10
#define CASCADE_ACTIVATION_FUNCTION_1        3
#define CASCADE_ACTIVATION_FUNCTION_2        5
#define CASCADE_ACTIVATION_FUNCTION_3        7
#define CASCADE_ACTIVATION_FUNCTION_4        8
#define CASCADE_ACTIVATION_FUNCTION_5        10
#define CASCADE_ACTIVATION_FUNCTION_6        11
#define CASCADE_ACTIVATION_FUNCTION_7        14
#define CASCADE_ACTIVATION_FUNCTION_8        15
#define CASCADE_ACTIVATION_FUNCTION_9        16
#define CASCADE_ACTIVATION_FUNCTION_10       17
#define CASCADE_ACTIVATION_STEEPNESSES_COUNT 4
#define CASCADE_ACTIVATION_STEEPNESS_1       2048
#define CASCADE_ACTIVATION_STEEPNESS_2       4096
#define CASCADE_ACTIVATION_STEEPNESS_3       6144
#define CASCADE_ACTIVATION_STEEPNESS_4       8192
#define LAYER_SIZE_1                         3
#define LAYER_SIZE_2                         4
#define LAYER_SIZE_3                         2

fann_type neurons[][3] = {
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {3, 5, 8192},
    {3, 5, 8192},
    {3, 5, 8192},
    {0, 5, 8192},
    {4, 5, 8192},
    {0, 5, 8192}
};

fann_type connections[][2] = {
    {0, 3612},
    {1, -6063},
    {2, -13542},
    {0, 14588},
    {1, 17502},
    {2, 15447},
    {0, -26736},
    {1, -22053},
    {2, 17389},
    {3, 13729},
    {4, 25001},
    {5, 23357},
    {6, -10096}
};


#endif // __XOR_TRAINED__
