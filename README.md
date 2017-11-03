# XOR example for MSP430

Machine learning example running on MSP430, _not_ intermittent-safe. It uses _fixed-points_.

## How to use

Just clone/download the repository and import the project in CCS, build and run.

## Program

The main function _dynamically_ allocates an Artificial Neural Network (ANN) using network parameters provided in `database/xor_trained.h`. This header file is constructed from `database/xor_trained.net`.

After the ANN has been allocated, 4 input tests are fed to the network, and the resulting inference is compared with the expected output. Input and output vectors are provided in `database/xor_test.h`.

## Known issues

#### Compiler version
The project was built and tested with the latest MSP430 compiler version (17.9.0). If you don't have it, please download it from CCS by going to `Help > Install New Software` and typing "Code Generation Tools Updates" in the search bar. On Linux distributions you may have to run CCS as a superuser, e.g. in Ubuntu
```bash
sudo /opt/ti/ccsv7/eclipse/ccstudio
```

#### Compiler flags
When importing the CCS project, some systems may cut out some of the compiler flags needed to build this project without errors. To make sure all your compiler flags are set correctly, have a look at [this list](https://github.com/phdschooltpc/msp430-xor/blob/master/list_of_compiler_flags.md) and compare it with your project's compiler settings in `Project > Properties > CCS Build`.
