#!/bin/bash

export MODULES= "devtoolset/10", "cuda/11.4"

for module in $(MODULES); do
    module load $$module;
    echo "Module $$module loaded";
done;
echo "Modules loaded successfully";
