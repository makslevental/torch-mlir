#
# Copyright 2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create a project
open_project -reset proj

# Add design files
add_files _forward.cpp
add_files -blackbox [ glob ./jsons/*.json ]

# Set the top-level function
# set ::PROJECT_PATH_MINE [pwd]
# set ::LLVM_CUSTOM_OUTPUT [pwd]/forward.ll
set ::LLVM_CUSTOM_CMD {$LLVM_CUSTOM_OPT forward_final.ll -o $LLVM_CUSTOM_OUTPUT}

set_top forward

# ########################################################
# Create a solution
open_solution -reset solution1
# Define technology and clock rate
set_part  {xcvu9p-flga2104-2-i}
create_clock -period "200MHz"

csynth_design

exit
