
###
### HLS_HOOKS commands are called from within the Vitis HLS flow
###
namespace eval ::HLS_HOOKS {
  ### compilation preprocessing
  proc compile_preprocess_required {filepath} {
    # return true/false to indicate if compile_preprocess should be called for $filepath
    return false
  }
  proc compile_preprocess {filepath} {
    # this example does not do any processing of the file content
    set fh [open $filepath r]
    set result [read $fh]
    close $fh
    # return pre-processed file contents
    return $result
  }

  ### opt (post-link)
  proc opt_required {} {
    # return true/false to indicate if opt should be called within csynth_design

    ### Exmple of version check
    if { 0 } {
      set version [version -short]
      regexp {^(\d+\.\d+).*} $version -> version; # remove trailing patch version
      if { $version < 2020.3 } {
        puts "WARNING: custom opt command is not supported by HLS version $version"
        return false
      }
    }

    if { [info exists ::LLVM_CUSTOM_CMD] && $::LLVM_CUSTOM_CMD ne "" } {
      return true
    }
    return false
  }
  proc opt {opt input output} {
    # check for user defined opt/input/output variables
    if { [info exists ::LLVM_CUSTOM_OPT ] && $::LLVM_CUSTOM_OPT ne "" } {
      set opt $::LLVM_CUSTOM_OPT
    }
    if { [info exists ::LLVM_CUSTOM_INPUT ] && $::LLVM_CUSTOM_INPUT ne "" } {
      set input $::LLVM_CUSTOM_INPUT
    }
    if { [info exists ::LLVM_CUSTOM_OUTPUT ] && $::LLVM_CUSTOM_OUTPUT eq "" } {
      set output $::LLVM_CUSTOM_OUTPUT
    }

    # set local (not global ::) LLVM_CUSTOM_* variables to be referenced in LLVM_CUSTOM_CMD
    set LLVM_CUSTOM_OPT $opt
    set LLVM_CUSTOM_INPUT $input
    set LLVM_CUSTOM_OUTPUT $output

    # run LLVM_CUSTOM_CMD
    puts "INFO: \[HLS_HOOKS::opt\] Using ::LLVM_CUSTOM_CMD: $::LLVM_CUSTOM_CMD"
    puts "INFO: \[HLS_HOOKS::opt\] Running ::LLVM_CUSTOM_CMD: [eval concat $::LLVM_CUSTOM_CMD]"

    #exec -ignorestderr /home/mlevental/dev_projects/Xilinx/Vitis_HLS/2021.2/lnx64/tools/clang-3.9-csynth/bin/opt XXX_LL_FILE_XXX -o XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.5.5.user.bc
    #run_link_or_opt -out XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.5.6.user.bc -args "XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.4.m2.bc XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.5.5.user.bc"
    #run_link_or_opt -opt -out XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.6.user.bc -args "XXX_DIR_XXX/proj/solution1/.autopilot/db/a.g.ld.5.6.user.bc -hls-top-function-name=wrapper -internalize-public-api-list=forward"
    eval exec -ignorestderr $::LLVM_CUSTOM_CMD

    # return output file name, if blank opt results will not be used
    return $LLVM_CUSTOM_OUTPUT
  }

  ### location of local llvm build
  proc get_llvm_bin {} {
    # return path to bin directory
    if { [info exists ::env(XILINX_OPEN_SOURCE_LLVM_BUILD_PATH)] } {
      return [file join $::env(XILINX_OPEN_SOURCE_LLVM_BUILD_PATH) bin]
    }
    # return empty string if no local build
    return ""
  }

  ### project/solution management hooks

  ### hook called at end of HLS open_project command
  proc open_project {reset} {
    # example of reset code that deletes some custom files in the solution directories
    if { 0 && $reset } {
      set directory [get_project -directory]
      foreach solution [get_project -solutions] {
        file delete -force [file join $directory $solution my_custom_files]
      }
    }
  }

  ### hook called at end of HLS open_solution command
  proc open_solution {reset} {
    # example of reset code that deletes some custom files in the solution directory
    if { 0 && $reset } {
      set directory [get_solution -directory]
      file delete -force [file join $directory my_custom_files]
    }
  }
}; # end HLS_HOOKS namespace
