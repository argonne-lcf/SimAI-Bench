"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use the same GPU
"""
def gen_affinity_bash_script_1(ngpus_per_process: int, gpu_selector: str) -> str:
    bash_script = [
                      "#!/bin/bash",
                      "##get the free gpus from the environment variable",
                      r'IFS="," read -ra my_free_gpus <<< "$AVAILABLE_GPUS"',
                      "# Get the RankID from different launcher",
                      "if [[ -v MPI_LOCALRANKID ]]; then",
                      "   _MPI_RANKID=$MPI_LOCALRANKID ",
                      "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                      "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                      "fi",
                      "unset EnableWalkerPartition",
                      "export ZE_FLAT_DEVICE_HIERARCHY=FLAT" if gpu_selector=="ZE_AFFINITY_MASK" else "",
                      "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" if gpu_selector=="ZE_AFFINITY_MASK" else "",
                      "# Calculate the GPUs assigned to this rank",
                      f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                      f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                      r"echo $rank_gpus $_MPI_RANKID",
                      f"export {gpu_selector}" + r"=${rank_gpus}",
                      '"$@"'
                 ]
    return "\n".join(bash_script)


"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use different GPUs
"""
def gen_affinity_bash_script_2(ngpus_per_process: int, gpu_selector: str) -> str:
   """
   the below bash script is adapted from gpu_tile_compact.sh script from aurora
   """
   bash_script = [
                    "#!/bin/bash",
                    "##get the hostname",
                    "hname=$(hostname)",
                    "##get the free gpus from the environment variable",
                    r'IFS="," read -ra my_free_gpus <<< "${AVAILABLE_GPUS_${hname}}"',
                    "# Get the RankID from different launcher",
                    "if [[ -v MPI_LOCALRANKID ]]; then",
                    "   _MPI_RANKID=$MPI_LOCALRANKID ",
                    "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                    "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                    "fi",
                    "unset EnableWalkerPartition",
                    "export ZE_FLAT_DEVICE_HIERARCHY=FLAT" if gpu_selector == "ZE_AFFINITY_MASK" else "",
                    "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" if gpu_selector == "ZE_AFFINITY_MASK" else "",
                    "# Calculate the GPUs assigned to this rank",
                    f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                    f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                    f"export {gpu_selector}"+r"=${rank_gpus}",
                    '"$@"'
                ]
   return "\n".join(bash_script)