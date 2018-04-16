#!/usr/bin/env bash
__script_dir=$(cd "$(dirname "$0")"; pwd)
source ${__script_dir}/../../conf/env.sh

#sh -x ${__script_dir}/start_group.sh
sh -x ${__script_dir}/start_sim.sh
#sh -x ${__script_dir}/start_uniq.sh
sh -x ${__script_dir}/start_collect.sh
