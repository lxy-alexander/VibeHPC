#!/bin/bash

# =============================================================================
# Common utilities for Llama2-70b disaggregated inference scripts
# 
# This file provides:
# - Colored logging functions for cleaner output
# - Shared configuration validation
# - Helper functions used across all scripts
# =============================================================================

# Color codes for terminal output
COLOR_RESET='\033[0m'
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_GRAY='\033[0;90m'

# =============================================================================
# Logging Functions - Use these for all output
# =============================================================================

# Print an informational message (blue)
log_info() {
    echo -e "${COLOR_BLUE}ℹ${COLOR_RESET} $*"
}

# Print a success message (green checkmark)
log_success() {
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} $*"
}

# Print a warning message (yellow)
log_warning() {
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} $*"
}

# Print an error message (red X)
log_error() {
    echo -e "${COLOR_RED}✗${COLOR_RESET} $*"
}

# Print a step header (cyan, numbered)
log_step() {
    local step_num=$1
    local total_steps=$2
    shift 2
    echo ""
    echo -e "${COLOR_CYAN}[$step_num/$total_steps]${COLOR_RESET} $*"
}

# Print a sub-item (gray arrow)
log_detail() {
    echo -e "${COLOR_GRAY}  →${COLOR_RESET} $*"
}

# Print a command being executed (gray, indented)
log_command() {
    echo -e "${COLOR_GRAY}  \$ $*${COLOR_RESET}"
}

# Print a section header (with border)
log_section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

# =============================================================================
# Configuration Helpers
# =============================================================================

# Calculate total GPUs needed for a configuration
calculate_total_gpus() {
    local num_ctx=$1
    local ctx_tp=$2
    local ctx_pp=$3
    local num_gen=$4
    local gen_tp=$5
    local gen_pp=$6
    
    local ctx_gpus=$((num_ctx * ctx_tp * ctx_pp))
    local gen_gpus=$((num_gen * gen_tp * gen_pp))
    echo $((ctx_gpus + gen_gpus))
}

# Calculate number of nodes needed (assuming 4 GPUs per node)
calculate_nodes_needed() {
    local total_gpus=$1
    local gpus_per_node=${2:-4}  # Default: 4 GPUs per node (GB200/GB300)
    echo $(( (total_gpus + gpus_per_node - 1) / gpus_per_node ))
}

# =============================================================================
# Validation Functions
# =============================================================================

# Check if running in SLURM environment
check_slurm_env() {
    if [ -z "$SLURM_JOB_ID" ]; then
        log_error "Not running in SLURM environment"
        return 1
    fi
    return 0
}

# Validate that required nodes are allocated
validate_node_allocation() {
    local required_nodes=$1
    local allocated_nodes=${SLURM_JOB_NUM_NODES:-0}
    
    if [ $allocated_nodes -lt $required_nodes ]; then
        log_error "Insufficient nodes allocated"
        log_detail "Required: $required_nodes nodes"
        log_detail "Allocated: $allocated_nodes nodes"
        return 1
    fi
    return 0
}

