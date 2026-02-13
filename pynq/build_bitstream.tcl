# Vivado Tcl Script for SVM FPGA Project
# Target Board: PYNQ-Z2 (xc7z020clg400-1)

# 1. Configuration
# Get the absolute path to the directory containing this script
set script_path [file normalize [file dirname [info script]]]
# The RTL is in ../rtl relative to this script (which is in pynq/)
set rtl_dir [file normalize [file join $script_path ".." "rtl"]]
set output_dir [file join $script_path "output"]
set project_name "svm_pynq_project"
set project_dir [file join $script_path $project_name]
set part_name "xc7z020clg400-1"

puts "------------------------------------------------"
puts "Using RTL Directory: $rtl_dir"
puts "Output Directory:    $output_dir"
puts "------------------------------------------------"

file mkdir $output_dir

# Close any open project
if {[current_project -quiet] != ""} {
    close_project
}

# Clean previous project
if {[file exists $project_dir]} {
    puts "Removing existing project directory: $project_dir"
    file delete -force $project_dir
}

# 2. Create Project
create_project -force $project_name $project_dir -part $part_name

# 3. Add RTL Sources
add_files $rtl_dir/linear_svm.v
add_files $rtl_dir/axi_lite_wrapper.v

# 4. Create Block Design
create_bd_design "system"

# 5. Add Zynq PS
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

# Apply Automation (This connects DDR and FIXED_IO to external ports)
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

# 6. Add SVM IP
create_bd_cell -type module -reference axi_lite_wrapper axi_lite_wrapper_0

# 8. Connection Automation
set_property -dict [list PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells processing_system7_0]

# Run automation for AXI
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} Clk_slave {Auto} Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_lite_wrapper_0/s_axi} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins axi_lite_wrapper_0/s_axi]

# DYNAMIC RESET HANDLING - Only if needed, find the reset created by block automation
set reset_cell [get_bd_cells -quiet -filter {VLNV =~ "xilinx.com:ip:proc_sys_reset:*"}]

if {$reset_cell eq ""} {
    # If automation didn't create one (unlikely but possible), create it manually
    puts "Reset block not found from automation. Creating manual reset..."
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 manual_reset
    set reset_cell "manual_reset"
    
    # Connect clock and external reset input
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins $reset_cell/slowest_sync_clk]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins $reset_cell/ext_reset_in]
} else {
    # If multiple resets exist, pick the first one (usually the one connected to FCLK_CLK0)
    set reset_cell [lindex $reset_cell 0]
}

puts "Using Reset Block: $reset_cell"

# Validate and save BD
validate_bd_design
save_bd_design

# Use safer PATH-BASED access for all these commands
set bd_file "$project_dir/$project_name.srcs/sources_1/bd/system/system.bd"

# Generate all IP output products
generate_target all [get_files $bd_file]

# Export IP user files
export_ip_user_files -of_objects [get_files $bd_file] -no_script -sync -force

# Create HDL wrapper (Path-based is safer)
make_wrapper -files [get_files $bd_file] -top
add_files -norecurse $project_dir/$project_name.srcs/sources_1/bd/system/hdl/system_wrapper.v

set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# 12. Run Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# 13. Run Implementation
# Force clean run
reset_run impl_1
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# 14. Bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 15. Export
file copy -force $project_dir/$project_name.runs/impl_1/system_wrapper.bit $output_dir/svm.bit
file copy -force $project_dir/$project_name.srcs/sources_1/bd/system/hw_handoff/system.hwh $output_dir/svm.hwh

puts "Success!"
