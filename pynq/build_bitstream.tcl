
# Vivado Tcl Script to generate PYNQ-Z2 Bitstream for SVM Accelerator

# 1. Set Project Config
set project_name "svm_pynq"
set part_name "xc7z020clg400-1"
set output_dir "pynq_output"

# Close any open project
if { [current_project -quiet] != "" } {
    close_project
}

# Create Output Directory
file mkdir $output_dir

# Create Project
create_project -force $project_name ./$project_name -part $part_name

# 2. Add RTL Files
add_files -norecurse {
    ../rtl/linear_svm.v
    ../rtl/axi_lite_wrapper.v
}

update_compile_order -fileset sources_1

# 3. Create Block Design
create_bd_design "design_1"

# 4. Add Zynq Processing System
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

# Apply PYNQ-Z2 Board Preset (if available) or minimal config
# Assuming board files are not present, simple config:
set_property -dict [list TOPOLOGY {Zynq 7020} PCW_FPGA0_PERIPHERAL_FREQMHZ {100} PCW_USE_S_AXI_HP0 {0} PCW_USE_M_AXI_GP0 {1}] [get_bd_cells processing_system7_0]

# Enable FCLK_CLK0 (100MHz) and M_AXI_GP0
# Run Block Automation to fix basic settings
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

# 5. Add RTL Module (AXI Lite Wrapper)
create_bd_cell -type module -reference axi_lite_wrapper axi_lite_wrapper_0

# 6. Add AXI Interconnect/SmartConnect
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
endgroup

set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

# 7. Connect Everything
# Clock (FCLK_CLK0 -> 100MHz)
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_lite_wrapper_0/s_axi_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M00_ACLK]

# Reset (FCLK_RESET0_N)
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_lite_wrapper_0/s_axi_aresetn]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/M00_ARESETN]

# AXI Bus
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins axi_lite_wrapper_0/s_axi]

# Address Map
assign_bd_address

# 8. Create Wrapper and Generate Output Products
make_wrapper -files [get_files ./$project_name/$project_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./$project_name/$project_name.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

# 9. Run Synthesis and Implementation
launch_runs synth_1 -jobs 4
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 10. copy bitstream and HWH
file copy -force ./$project_name/$project_name.runs/impl_1/design_1_wrapper.bit $output_dir/svm.bit
file copy -force ./$project_name/$project_name.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh $output_dir/svm.hwh

puts "Bitstream generation complete: $output_dir/svm.bit"
