open_checkpoint {C:/Users/Ross/Downloads/neuraedge--main/neuraedge--main/cifar10_neuromorphic/outputs_spikingformer_head_accelerator/vivado/spikingformer_head_routed.dcp}
if {[llength [get_clocks ap_clk -quiet]] == 0} { create_clock -period 10 -name ap_clk [get_ports ap_clk] }
set inference_saif {C:/Users/Ross/Downloads/neuraedge--main/neuraedge--main/cifar10_neuromorphic/outputs_spikingformer_head_accelerator/activity_power/spikingformer_head_inference.saif}
set learning_saif {C:/Users/Ross/Downloads/neuraedge--main/neuraedge--main/cifar10_neuromorphic/outputs_spikingformer_head_accelerator/activity_power/spikingformer_head_learning.saif}
set inference_report {C:/Users/Ross/Downloads/neuraedge--main/neuraedge--main/cifar10_neuromorphic/outputs_spikingformer_head_accelerator/activity_power/spikingformer_head_power_inference_saif.rpt}
set learning_report {C:/Users/Ross/Downloads/neuraedge--main/neuraedge--main/cifar10_neuromorphic/outputs_spikingformer_head_accelerator/activity_power/spikingformer_head_power_learning_saif.rpt}
if {[file exists $inference_saif]} {
  reset_switching_activity [get_nets -hierarchical *]
  read_saif -strip_path tb_cifar10_spikingformer_head_axi_inference_activity/base/dut $inference_saif
  report_power -file $inference_report -hier all
} else {
  puts "Missing inference SAIF: $inference_saif"
}
if {[file exists $learning_saif]} {
  reset_switching_activity [get_nets -hierarchical *]
  read_saif -strip_path tb_cifar10_spikingformer_head_axi_learning_activity/base/dut $learning_saif
  report_power -file $learning_report -hier all
} else {
  puts "Missing learning SAIF: $learning_saif"
}
exit
