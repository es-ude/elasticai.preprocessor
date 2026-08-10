//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	10.08.2026
// Copied on: 	    §{date_copy_created}
// Module Name:     Constant Threshold Value
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Description:     Returning a constant threshold value
// Processing:      Constant
// Dependencies:    None
//
// State:		    None
// Improvements:    None
// Parameters:      BITWIDTH    --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module CONST_THRESHOLD#(
    parameter integer BITWIDTH = 8,
    parameter integer CONST_THR = 100
)(
    output wire [BITWIDTH-'d1:0] DATA_OUT
);

    assign DATA_OUT = CONST_THR['d0+:BITWIDTH];

endmodule
