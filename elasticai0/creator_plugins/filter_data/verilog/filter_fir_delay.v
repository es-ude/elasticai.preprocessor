//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.10.2024 14:16:08
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR Delay-Line Filter
// Target Devices:  ASIC
//                  FPGA
// Tool Versions:   1v0
// Description:     Performing a delay (FIR) filter
// Processing:      Data applied on posedge clk
// Dependencies:    None
//
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  LENGTH --> Length of used taps (= FIR filter order)
//////////////////////////////////////////////////////////////////////////////////


module FIR_DELAY#(
    parameter BITWIDTH = 6'd8,
    parameter LENGTH = 10'd8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    RING_BUFFER#(BITWIDTH, LENGTH) BUFFER(
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_SHIFT(DO_CALC),
        .DATA_IN(DATA_IN),
        .DATA_OUT(DATA_OUT),
        .DATA_BUF(),
        .DVALID(DVALID)
    );
endmodule
