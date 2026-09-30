//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.09.2026, 09:40:41
// Copied on: 	    §{date_copy_created}
// Module Name:     Template of a linear / fully connected layer (only inference)
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:
// Dependencies:    MAC operator, multipliers
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH            --> Bitwidth of input data
//                  SIZE_INPUT          --> Number of input values in input data
//                  SIZE_OUTPUT         --> Number of output values in output data
//////////////////////////////////////////////////////////////////////////////////


module PREPROCESSOR#(
    parameter integer BITWIDTH = 8,
    parameter integer SIZE_INPUT = 4,
    parameter integer SIZE_OUTPUT = 2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [SIZE_INPUT* BITWIDTH-1:0] DATA_IN,
    output wire [SIZE_OUTPUT* BITWIDTH-1:0] DATA_OUT,
    output reg DATA_VALID
);

    wire last_module_ready;
    // --- Place signals above

    // --- Place modules above

    reg do_calc_dly;
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            DATA_VALID <= 1'd1;
            do_calc_dly <= 1'd0;
        end else begin
            do_calc_dly <= DO_CALC;
            DATA_VALID <= (DATA_VALID) ? !(DO_CALC && !do_calc_dly) : last_module_ready;
        end
    end

    assign last_module_ready = 1'd1;
    assign DATA_OUT = 'd0;

endmodule
