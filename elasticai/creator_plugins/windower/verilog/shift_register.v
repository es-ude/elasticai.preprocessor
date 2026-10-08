//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     12.01.2026 11:37:12
// Copied on: 	    §{date_copy_created}
// Module Name:     Module for Implementing a Shift Register in Hardware
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Not tested on hardware!
// Dependencies:    None
// Improvements:    None
// Parameters:      BITWIDTH - Number of bitwidth of each sample
//                  SAMPLES - Number of sample in the window
//
//////////////////////////////////////////////////////////////////////////////////


module SHIFT_REGISTER#(
    parameter integer BITWIDTH = 12,
    parameter integer SAMPLES = 2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_SHIFT,
    input wire [BITWIDTH-1:0] DATA_IN,
    output reg [BITWIDTH-1:0] DATA_OUT,
    output wire [BITWIDTH* SAMPLES-1:0] DATA_BUF,
    output reg DVALID
);
    reg first_run_done;
    reg do_shift_dly;
    reg [BITWIDTH-1:0] buffer [SAMPLES-1:0];

    // Slicing buffer array output vector
    genvar i0;
    for(i0 = 0; i0 < SAMPLES; i0 = i0 + 1) begin
        assign DATA_BUF[i0 * BITWIDTH+:BITWIDTH] = buffer[i0];
    end

    integer i1;
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            first_run_done <= 1'd0;
            do_shift_dly <= 1'd0;
            for(i1 = 0; i1 < SAMPLES; i1 = i1 + 1) begin
                buffer[i1] <= 'd0;
            end
            DATA_OUT <= 'd0;
            DVALID <= 1'd0;
        end else begin
            do_shift_dly <= DO_SHIFT && EN;
            if(DO_SHIFT && ~do_shift_dly) begin
                first_run_done <= 1'd1;
                buffer[0] <= DATA_IN;
                for(i1 = 1; i1 < SAMPLES; i1 = i1 + 1) begin
                    buffer[i1] <= buffer[i1-1];
                end
                DATA_OUT <= buffer[SAMPLES-1];
                DVALID <= 1'd0;
            end else begin
                first_run_done <= first_run_done;
                for(i1 = 0; i1 < SAMPLES; i1 = i1 + 1) begin
                    buffer[i1] <= buffer[i1];
                end
                DATA_OUT <= DATA_OUT;
                DVALID <= first_run_done;
            end
        end
    end
endmodule
