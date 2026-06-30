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
    parameter BITWIDTH = 6'd12,
    parameter SAMPLES = 6'd2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_SHIFT,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH-'d1:0] DATA_OUT,
    output wire [BITWIDTH* SAMPLES-'d1:0] DATA_BUF,
    output wire DVALID
);
    reg first_run_done;
    reg [1:0] do_shift_dly;
    reg [BITWIDTH-'d1:0] buffer [SAMPLES-'d1:0];

    // Slicing buffer array output vector
    genvar i0;
    for(i0 = 'd0; i0 < SAMPLES; i0 = i0 + 'd1) begin
        assign DATA_BUF[i0 * BITWIDTH+:BITWIDTH] = buffer[i0];
    end
    assign DVALID = first_run_done && !DO_SHIFT;

    // Trigger rising edge
    wire do_sampling;
    assign do_sampling = (DO_SHIFT && ~do_shift_dly[0]);

    integer i1;
    always@(posedge CLK_SYS) begin
        if(~RSTN && ~EN) begin
            first_run_done <= 1'd0;
            do_shift_dly <= 2'd0;
            for(i1 = 'd0; i1 < SAMPLES; i1 = i1 + 'd1) begin
                buffer[i1] <= 'd0;
            end
            DATA_OUT <= 'd0;
        end else begin
            do_shift_dly <= {do_shift_dly[0], DO_SHIFT};
            if(do_sampling) begin
                first_run_done <= 1'd1;
                buffer[0] <= DATA_IN;
                for(i1 = 'd1; i1 < SAMPLES; i1 = i1 + 'd1) begin
                    buffer[i1] <= buffer[i1-'d1];
                end
                DATA_OUT <= buffer[SAMPLES-'d1];
            end else begin
                first_run_done <= first_run_done;
                for(i1 = 'd0; i1 < SAMPLES; i1 = i1 + 'd1) begin
                    buffer[i1] <= buffer[i1];
                end
                DATA_OUT <= DATA_OUT;
            end
        end
    end
endmodule
