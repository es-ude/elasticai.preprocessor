//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     12.01.2026 15:44:31
// Copied on: 	    §{date_copy_created}
// Module Name:     Module for Implementing a Ring Register in Hardware
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


module RING_BUFFER#(
    parameter integer BITWIDTH = 12,
    parameter integer SAMPLES = 2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_SHIFT,
    input wire [BITWIDTH-1:0] DATA_IN,
    output reg [BITWIDTH-1:0] DATA_OUT,
    output reg [BITWIDTH* SAMPLES-1:0] DATA_BUF,
    output reg DVALID
);
    reg first_run_done;
    reg do_shift_dly;
    reg [$clog2(SAMPLES)-1:0] cnt;
    reg [BITWIDTH-1:0] buffer [SAMPLES-1:0];

    // Slicing buffer array output vector (backward indexing)
    integer i0, idx;
    always_comb begin
    for (i0 = 0; i0 < SAMPLES; i0 = i0 + 1) begin
        idx = cnt - i0 - 1;
        if (idx < 0) begin
            idx = idx + SAMPLES;
        end
        DATA_BUF[i0*BITWIDTH +: BITWIDTH] = buffer[idx];
        end
    end

    // Processing
    integer i1;
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            first_run_done <= 1'd0;
            cnt <= 'd0;
            do_shift_dly <= 1'd0;
            for(i1 = 0; i1 < SAMPLES; i1 = i1 + 1) begin
                buffer[i1] <= 'd0;
            end
            DATA_OUT <= 'd0;
            DVALID <= 1'd0;
        end else begin
            do_shift_dly <= DO_SHIFT;
            if(~do_shift_dly && DO_SHIFT && EN) begin
                first_run_done <= 1'd1;
                cnt <= (cnt == SAMPLES-1) ? 'd0 : cnt + 'd1;
                buffer[cnt] <= DATA_IN;
                DATA_OUT <= buffer[cnt];
                DVALID <= 1'd0;
            end else begin
                first_run_done <= first_run_done;
                cnt <= cnt;
                for(i1 = 0; i1 < SAMPLES; i1 = i1 + 1) begin
                    buffer[i1] <= buffer[i1];
                end
                DATA_OUT <= DATA_OUT;
                DVALID <= first_run_done;
            end
        end
    end
endmodule
