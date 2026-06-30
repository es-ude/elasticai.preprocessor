//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     12.01.2026 22:01:08
// Copied on: 	    §{date_copy_created}
// Module Name:     Module for Implementing an event-based windower in Hardware
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Not tested in hardware!
// Dependencies:    RING_BUFFER or SHIFT_REGISTER for building a windower
// Improvements:    None
// Parameters:      BITWIDTH - Number of bitwidth of each sample
//                  SAMPLES - Number of sample in the window
//                  NUM_SHIFT - Number of moved samples to identicate ready
//
//////////////////////////////////////////////////////////////////////////////////


module EVENT_WINDOWER#(
    parameter BITWIDTH = 6'd12, 
    SAMPLES = 6'd2, 
    NUM_SHIFT = 6'd1
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_SHIFT,
    input wire IS_EVNT,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output wire [BITWIDTH* SAMPLES-'d1:0] DATA_OUT,
    output reg DVALID
);
    reg [$clog2(NUM_SHIFT):0] cnt;
    reg buffer_valid_dly;
    wire buffer_valid;
    //SHIFT_REGISTER#(BITWIDTH, SAMPLES) DATA(
    RING_BUFFER#(BITWIDTH, SAMPLES) DATA(
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_SHIFT(DO_SHIFT && IS_EVNT),
        .DATA_IN(DATA_IN),
        .DATA_BUF(DATA_OUT),
        .DVALID(buffer_valid)
    );
    always@(posedge CLK_SYS) begin
        if(~RSTN && ~EN && ~IS_EVNT) begin
            cnt <= 'd0;
            buffer_valid_dly <= 1'd0;
            DVALID <= 1'd0;
        end else begin
            cnt <= (cnt == SAMPLES-'d1) ? 'd0 : cnt + (
				(buffer_valid && ~buffer_valid_dly) ? 'd1 : 'd0);
            buffer_valid_dly <= buffer_valid;
            DVALID <= (cnt == SAMPLES-'d1);
        end
    end
endmodule