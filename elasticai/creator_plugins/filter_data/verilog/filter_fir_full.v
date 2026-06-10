//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     15.10.2024 14:16:08
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR Filter (Full approximation)
// Target Devices:  ASIC (Using MAC_LUT for multiplication)
//                  FPGA (Using MAC_DSP for multiplication)
// Tool Versions:   2v2
// Description:     Performing a FIR filtering on FPGA with custom made filter coefficients (Full implementation)
// Processing:      Data applied on posedge clk
// Dependencies:    MAC_[DSP, LUT] incl. multiplier, RING_BUFFER
//
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input signed data
//                  LENGTH --> Length of used taps (=FIR filter order)
//                  NUM_MULT_FILE --> Number of used multiplier in MAC operator (>= 1)
//////////////////////////////////////////////////////////////////////////////////
//`define USE_EXT_WEIGHTS
//`define USE_EXT_MAC

// Input values are signed integers with size of BITWIDTH (no fixed point)
// Internal operation with signed values and all weights have fraction width of BITWIDTH-1 [-0.15, +1.);
module FIR_FULL#(
	parameter BITWIDTH = 6'd8,
	parameter LENGTH = 10'd11,
	parameter NUM_MULT = 4'd1
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
	// Filter coefficients input (b0, b1, b2, ..., bN)
	`ifdef USE_EXT_WEIGHTS
		input wire signed [LENGTH * BITWIDTH-'d1:0] FILT_WEIGHTS,
	`endif
    `ifdef USE_EXT_MAC
        output wire signed [LENGTH* BITWIDTH-'d1:0] MAC_INPUT_A,   
	    output wire signed [LENGTH* BITWIDTH-'d1:0] MAC_INPUT_B,
	    output wire MAC_START_FLAG,
	    input wire signed [2* BITWIDTH-'d1:0] MAC_OUTPUT,
	    input wire MAC_DVALID,
    `endif
    input wire signed [BITWIDTH-'d1:0] DATA_IN,
    output wire signed [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    // --- Control signals
    // Used filter coefficients
    localparam signed [LENGTH* BITWIDTH-'d1:0] FILT_COEFFS = {8'sd1, 8'sd4, 8'sd11, 8'sd23, 8'sd38, 8'sd50, 8'sd38, 8'sd23, 8'sd11, 8'sd4, 8'sd1};

    // Data flow to MAC operator
    wire dvalid_buffer, dvalid_mac;
    wire [LENGTH* BITWIDTH -'d1:0] mac_ina, mac_inb;
    wire signed [2* BITWIDTH-'d1:0] mac_out;
    assign DVALID = dvalid_mac && dvalid_buffer;
    assign DATA_OUT = (DVALID) ? mac_out[(2*BITWIDTH-'d2)-:BITWIDTH] : DATA_OUT;

    // Filter coefficients
    `ifdef USE_EXT_WEIGHTS
        assign mac_inb = FILT_WEIGHTS;
    `else
        assign mac_inb = FILT_COEFFS;
    `endif

    // --- Implementation of modules
    // Data buffer
    RING_BUFFER#(BITWIDTH, LENGTH) BUFFER(
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_SHIFT(DO_CALC),
        .DATA_IN(DATA_IN),
        .DATA_OUT(),
        .DATA_BUF(mac_ina),
        .DVALID(dvalid_buffer)
    );
    `ifdef USE_EXT_MAC
        // Handling external MAC operator
        assign MAC_START_FLAG = dvalid_buffer;
        assign MAC_INPUT_A = mac_ina;
        assign MAC_INPUT_B = mac_inb;
        assign mac_out = MAC_OUTPUT;
        assign dvalid_mac = MAC_DVALID;
    `else
        // Handling MAC operator (incl. Pipelining and Parallelisation)
        MAC#(BITWIDTH, LENGTH, NUM_MULT) MAC(
            .CLK_SYS(CLK_SYS),
            .RSTN(RSTN),
            .EN(EN),
            .DO_CALC(dvalid_buffer),
            .IN_BIAS({(BITWIDTH){1'b0}}),
            .IN_WEIGHTS(mac_inb),
            .IN_DATA(mac_ina),
            .OUT_DATA(mac_out),
            .DATA_RDY(dvalid_mac)
        );
    `endif
endmodule
