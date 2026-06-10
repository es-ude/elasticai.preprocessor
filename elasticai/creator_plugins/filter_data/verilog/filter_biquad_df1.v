//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     15.10.2024 13:26:51
// Copied on: 	    §{date_copy_created}
// Module Name:     Biquad (2nd Order IIR) Filter with pipelined and parallized MAC operator
// Target Devices:  ASIC (Using MAC_LUT for multiplication)
//                  FPGA (Using MAC_DSP for multiplication)
// Tool Versions:   2v2
// Description:     Structure: Direct Form 1, signed integer operation
// Processing:      Data applied on posedge clk
// Dependencies:    mutArrayS with custom-made multiplier (for ASIC)
// 
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////
//`define USE_EXT_WEIGHTS
//`define USE_EXT_MAC

// Input values are signed integers with size of BITWIDTH (no fixed point)
// Internal operation with signed values and all weights have fraction width of BITWIDTH-2 [-2., +2.);
module BIQUAD_DF1#(
    parameter BITWIDTH = 6'd8,
    parameter NUM_MULT = 3'd1
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
	`ifdef USE_EXT_WEIGHTS
	    // Filter coefficients input (b0, b1, b2, -a1, -a2)
		input wire signed ['d5* BITWIDTH-'d1:0] FILT_WEIGHTS,
	`endif
	`ifdef USE_EXT_MAC
	    output wire signed ['d5* BITWIDTH-'d1:0] MAC_INPUT_A,   
	    output wire signed ['d5* BITWIDTH-'d1:0] MAC_INPUT_B,
	    output wire MAC_START_FLAG,
	    input wire signed ['d2* BITWIDTH-'d1:0] MAC_OUTPUT,
	    input wire MAC_DVALID,
	`endif
    input wire signed [BITWIDTH-'d1:0] DATA_IN,
    output wire signed [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    localparam UPPER_MASK = 2*BITWIDTH - 'd3;
    localparam STATE_IDLE = 2'd0, STATE_MAC = 2'd1, STATE_CALC = 2'd2, STATE_TAPS = 2'd3;
    // --- Filter coefficients input (b0, b1, b2, -a1, -a2), a0 is ignored due to 1
    localparam signed ['d5* BITWIDTH-'d1:0] FILT_COEFFS = { 8'sd19 , 8'sd37, 8'sd19, 8'sd0 , -8'sd11 };

    //################## Internal signals ##################
    reg first_run_done;
    reg do_calc_dly;
    reg [1:0] state;
    reg signed [BITWIDTH-'d1:0] tap_input [1:0], tap_output [1:0];

    wire mac_dvalid_int;
    wire ['d5* BITWIDTH-'d1:0] filt_data, filt_coeff;
    wire signed [2* BITWIDTH-'d1:0] mac_out;
    
    assign DATA_OUT = tap_output[0];
    assign DVALID = mac_dvalid_int && first_run_done;
   
    //################## Choosing the multiplier module ##################
    // --- Control signals and data flow to Multiplier
    //assign filt_data = {tap_output[1], tap_output[0], tap_input[1], tap_input[0], DATA_IN[BITWIDTH-'d1:0]};
    assign filt_data = {DATA_IN[BITWIDTH-'d1:0], tap_input[0], tap_input[1], tap_output[0], tap_output[1]};
    `ifdef USE_EXT_WEIGHTS
        assign filt_coeff = FILT_WEIGHTS;
    `else
        assign filt_coeff = FILT_COEFFS;
    `endif

    //Choicing the multiplier module
    // Using DSP-based MAC Operator (incl. Pipelining and Parallelisation)
    `ifndef USE_EXT_MAC
        MAC#(BITWIDTH, 'd5, NUM_MULT) MAC(
            .CLK_SYS(CLK_SYS),
            .RSTN(RSTN),
            .EN(EN),
            .DO_CALC(state == STATE_MAC),
            .IN_BIAS({(BITWIDTH){1'b0}}),
            .IN_WEIGHTS(filt_coeff),
            .IN_DATA(filt_data),
            .OUT_DATA(mac_out),
            .DATA_RDY(mac_dvalid_int)
        );
     `else
        assign MAC_INPUT_A = filt_coeff;
        assign MAC_INPUT_B = filt_data;
        assign MAC_START_FLAG = (state == STATE_MAC);
        assign mac_out = MAC_OUTPUT;
        assign mac_dvalid_int = MAC_DVALID;
     `endif

    integer i0;
    //Control-Structure
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly <= 1'd0;
            state <= STATE_IDLE;
            first_run_done <= 1'd0;
            for(i0 = 'd0; i0 < 'd2; i0 = i0 + 'd1) begin
                tap_input[i0] <= 'd0;
                tap_output[i0] <= 'd0;
            end
        end else begin
            do_calc_dly <= DO_CALC;
            case(state)
                STATE_IDLE: begin
                    state = (DO_CALC && ~do_calc_dly) ? STATE_MAC : STATE_IDLE;
                end
                STATE_MAC: begin
                    state <= (!mac_dvalid_int) ? STATE_CALC : STATE_MAC;
                end
                STATE_CALC: begin
                    state <= (mac_dvalid_int) ? STATE_TAPS : STATE_CALC;
                end
                STATE_TAPS: begin
                    tap_input[0] <= DATA_IN[BITWIDTH-'d1:0];
                    tap_input[1] <= tap_input[0];
                    tap_output[0] <= mac_out[UPPER_MASK-:BITWIDTH];
                    tap_output[1] <= tap_output[0];
                    first_run_done <= 1'd1;
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
endmodule
