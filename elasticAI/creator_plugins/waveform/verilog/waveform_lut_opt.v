//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     LUT Generator for Storing/Calling Optimized (Quarter) Waveforms
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v1
// Description:     Digital Direct Syntheziser with Analog Signal Waveforms (§{LUTWIDTH} x §{BITWIDTH} bit)
//                  LUT_DATA and LUT_DATA_EXT must be sorted in reversed way
// Dependencies:    None
//
// State:		    Works! (System Test done: 07.11.2024 on Arty A7-35T with 22% usage)
// Improvements:    None
// Parameters:      BITWIDTH   --> Bitwidth of the output value (input is 1-bit smaller to have full range)
//                  LUTWIDTH   --> Length of LUT for saving waveform (quarter)
//                  WAIT_WIDTH --> Bitwidth for defining WAIT_CYC with external middleware
//                  SIGNED_OUT --> Datatype of the output is signed (true) or unsigned (false)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//`define ACCESS_EXTERNAL
//`define TRGG_EXTERNAL

// --- CODE FOR READING DATA FROM EXTERNAL
// wire [LUTWIDTH* (BITWIDTH-'d1)-'d1:0] LUT_DATA_EXT;
// assign LUT_DATA_EXT = §{LUT_DATA};

// --- CODE FOR SETTING WAIT CYCLES FROM EXTERNAL
// wire WAIT_CYC;
// assign [WAIT_WIDTH-'d1:0] WAIT_CYC = 'd4;


module LUT_WAVEFORM_OPT#(
	parameter BITWIDTH = 6'd6,
	parameter LUTWIDTH = 10'd9,
    `ifndef TRGG_EXTERNAL
        parameter WAIT_WIDTH = 6'd6,
    `endif
    parameter SIGNED_OUT = 1'b0
)(
	input wire CLK_SYS,
	input wire RSTN,
	input wire EN_FLAG,
	`ifdef TRGG_EXTERNAL
	    input wire TRGG_CNT,
    `else
        input wire [WAIT_WIDTH-'d1:0] WAIT_CYC,
    `endif
    `ifdef ACCESS_EXTERNAL
        input wire [(BITWIDTH-'d1)* LUTWIDTH - 'd1:0] LUT_DATA_EXT,
    `endif
	output wire [BITWIDTH-'d1:0] LUT_OUT,
	output wire LUT_END
);
    // --- Data definition
    `ifndef ACCESS_EXTERNAL
        localparam LUT_DATA = {5'd31, 5'd28, 5'd24, 5'd20, 5'd16, 5'd12, 5'd8, 5'd4, 5'd0};
    `endif

    // --- Registers for counting and controlling
    wire inc_cnt_pos;
    wire [$clog2(LUTWIDTH)-'d1:0] sel_adr;
    wire [BITWIDTH-'d2:0] lut_data_int [LUTWIDTH-'d1:0];

    reg state;
    reg [1:0] cnt_phase;
    reg [$clog2(LUTWIDTH)-'d1:0] cnt_wvf_pos;
    `ifndef TRGG_EXTERNAL
        reg [WAIT_WIDTH-'d1:0] cnt_wait;
        // --- Counter for Downsampling System Clock
        always@(posedge CLK_SYS) begin
            if(~(RSTN && state)) begin
                cnt_wait <= 'd0;
            end else begin
                cnt_wait <= (cnt_wait == WAIT_CYC-'d1) ? 'd0 : cnt_wait + 'd1;
            end
        end
        assign inc_cnt_pos = (cnt_wait == WAIT_CYC-'d1);
    `else
        reg trgg_cnt_dly;
        always@(posedge CLK_SYS or negedge RSTN) begin
            trgg_cnt_dly <= (!RSTN) ? 1'b0 : TRGG_CNT;
        end
        assign inc_cnt_pos = !trgg_cnt_dly && TRGG_CNT;
    `endif

    // --- Processing LUT data
    assign LUT_END = (cnt_wvf_pos == (LUTWIDTH-'d1)) && (cnt_phase == 2'd3);
    assign sel_adr = (cnt_phase == 2'd0 || cnt_phase == 2'd2) ? cnt_wvf_pos : LUTWIDTH - cnt_wvf_pos - 'd1;
    assign LUT_OUT = (SIGNED_OUT) ? ((cnt_phase == 2'd0) ? {1'd0, lut_data_int[sel_adr]} : ((cnt_phase == 2'd1) ? {1'd0, lut_data_int[sel_adr]} : ((cnt_phase == 2'd2) ? {1'd1, - lut_data_int[sel_adr]}                            : ((LUT_END) ? {1'd0, lut_data_int[sel_adr]} : {1'd1, - lut_data_int[sel_adr]})))) :
                                    ((cnt_phase == 2'd0) ? {1'd1, lut_data_int[sel_adr]} : ((cnt_phase == 2'd1) ? {1'd1, lut_data_int[sel_adr]} : ((cnt_phase == 2'd2) ? {1'd0, ({(BITWIDTH-'d1){1'd0}} - lut_data_int[sel_adr])}   : ((LUT_END) ? {1'd1, lut_data_int[sel_adr]} : {1'd0, ({(BITWIDTH-'d1){1'd0}} - lut_data_int[sel_adr])}))));

    genvar i0;
    for(i0 = 'd0; i0 < LUTWIDTH; i0 = i0 + 'd1) begin
        `ifdef ACCESS_EXTERNAL
            assign lut_data_int[i0] = LUT_DATA_EXT[i0*(BITWIDTH-'d1)+:(BITWIDTH-'d1)];
        `else
            assign lut_data_int[i0] = LUT_DATA[i0*(BITWIDTH-'d1)+:(BITWIDTH-'d1)];
        `endif
    end

    //--- Counter for Quarter Wave Reading (Symmetric)
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            cnt_phase <= 2'd0;
            cnt_wvf_pos <= 'd0;
            state <= 1'd0;
        end else begin
            state <= (EN_FLAG) ? 1'd1 :
                    ((cnt_wvf_pos == LUTWIDTH-'d1 && cnt_phase == 2'd3 && inc_cnt_pos) ? 1'd0 :
                    state);
            if(inc_cnt_pos && state) begin
                cnt_phase   <= cnt_phase + ((cnt_wvf_pos == LUTWIDTH-'d1) ? 2'd1 : 2'd0);
                cnt_wvf_pos <= (cnt_wvf_pos == (LUTWIDTH-'d1)) ?
                                    ((cnt_phase == 2'd3) ? ((EN_FLAG) ? 'd1 : 'd0) : 'd1) :
                                    cnt_wvf_pos + 'd1;
            end else begin
                cnt_phase <= cnt_phase;
                cnt_wvf_pos <= cnt_wvf_pos;
            end
        end
    end
endmodule
