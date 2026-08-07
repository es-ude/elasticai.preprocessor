//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     LUT Generator for Storing/Calling Full Waveforms
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v2
// Description:     Digital Direct Syntheziser with Analog Signal Waveforms (§{LUTWIDTH} x §{BITWIDTH} bit)
// Dependencies:    None
//
// State:		    Works! (System Test done: 06.08.2026 on Lattice ice40up5k)
// Improvements:    None
// Parameters:      BITWIDTH   --> Bitwidth of the output value
//                  LUTWIDTH   --> Length of LUT for saving waveform
//                  WAIT_WIDTH  --> Bitwidth for defining WAIT_CYC with external middleware
//////////////////////////////////////////////////////////////////////////////////
//`define ACCESS_EXTERNAL
//`define TRGG_EXTERNAL

// --- CODE FOR READING DATA FROM EXTERNAL
// wire [LUTWIDTH* (BITWIDTH-'d1)-'d1:0] LUT_DATA_EXT;
// assign LUT_DATA_EXT = §{LUT_DATA};

// --- CODE FOR SETTING WAIT CYCLES FROM EXTERNAL
// wire WAIT_CYC;
// assign [WAIT_WIDTH-'d1:0] WAIT_CYC = 4;


module LUT_WAVEFORM_FULL#(
    parameter integer BITWIDTH = 6,
    `ifndef TRGG_EXTERNAL
        parameter integer WAIT_WIDTH = 6,
    `endif
	parameter integer LUTWIDTH = 23
)(
	input wire CLK_SYS,
	input wire RSTN,
	input wire EN_FLAG,	
	`ifdef TRGG_EXTERNAL
	    input wire TRGG_CNT,
    `else
        input wire [WAIT_WIDTH-'d1:0] WAIT_CYC,
    `endif
    output reg  TRGG_NEW_SAMPLE,
    `ifdef ACCESS_EXTERNAL
        input wire [BITWIDTH* LUTWIDTH - 'd1:0] LUT_DATA_EXT,
    `endif
    output wire [BITWIDTH-'d1:0] LUT_OUT,
	output wire LUT_END
);
    // --- Data definition
    `ifndef ACCESS_EXTERNAL
        localparam LUT_DATA = {6'd0, 6'd2, 6'd5, 6'd8, 6'd11, 6'd14, 6'd17, 6'd20, 6'd23, 6'd26, 6'd29, 6'd31, 6'd29, 6'd26, 6'd23, 6'd20, 6'd17, 6'd14, 6'd11, 6'd8, 6'd5, 6'd2, 6'd0};
    `endif

    reg state;
    // --- Registers for counting and controlling
    wire inc_cnt_pos;
    reg [$clog2(LUTWIDTH)-'d1:0] cnt_wvf_pos;
    `ifndef TRGG_EXTERNAL
        reg [WAIT_WIDTH-'d1:0] cnt_wait;
        // --- Counter for Downsampling System Clock
        always@(posedge CLK_SYS) begin
            if(!RSTN) begin
                cnt_wait <= 'd0;
            end else begin
                cnt_wait <= (!state || cnt_wait == WAIT_CYC-'d1) ? 'd0 : cnt_wait + 'd1;
            end
        end
        assign inc_cnt_pos = (cnt_wait == WAIT_CYC-'d1);
    `else
        reg trgg_cnt_dly;
        always@(posedge CLK_SYS) begin
            trgg_cnt_dly <= (!RSTN) ? 1'b0 : TRGG_CNT;
        end
        assign inc_cnt_pos = !trgg_cnt_dly && TRGG_CNT;
    `endif

    // --- Processing LUT data
    assign LUT_END = (cnt_wvf_pos == (LUTWIDTH-'d1));
    wire [BITWIDTH-'d1:0] lut_data_int [LUTWIDTH-'d1:0];
    assign LUT_OUT = lut_data_int[cnt_wvf_pos];

    genvar i0;
    for(i0 = 'd0; i0 < LUTWIDTH; i0 = i0 + 'd1) begin
        `ifdef ACCESS_EXTERNAL
            assign lut_data_int[i0] = LUT_DATA_EXT[i0*BITWIDTH+:BITWIDTH];
        `else
            assign lut_data_int[i0] = LUT_DATA[i0*BITWIDTH+:BITWIDTH];
        `endif
    end

    // --- Counter for Getting LUT Value
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            state <= 1'd0;
            TRGG_NEW_SAMPLE <= 1'b0;
            cnt_wvf_pos <= 'd0;

        end else begin
            state <= (EN_FLAG) ? 1'd1 : ((cnt_wvf_pos == LUTWIDTH-'d1 && inc_cnt_pos) ? 1'd0 : state);
            TRGG_NEW_SAMPLE <= inc_cnt_pos;
            if(inc_cnt_pos && state) begin
                cnt_wvf_pos <= (cnt_wvf_pos == LUTWIDTH-'d1) ? ((EN_FLAG) ? 'd1 : 'd0) : cnt_wvf_pos + 'd1;
            end else begin
                cnt_wvf_pos <= (state) ? cnt_wvf_pos : 'd0;
            end
        end
    end
endmodule
