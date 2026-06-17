//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     BlockRam Generator for Storing/Calling Full Waveforms
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v1
// Description:     Digital Direct Syntheziser with Analog Signal Waveforms (§{BITWIDTH} x §{RAMWIDTH} bit)
// Dependencies:    None
//
// State:		    Works! (System Test done: 07.11.2024 on Arty A7-35T with 22% usage)
// Improvements:    None
// Parameters:      BITWIDTH        --> Bitwidth of the output value
//                  RAMWIDTH        --> Length of RAM for saving waveform
//                  WAIT_WIDTH      --> Bitwidth for defining WAIT_CYC with external middleware
//////////////////////////////////////////////////////////////////////////////////
//`define TRGG_EXTERNAL

// --- CODE FOR SETTING WAIT CYCLES FROM EXTERNAL
// wire WAIT_CYC;
// assign [WAIT_WIDTH-'d1:0] WAIT_CYC = 4;


module RAM_WAVEFORM_FULL#(
    parameter BITWIDTH = 6'd6,
    `ifndef TRGG_EXTERNAL
	   parameter WAIT_WIDTH = 10'd7,
	`endif
	parameter RAMWIDTH = 10'd23,
	parameter PATH2MEM = ""
)(
	input wire CLK_SYS,
	input wire RSTN,
	input wire EN_FLAG,
	`ifdef TRGG_EXTERNAL
	    input wire TRGG_CNT,
    `else
        input wire [WAIT_WIDTH-'d1:0] WAIT_CYC,
    `endif
    input wire RAM_WE,
    input wire [$clog2(RAMWIDTH)-'d1:0] RAM_ADR,
	input wire [BITWIDTH-'d1:0] RAM_IN,
	output wire [BITWIDTH-'d1:0] RAM_OUT,
	output wire RAM_END
);
    // --- Registers for counting and controlling
    wire inc_cnt_pos;
    reg state;
    reg [$clog2(RAMWIDTH)-'d1:0] cnt_wvf_pos;
    wire [$clog2(RAMWIDTH)-'d1:0] sel_ram_adr;
    assign sel_ram_adr = (RAM_WE) ? RAM_ADR : cnt_wvf_pos;
    
    // --- Processing LUT data
    assign RAM_END = (cnt_wvf_pos == (RAMWIDTH-'d1));
    BRAM_SINGLE#(BITWIDTH, RAMWIDTH, PATH2MEM) BRAM(
        .CLK_RAM(CLK_SYS),
        .EN(state || RAM_WE),
        .WE(RAM_WE),
        .ADR(sel_ram_adr),
        .DIN(RAM_IN),
        .DOUT(RAM_OUT)
    );

    // --- Control scheme for handling the read function
    `ifndef TRGG_EXTERNAL
        reg [WAIT_WIDTH-'d1:0] cnt_wait;
        // --- Counter for Downsampling System Clock
        always@(posedge CLK_SYS or negedge RSTN) begin
            if(~(RSTN && state)) begin
                cnt_wait <= 'd0;
            end else begin
                if(cnt_wait == WAIT_CYC-'d1) begin
                    cnt_wait <= 'd0;
                end else begin
                    cnt_wait <= cnt_wait + 'd1;
                end
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

    // --- Counter for Getting RAM value
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            cnt_wvf_pos <= 'd0;
            state = 'd0;
        end else begin
            state <= (EN_FLAG) ? 1'd1 : ((cnt_wvf_pos == RAMWIDTH-'d1 && inc_cnt_pos) ? 1'd0 : state);
            if(inc_cnt_pos && state) begin
                cnt_wvf_pos <= (cnt_wvf_pos == RAMWIDTH-'d1) ? ((EN_FLAG) ? 'd1 : 'd0) : cnt_wvf_pos + 'd1;
            end else begin
                cnt_wvf_pos <= (state) ? cnt_wvf_pos : 'd0;
            end
        end
    end
endmodule
