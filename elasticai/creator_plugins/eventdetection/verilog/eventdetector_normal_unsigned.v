//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	28.08.2026 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Event Detection for unsigned data input (normal comparison)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:
// Processing:      Data applied on posedge clk
// Dependencies:
//
// State:		    Not tested!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module EVENTDETECTION_NORMAL_UNSIGNED#(
    parameter integer BITWIDTH = 8,
    parameter integer OUT_INVERT = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    input wire [BITWIDTH-'d1:0] THR,
    output reg IS_EVNT,
    output reg DVALID
);

    reg calc_dly;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            calc_dly <= 1'b0;
            IS_EVNT <= OUT_INVERT;
            DVALID <= 1'b0;
        end else begin
            calc_dly <= DO_CALC;
            if (!calc_dly && DO_CALC && EN) begin
                IS_EVNT <= (DATA_IN >= THR) ^ OUT_INVERT;
                DVALID <= 1'd1;
            end else begin
                IS_EVNT <= IS_EVNT;
                DVALID <= 1'd0;
            end
        end
    end
endmodule
