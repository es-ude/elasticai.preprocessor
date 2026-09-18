//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	28.08.2026 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Event Detection for signed data input (with hysterese)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Description:
// Processing:      Data applied on posedge clk
// Dependencies:
//
// State:		    Not tested!
// Improvements:    None
// Parameters:      BITWIDTH    --> Bitwidth of input data
//                  THR_ON      --> Window size for enabling the output state (THR + THR_ON <= DATA)
//                  THR_OFF     --> Window size for disabling the output state (THR - THR_OFF >= DATA)
//                  OUT_INVERT  --> Boolean for inverting the output value
//////////////////////////////////////////////////////////////////////////////////


module EVENTDETECTOR_HYSTERESE_SIGNED #(
    parameter integer BITWIDTH = 8,
    parameter integer THR_ON = 4,
    parameter integer THR_OFF = 4,
    parameter integer OUT_INVERT = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [BITWIDTH-1:0] DATA_IN,
    input wire signed [BITWIDTH-1:0] THR,
    output reg IS_EVNT,
    output wire DVALID
);

    assign DVALID = !DO_CALC && EN;
    reg calc_dly;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            IS_EVNT <= OUT_INVERT;
            calc_dly <= 1'b0;
        end else begin
            calc_dly <= DO_CALC;
            if (EN && DO_CALC && !calc_dly) begin
                if (IS_EVNT ^ OUT_INVERT) begin
                    IS_EVNT <= (DATA_IN >= (THR - THR_OFF)) ^ OUT_INVERT;
                end else begin
                    IS_EVNT <= (DATA_IN >= (THR + THR_ON)) ^ OUT_INVERT;
                end
            end else begin
                IS_EVNT <= IS_EVNT;
            end
        end
    end
endmodule
