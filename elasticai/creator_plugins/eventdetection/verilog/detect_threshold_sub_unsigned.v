//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	28.08.2026 12:38:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Event Detection with threshold
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


module SIGNED_THRESHOLD#(
    parameter integer BITWIDTH = 8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [BITWIDTH-'d1:0] DATA_IN,
    input wire signed [BITWIDTH-'d1:0] THR,  //Threshold
    output reg IS_EVNT, //True, when event is detected
    output wire DVALID
);
    wire [BITWIDTH:0] DIFF;

    assign DIFF = {1'b0, DATA_IN} - {1'b0, THR};

    assign DVALID = DO_CALC;

    always @(posedge CLK_SYS) begin
        if (!RSTN) begin
            IS_EVNT <= 1'b0;
        end
        else if (EN) begin
            if (DO_CALC) begin
                IS_EVNT <= ~DIFF[BITWIDTH];
            end
            else begin
                // Kein neuer Berechnungsvorgang
                IS_EVNT <= IS_EVNT;
            end
        end
    end

endmodule