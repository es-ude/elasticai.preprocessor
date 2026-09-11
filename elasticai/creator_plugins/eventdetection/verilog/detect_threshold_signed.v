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
    output wire IS_EVNT, //True, when event is detected
    output wire DVALID
);


    assign DVALID = ~DO_CALC;
    assign DATA_OUT = pre_out[$clog2(LENGTH)+:BITWIDTH];

    always @(posedge CLK_SYS) begin
        if (!(RSTN && EN)) begin
            IS_EVNT <= 1'b0;
            DVALID  <= 1'b0;
        end
        else if (DO_CALC) begin

            if (DATA_IN >= THR)
                IS_EVNT <= 1'b1;
            else
                IS_EVNT <= 1'b0;

            DVALID <= 1'b1;
        end
        else begin
            // Kein neuer Berechnungsvorgang
            IS_EVNT <= IS_EVNT;
            DVALID  <= 1'b0;
        end
    end


endmodule