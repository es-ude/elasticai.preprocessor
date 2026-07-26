//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	20.07.2026
// Copied on: 	    §{date_copy_created}
// Module Name:     Return Input
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Module returns a steady value
// Processing:      Data applied on posedge clk
// Dependencies:    LENGTH is only a logarithmic value (otherwise result is invalid)
//                  Internal operation with unsigned values and scaling weight has fraction width of bitwidth
//
// State:		    
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  LENGTH --> Length of used taps (=FIR filter order)
//////////////////////////////////////////////////////////////////////////////////


module STEADY_RETURN#(
    parameter BITWIDTH = 6'd8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    // --- Control Signals
    reg [1:0] do_calc_dly;
    reg first_run_done;
    wire do_process;

    assign do_process = ~do_calc_dly[1] && do_calc_dly[0];
    assign DVALID = first_run_done && ~do_process;

   always @(posedge CLK_SYS) begin
        if (!(RSTN && EN)) begin
            do_calc_dly    <= 2'b00;
            first_run_done <= 1'b0;
            DATA_OUT       <= 'd0;
        end
        else begin
            do_calc_dly <= {do_calc_dly[0], DO_CALC};

            if (do_process) begin
                DATA_OUT       <= DATA_IN;
                first_run_done <= 1'b1;
            end
        end
    end

endmodule
