//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	19.07.2026
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR-based Moving Average Filter (Binary division)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Mean Total Average for signed input data
// Processing:      Data applied on posedge clk
// Dependencies:    
//
// State:		    
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  
//////////////////////////////////////////////////////////////////////////////////


module MEAN_AVERAGE#(
    parameter BITWIDTH = 6'd8,
    parameter COUNTWIDTH = 5'd8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire [BITWIDTH-'d1:0] DATA_IN,
    output reg [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);

    localparam SUMWIDTH = BITWIDTH + COUNTWIDTH;

    // --- Control Signals
    reg [1:0] do_calc_dly;
    reg first_run_done;
    reg [SUMWIDTH-1:0] sum;
    reg [COUNTWIDTH-1:0] count;
    wire do_process;

    assign do_process = ~do_calc_dly[1] && do_calc_dly[0];
    assign DVALID = first_run_done && ~do_process;

    // --- Performing computation
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly    <= 2'd0;
            first_run_done <= 1'b0;
            sum            <= 'd0;
            count          <= 'd0;
            DATA_OUT       <= 'd0;
        end else begin
            do_calc_dly <= {do_calc_dly[0], DO_CALC};
            if(do_process) begin 
                sum <= sum + DATA_IN;
                count <= count + 1'b1;
                first_run_done <= 1'b1;
                // Mittelwert inklusive aktuellem Sample
                DATA_OUT <= (sum + DATA_IN) / (count);
            end else begin
                sum <= sum;
                count <= count;
                first_run_done <= first_run_done;
                DATA_OUT <= DATA_OUT;
            end
        end
    end
endmodule
