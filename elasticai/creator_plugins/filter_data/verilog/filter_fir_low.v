//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     23.05.2024 17:07:43
// Copied on: 	    §{date_copy_created}
// Module Name:     FIR Filter, Simple Low-pass Filter
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Simple FIR Filter (Low-pass filter, 2nd order, corner frequency at fs/2)
// Processing:      Data applied on posedge clk
// Dependencies:    None
//
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input signed data
//////////////////////////////////////////////////////////////////////////////////


module FIR_SIMPLE_LOW#(
    parameter BITWIDTH = 6'd16
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [BITWIDTH-'d1:0] DATA_IN,
    output wire signed [BITWIDTH-'d1:0] DATA_OUT,
    output wire DVALID
);
    reg signed [BITWIDTH-'d1:0] dly_data;
    reg signed [BITWIDTH:0] sum;
    reg [1:0] do_calc_dly;
    reg first_run_done;
    wire do_process;
    assign DATA_OUT = sum[BITWIDTH-:BITWIDTH];

    assign do_process = ~do_calc_dly[1] && do_calc_dly[0];
    assign DVALID = first_run_done && ~do_process;

    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly <= 1'b0;
            dly_data <= 'd0;
            sum <= 'd0;
            first_run_done <= 1'd0;
        end else begin
            do_calc_dly <= {do_calc_dly[0], DO_CALC};
            dly_data <= (do_process) ? DATA_IN : dly_data;
            sum <= (do_process) ? DATA_IN + dly_data : sum;
            first_run_done <= (do_process) ? 1'd1 : first_run_done;
        end
    end
endmodule
